import logging
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Sized, Iterator

import torch
from torch import nn
from torch.utils.data import Sampler
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, instrument_w_nvtx
from deepspeed.runtime.fp16.unfused_optimizer import FP16_UnfusedOptimizer
from deepspeed.runtime import bf16_optimizer
from transformers.trainer import (
    Trainer, DataLoader, is_sagemaker_mp_enabled, has_length
)

from .arguments import MaxTrainingArguments
from optimized_module.recompute import frame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GroupRandomSampler(Sampler[int]):
    data_source: Sized
    group_size: int

    def __init__(self, data_source: Sized, group_size: int, generator=None) -> None:
        self.data_source = data_source
        self.group_size = group_size
        self.generator = generator
        assert len(data_source) % group_size == 0
        
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        group = torch.arange(self.group_size)

        group_ids = torch.randperm(n // self.group_size, generator=generator)
        for idx in group_ids:
            yield from (group + idx * self.group_size).tolist()
    def __len__(self) -> int:
        return len(self.data_source)

class BaseTrainer(Trainer):
    args : MaxTrainingArguments
    wrap_fisrt = True
    engine : DeepSpeedEngine
    step_log : Dict[str, Any] = {}
    num_fns: Dict[str, Callable] = []

    def training_step_init(self):
        self.step_log = {}

    def training_init(self):
        self.num_fns = {}

    def log_gpu_mem(self, state = ""):
        if self.args.local_rank == 0:
            logger.info(f"{state}  "
                        f"MA  {torch.cuda.memory_allocated(self.args.device) / 1024 ** 3:.4f} GiB    "
                        f"CA  {torch.cuda.memory_reserved(self.args.device) / 1024 ** 3: .4f} GiB    "
                        f"MAX_MA  {torch.cuda.max_memory_allocated(self.args.device) / 1024 ** 3: .4f} GiB    "
                        f"MAX_CA  {torch.cuda.max_memory_reserved(self.args.device) / 1024 ** 3: .4f} GiB    ")
            torch.cuda.reset_peak_memory_stats(self.args.device)

    def get_train_dataloader(self):
        _data_collator = self.data_collator
        self.data_collator = lambda batch: batch[0]
        dataloader = super().get_train_dataloader()
        self.data_collator = _data_collator
        return dataloader
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[Dict[str, int], List[Dict[str, torch.torch.Tensor]]]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # structure of inputs
        if self.wrap_fisrt:
            if not self.is_deepspeed_enabled:
                raise NotImplementedError
            self.wrap_fisrt = False
            self.engine : DeepSpeedEngine = self.accelerator.deepspeed_engine_wrapped.engine
            def empty_save_checkpoint(self, save_dir, tag=None, client_state={}, save_latest=True, exclude_frozen_parameters=False):
                return True
            self.engine.save_checkpoint = partial(empty_save_checkpoint, self.engine)
            optimizer = self.engine.optimizer
            try:
                optimizer.loss_scaler.raise_error_at_min_scale = False
                if self.args.local_rank == 0:
                    logger.info('set raise_error_at_min_scale to False')
            except:
                pass
            if isinstance(optimizer, DeepSpeedZeroOptimizer):
                def unscale_and_clip_grads(self, grad_groups_flat, total_norm):
                    # compute combined scale factor for this group
                    combined_scale = self.loss_scale
                    if self.clip_grad > 0.:
                        # norm is in fact norm*scale
                        clip = (total_norm / self.loss_scale) / self.clip_grad
                        clip = torch.clamp(clip, min=1.0)
                        combined_scale = clip * self.loss_scale
                    combined_scale = 1. / combined_scale

                    for grad in grad_groups_flat:
                        if isinstance(grad, list):
                            sub_partitions = grad
                            for g in sub_partitions:
                                g.data.mul_(combined_scale)
                        else:
                            grad.data.mul_(combined_scale)
                optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)
                def _has_inf_or_nan(x: torch.Tensor, j=None):
                    # inf_or_nan = ~x.isfinite().all()
                    inf_or_nan = ~x.sum(dtype=torch.float).isfinite()
                    return inf_or_nan.float()
                optimizer._has_inf_or_nan = _has_inf_or_nan
            elif isinstance(optimizer, bf16_optimizer.BF16_Optimizer):
                from deepspeed.runtime.utils import graph_process, graph_cache, get_global_norm_of_tensors, get_accelerator
                def clip_tensors_by_global_norm(input_tensors, max_norm=1.0, global_norm=None, mpu=None, eps=1e-6, use_graph=False):
                    """Clip list of tensors by global norm.
                    Args:
                        input_tensors: List of tensors to be clipped
                        global_norm (float, optional): Precomputed norm. Defaults to None.
                        mpu (optional): model parallelism unit. Defaults to None.
                        eps (float, optional): epsilon value added to grad norm. Defaults to 1e-6
                    Returns:
                        float: the global norm
                    """
                    if global_norm is None:
                        global_norm = get_global_norm_of_tensors(input_tensors, mpu=mpu, use_graph=use_graph)
                    clip_coef = global_norm / max_norm
                    if clip_coef > 1:
                        clip_coef = 1. / clip_coef
                        if use_graph:

                            def clip_tensors(_tensor_list, _clip_coef_tensor):
                                for t in _tensor_list:
                                    t.detach().mul_(_clip_coef_tensor)

                            if 'clip_coef_tensor' not in graph_cache:
                                # Alloc memory
                                graph_cache['clip_coef_tensor'] = torch.tensor(clip_coef,
                                                                            dtype=torch.float32).to(get_accelerator().device_name())
                            clip_coef_tensor = graph_cache['clip_coef_tensor']
                            clip_coef_tensor.copy_(torch.tensor(clip_coef, dtype=torch.float32))
                            graph_process(False, clip_tensors, input_tensors, clip_coef_tensor)

                        else:
                            for t in input_tensors:
                                t.detach().mul_(clip_coef)
                    return global_norm

                bf16_optimizer.clip_tensors_by_global_norm = clip_tensors_by_global_norm
            elif isinstance(optimizer, FP16_UnfusedOptimizer):
                def unscale_and_clip_grads(self, total_norm, apply_scale=True):
                    # compute combined scale factor for this group
                    combined_scale = self.cur_scale
                    if self.clip_grad > 0.:
                        # norm is in fact norm*scale
                        clip = (total_norm / self.cur_scale) / self.clip_grad
                        if clip > 1:
                            combined_scale = clip * self.cur_scale

                    if apply_scale:
                        for group in self.fp32_groups:
                            for param in group:
                                if param.grad is not None:
                                    param.grad.data.mul_(1. / combined_scale)

                    return combined_scale
                optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)
            elif isinstance(optimizer, DeepSpeedZeroOptimizer_Stage3):
                @instrument_w_nvtx
                def unscale_and_clip_grads(self, sub_group_id, total_norm):
                    # compute combined scale factor for this group
                    combined_scale = self.loss_scale
                    if self.clip_grad > 0.:
                        # norm is in fact norm*scale
                        clip = (total_norm / self.loss_scale) / self.clip_grad
                        clip = torch.clamp(clip, min=1.0)
                        combined_scale = clip * self.loss_scale

                    self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)

                optimizer.unscale_and_clip_grads = partial(unscale_and_clip_grads, optimizer)

            self.training_init()

        self.training_step_init()
        model.train()
        inputs = self._prepare_inputs(inputs)

        t_loss = 0

        self.nums = inputs["nums"]
        inputs = inputs["batch"]
        self.nums = dict((n, num) for n, num in self.nums.items())
        self.nums_div = dict((n, num / self.args.world_size) for n, num in self.nums.items())
        
        if len(inputs) > 1:
            self.engine.optimizer.gradient_accumulation_steps = 2
            self.engine.set_gradient_accumulation_boundary(False)
            for _ in range(len(inputs) - 1):
                frame.clear()
                loss = self.compute_train_loss(model, inputs[_], **self.nums_div)
                self.accelerator.backward(loss)
                t_loss += loss.item()

            self.engine.set_gradient_accumulation_boundary(True)
        else:
            self.engine.optimizer.gradient_accumulation_steps = 1

        frame.clear()
        loss = self.compute_train_loss(model, inputs[-1], **self.nums_div)
        self.accelerator.backward(loss)
        t_loss += loss.item()

        if self.step_log:
            vs = torch.tensor(list(self.step_log.values()), dtype=torch.float32, device=self.args.device)
            vs = self.accelerator.reduce(vs, reduction="mean").tolist()
            self.step_log = dict(zip(self.step_log.keys(), vs))

        return torch.tensor(t_loss).to(self.args.device)
    
    def evaluate(self, *args, **kwargs):
        torch.cuda.empty_cache()
        return super(BaseTrainer, self).evaluate(*args, **kwargs)

    def num_tokens(self, train_dl: DataLoader, max_steps: Optional[int] = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        try:
            for step, batchs in enumerate(train_dl):
                tokens = sum(batch['input_ids'].numel() for batch in batchs['batch'])
                train_tokens += tokens
            if max_steps is not None:
                return train_tokens / len(train_dl) * max_steps
            return train_tokens
        except KeyError:
            logger.warning("Cannot get num_tokens from dataloader")
            return train_tokens
        
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 5)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}, **self.step_log, **self.nums}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, {**logs, **self.step_log, **self.nums})

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            opt_model_named_parameters = dict(opt_model.named_parameters())
            if "score.weight" in opt_model_named_parameters:
                logger.info("score in model")
                param_score = opt_model_named_parameters.pop("score.weight")
            else:
                param_score = None

            if "cutpoints" in opt_model_named_parameters:
                logger.info("cutpoints in model")
                param_cutpoints = opt_model_named_parameters.pop("cutpoints")
            else:
                param_cutpoints = None

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model_named_parameters.items() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model_named_parameters.items() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                }
            ]
            
            if param_score is not None:
                optimizer_grouped_parameters.append({
                    "params": [param_score], 
                    "weight_decay": 0.0,
                    "lr": self.args.value_head_learning_rate
                })

            if param_cutpoints is not None:
                optimizer_grouped_parameters.append({
                    "params": [param_cutpoints], 
                    "weight_decay": 0.0,
                    "lr": self.args.cutpoints_learning_rate
                })

            optimizer_grouped_parameters = [param_group for param_group in optimizer_grouped_parameters if param_group["params"] != []]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, opt_model)

            if "lr" in optimizer_kwargs:
                optimizer_kwargs.pop("lr")

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if self.args.local_rank == 0:
                logger.info(f" self.optimizer number of params {[len(g['params']) for g in self.optimizer.param_groups]}")
                logger.info(f" self.optimizer lr {[g['lr'] for g in self.optimizer.param_groups]}")
                logger.info(f" self.optimizer weight_decay {[g['weight_decay'] for g in self.optimizer.param_groups]}")

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        
        generator = torch.Generator().manual_seed(self.args.seed)
        return GroupRandomSampler(self.train_dataset, self.args.world_size, generator)