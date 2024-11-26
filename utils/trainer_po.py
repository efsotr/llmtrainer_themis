import copy
import logging
from typing import cast, Any, Callable, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction

import numpy as np

from .arguments_init import get_training_args, get_data_args
from .utils import get_key, to_pt, logger_ids
from .trainer_base import BaseTrainer
from .trainer_sft import num_fns
from optimized_module.crossentropy import EffCrossEntropy
from tokenize import get_prompt_ids, get_token_ids


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def tokenize_data(data, worker_id, args):
    tokenized_data = []
    for ex in tqdm(data, disable=worker_id != args.num_workers - 1):
        prompt_ids = get_prompt_ids(ex, args.tokenizer, args.prompt_type, True, args.system_prompt)
        chosen_ids = get_token_ids(ex["chosen"] + args.tokenizer.eos_token, args.tokenizer)
        rejected_ids = get_token_ids(ex["rejected"] + args.tokenizer.eos_token, args.tokenizer)

        tokenized_data.append({
                                "prompt_id": ex["prompt_id"],
                                "prompt": prompt_ids, 
                                "chosen": chosen_ids,
                                "rejected": rejected_ids,
                                # "chosen_score": ex["chosen_score"],
                                # "rejected_score": ex["rejected_score"]
                                })
    return tokenized_data

def get_inputs(examples, name, split):
    data_args = get_data_args()
    training_args = get_training_args()
    inputs = []
    for ex in examples:
        prompt = to_pt(ex["prompt"])
        chosen = to_pt(ex["chosen"])
        rejected = to_pt(ex["rejected"])

        if len(prompt) + len(chosen) + len(rejected) + 1 > data_args.max_length * 2:
            continue

        if len(chosen) == len(rejected) and (chosen == rejected).all():
            continue

        inputs.append({
            "prompt": prompt, 
            "chosen": chosen, 
            "rejected": rejected, 
            "prompt_id": ex["prompt_id"]
        })

    if training_args.local_rank == 0:
        logger.info(f"{split} {name} dataset samples")
        for ex in inputs[:3]:
            logger_ids("prompt", ex["prompt"])
            logger_ids("chosen", ex["chosen"])
            logger_ids("rejected", ex["rejected"])

    return inputs

len_fn = lambda ex: len(ex["prompt"]) + len(ex["chosen"]) + len(ex["rejected"]) + 1

def pad_fn(inputs, pad_token_id = 0):
    prompts = get_key(inputs, "prompt")
    chosens = get_key(inputs, "chosen")
    rejecteds = get_key(inputs, "rejected")

    p_lens = np.array(list(map(len, prompts)))
    c_lens = np.array(list(map(len, chosens)))
    r_lens = np.array(list(map(len, rejecteds)))
    tot_len = p_lens + c_lens + 1 + r_lens
    max_len = tot_len.max()
    batch : Dict[str, Tensor] = {}

    batch["input_ids"] = torch.stack([
                                    F.pad(torch.cat([p, c, p[-1:], r]), 
                                    pad=(0, max_len - l), 
                                    value=pad_token_id) 
                                for p, c, r, l in zip(prompts, chosens, rejecteds, tot_len)])
    
    batch["labels"] = torch.stack([
                                    F.pad(torch.cat([c, torch.tensor([-100]), r, torch.tensor([-100])]), 
                                    pad=(p - 1, max_len - l), 
                                    value=-100)
                                for p, c, r, l in zip(p_lens, chosens, rejecteds, tot_len)])
    
    batch["attention_mask"] = torch.stack([
                                    F.pad(torch.ones((l,), dtype=bool), 
                                    pad=(0, max_len - l))
                                for l in tot_len])
    
    batch["position_ids"] = torch.stack([torch.cat([
                                    torch.arange(0, p + c), 
                                    torch.arange(p - 1, p + r), 
                                    torch.zeros((max_len - l,), dtype=torch.long)])
                                for p, c, r, l in zip(p_lens, c_lens, r_lens, tot_len)])
    
    batch["start_id_q"] = torch.stack([F.pad(torch.cat([
                                    torch.ones((c+1,)), 
                                    torch.ones((r+1,)) * 2]), (p - 1, max_len - l))
                                for p, c, r, l in zip(p_lens, c_lens, r_lens, tot_len)]).int() 
    batch["end_id_k"] = batch["start_id_q"].clone()
    batch["end_id_k"][batch["end_id_k"] == 0] = 2
    batch["triple_seqlens"] = torch.tensor([[p, c, r] for p, c, r in zip(p_lens, c_lens, r_lens)])

    for arg in inputs[0].keys():
        if arg in ["prompt", "chosen", "rejected"]:
            continue
        values = get_key(inputs, arg)
        if arg == "prompt_id":
            batch[arg] = values
        elif arg == "ref_log_probs":
            batch[arg] = torch.cat(values)
        else:
            batch[arg] = torch.tensor(values)

    return batch

num_fns = copy.deepcopy(num_fns)

class Metric:
    
    def __call__(self, eval_out: EvalPrediction):
        seq_preds = eval_out.predictions 
        seq_log_probs = eval_out.label_ids

        metric = {}
        for stats, name in [(seq_preds, "seq_preds"), (seq_log_probs, "seq_log_probs")]:
            chosen = stats[:, 0]
            rejected = stats[:, 1]
            metric[name+"/accuracies"] = (chosen > rejected).mean()
            metric[name+"/margins"] = (chosen - rejected).mean()
            metric[name+"/chosen"] = chosen.mean()
            metric[name+"/rejected"] = rejected.mean()

        return metric

# for deepspeed
class Trainer(BaseTrainer):

    step: int = 0

    def training_init(self):
        self.step = 0
        if self.args.use_sampo:
            self.sampo_gen = torch.Generator(self.args.device).manual_seed(self.args.seed)

    def training_step_init(self):
        self.step += 1
        self.step_log = {}

        self.step_log["po_loss"] = 0.

        for name in ["seq_preds", "seq_log_probs"]:
            self.step_log[name+"/accuracies"] = 0.
            self.step_log[name+"/margins"] = 0.
            self.step_log[name+"/chosen"] = 0.
            self.step_log[name+"/rejected"] = 0.

    def get_seqends(self, inputs: Dict[str, Tensor]):
        seqlens = inputs["triple_seqlens"]
        cu_seqlens = torch.column_stack([seqlens[:, 1], seqlens[:, 2]]).reshape(-1)
        return cu_seqlens.cumsum(0) - 1
    
    def get_eff_seqlens(self, inputs: Dict[str, Tensor]):
        seqlens = inputs["triple_seqlens"]
        return seqlens[:, 1:].reshape(-1)

    def compute_train_loss(self, model, inputs: Dict[str, Tensor], batch_num_effect_tokens, batch_num_samples, batch_num_tokens):
        labels = inputs["labels"]
        labels = labels[labels != -100]

        if self.args.with_ref:
            ref_log_probs = inputs.pop("ref_log_probs")
        
        outputs = model(**inputs)

        loss = torch.tensor(0., device=self.args.device)
        log_probs = - cast(Tensor, EffCrossEntropy.apply(outputs.logits, labels.unsqueeze(-1))).view(-1).float()

        seqends = self.get_seqends(inputs)
        _seq_log_probs = F.pad(log_probs.detach().cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
        _seq_log_probs = _seq_log_probs[1:] - _seq_log_probs[:-1]

        if self.args.with_ref:
            log_probs = log_probs - ref_log_probs.float()

        if self.args.use_sampo:
            def get_mask(tot_len, tot_mask):
                return torch.zeros(tot_len, dtype=torch.bool, device=self.args.device).scatter_(
                    0, torch.randperm(tot_len, generator=self.sampo_gen, device=self.args.device)[:tot_mask], 1)

            response_lens = inputs["triple_seqlens"][:, 1:].tolist()
            mask = []
            eff_seqlens = []
            
            for c_len, r_len in response_lens:
                min_len = min(c_len, r_len)
                mask.extend([get_mask(c_len, c_len - min_len), get_mask(r_len, r_len - min_len)])
                eff_seqlens.extend([min_len, min_len])

            mask = torch.cat(mask).to(log_probs.device)
            eff_seqlens = torch.tensor(eff_seqlens, device=log_probs.device)
            log_probs = log_probs.masked_fill(mask, 0)
        else:
            eff_seqlens = self.get_eff_seqlens(inputs)

        seq_log_probs = F.pad(log_probs.cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
        seq_log_probs = seq_log_probs[1:] -  seq_log_probs[:-1]

        seq_preds = seq_log_probs + self.args.norm_by_len_plus_one

        if self.args.norm_by_len:
            seq_preds = seq_preds / eff_seqlens

        seq_preds = seq_preds * self.args.po_beta
        margins = seq_preds[::2] - seq_preds[1::2]
        po_loss = - F.logsigmoid(margins - self.args.po_gamma).sum() / batch_num_samples 
        loss += po_loss
        self.step_log["po_loss"] += po_loss.item()

        for stats, name in [(seq_preds.detach(), "seq_preds"), (_seq_log_probs, "seq_log_probs")]:
            chosen = stats[::2]
            rejected = stats[1::2]
            self.step_log[name+"/accuracies"] += (chosen > rejected).sum().item() / batch_num_samples
            self.step_log[name+"/margins"] += (chosen - rejected).sum().item() / batch_num_samples
            self.step_log[name+"/chosen"] += chosen.sum().item() / batch_num_samples
            self.step_log[name+"/rejected"] += rejected.sum().item() / batch_num_samples

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        batch_num_samples = num_fns["batch_num_samples"](inputs)
        model.eval()
        labels = inputs["labels"]
        labels = labels[labels != -100]
        if self.args.with_ref:
            ref_log_probs = inputs.pop("ref_log_probs")
        with torch.no_grad():
            outputs = model(**inputs)
            log_probs = - cast(Tensor, EffCrossEntropy.apply(outputs.logits, labels.unsqueeze(-1)))

            seqends = self.get_seqends(inputs)
            _seq_log_probs = F.pad(log_probs.detach().cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
            _seq_log_probs = _seq_log_probs[1:] -  _seq_log_probs[:-1]

            if self.args.with_ref:
                log_probs -= ref_log_probs

            if self.args.use_sampo:
                def get_mask(tot_len, tot_mask):
                    return torch.zeros(tot_len, dtype=torch.bool, device=self.args.device).scatter_(
                        0, torch.randperm(tot_len, generator=self.sampo_gen, device=self.args.device)[:tot_mask], 1)

                response_lens = inputs["triple_seqlens"][:, 1:].tolist()
                mask = []
                eff_seqlens = []
                
                for c_len, r_len in response_lens:
                    min_len = min(c_len, r_len)
                    mask.extend([get_mask(c_len, c_len - min_len), get_mask(r_len, r_len - min_len)])
                    eff_seqlens.extend([min_len, min_len])

                mask = torch.cat(mask).to(log_probs.device)
                eff_seqlens = torch.tensor(eff_seqlens, device=log_probs.device)
                log_probs = log_probs.masked_fill(mask, 0)
            else:
                eff_seqlens = self.get_eff_seqlens(inputs)

            seq_log_probs = F.pad(log_probs.cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
            seq_log_probs = seq_log_probs[1:] -  seq_log_probs[:-1]

            seq_preds = seq_log_probs + self.args.norm_by_len_plus_one
            if self.args.norm_by_len:
                seq_preds = seq_preds / eff_seqlens
            
            seq_preds = seq_preds * self.args.po_beta
            margins = seq_preds[::2] - seq_preds[1::2]
            loss = - F.logsigmoid(margins - self.args.po_gamma).sum() / batch_num_samples 
            
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, seq_preds.view(-1, 2), _seq_log_probs.view(-1, 2))
    