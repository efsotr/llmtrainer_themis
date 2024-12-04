import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np

from .arguments_init import get_training_args, get_data_args, get_model_args
from .trainer_base import BaseTrainer
from .utils import logger_ids, to_pt, get_key, has_key
from optimized_module.crossentropy import EffCrossEntropy
from tokenize_ids import get_prompt_ids, get_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def tokenize_data(data, worker_id, args):
    tokenized_data = []
    response_format = "{Reason}\nRating: {Rating}"
    for ex in tqdm(data, disable=worker_id != args.num_workers - 1):
        prompt_ids = get_prompt_ids(ex, args.tokenizer, args.prompt_type, True, args.system_prompt)
        for ex_res in ex["response"]:
            response_ids = get_token_ids(response_format.format(Reason=ex_res["Reason"].strip(), Rating=ex_res["Rating"]) + args.tokenizer.eos_token, args.tokenizer)
            tokenized_data.append({"prompt": prompt_ids, "response": response_ids, "gate": ex.get("gate", None)})
    return tokenized_data

def get_inputs(examples, name, split):
    data_args = get_data_args()
    training_args = get_training_args()
    model_args = get_model_args()
    inputs = []
    for ex in examples:
        prompt = to_pt(ex["prompt"])
        response = to_pt(ex["response"])
        if len(prompt) > data_args.max_length - 16:
            continue
        if len(prompt) + len(response) > data_args.max_length:
            response = response[: data_args.max_length - len(prompt)]
        inputs.append({
            "prompt": prompt, 
            "response": response,
        })
        if model_args.use_lora_moe:
            inputs[-1]["gate"] = ex["gate"]

    if training_args.local_rank == 0:
        logger.info(f"{split} {name} dataset samples")
        for ex in inputs[:3]:
            logger_ids("prompt", ex["prompt"])
            logger_ids("response", ex["response"])

    return inputs

len_fn = lambda ex: len(ex["prompt"]) + len(ex["response"])

gate_mapping = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "a1": 4, "a2": 5, "a3": 6, "a4": 7}

def pad_fn(inputs, pad_token_id = 0):
    prompts = get_key(inputs, "prompt")
    responses = get_key(inputs, "response")
    p_lens = np.array(list(map(len, prompts)))
    r_lens = np.array(list(map(len, responses)))
    tot_len = p_lens + r_lens
    max_len = tot_len.max()
    batch = {}

    batch["input_ids"] = torch.stack([
                                    F.pad(torch.cat([p, r]), 
                                    pad=(0, max_len - l), 
                                    value=pad_token_id) 
                                for p, r, l in zip(prompts, responses, tot_len)])
    
    batch["labels"] = torch.stack([
                                    F.pad(r, 
                                    pad=(p, max_len - l + 1), 
                                    value=-100)
                                for p, r, l in zip(p_lens, responses, tot_len)])[:, 1:]
    
    batch["attention_mask"] = torch.stack([
                                    F.pad(torch.ones(l, dtype=bool), 
                                    pad=(0, max_len - l))
                                for l in tot_len])
    
    batch["position_ids"] = torch.arange(0, max_len)[None].repeat(len(prompts), 1)
    batch["eff_seqlens"] = (batch["labels"] != -100).int().sum(dim=1)
    if has_key(inputs, "gate"):
        seqlens = batch["attention_mask"].sum(1)
        seqends = F.pad(torch.cumsum(seqlens, 0), (1, 0))
        gates_idx = [[] for _ in range(8)]
        for i, gate in enumerate(get_key(inputs, "gate")):
            gate1 = gate_mapping[gate[:2]]
            gate2 = gate_mapping[gate[2:]]
            gates_idx[gate1].append(torch.arange(seqends[i], seqends[i+1]))
            gates_idx[gate2].append(torch.arange(seqends[i], seqends[i+1]))
        gates_idx = [torch.cat(ex) if ex != [] else torch.tensor([], dtype=torch.long) for ex in gates_idx]
        batch["gates_idx"] = gates_idx

    for arg in inputs[0].keys():
        if arg in ["prompt", "response", "gate"]:
            continue
        values = get_key(inputs, arg)
        if arg == "prompt_id":
            batch[arg] = values
        elif arg == "ref_log_probs":
            batch[arg] = torch.cat(values)
        else:
            batch[arg] = torch.tensor(values)

    return batch

num_fns = {
    "batch_num_effect_tokens": lambda b: torch.sum(b["labels"] != -100).item(), 
    "batch_num_tokens": lambda b: b["labels"].numel(), 
    "batch_num_samples": lambda b: b["position_ids"].eq(0).sum().item(), 
}

Metric = lambda: None

# for deepspeed
class Trainer(BaseTrainer):

    step: int = 0

    def training_init(self):
        self.step = 0

    def training_step_init(self):
        self.step += 1
        self.step_log = {}

    def compute_train_loss(self, model, inputs: Dict[str, Tensor], batch_num_effect_tokens, batch_num_samples, batch_num_tokens):
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = EffCrossEntropy.apply(outputs.logits, labels[labels != -100].unsqueeze(-1)).sum() / batch_num_effect_tokens
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        batch_num_effect_tokens = num_fns["batch_num_effect_tokens"](inputs)
        model.eval()
        labels = inputs["labels"]
        with torch.no_grad():
            outputs = model(**inputs)
            loss = EffCrossEntropy.apply(outputs.logits, labels[labels != -100].unsqueeze(-1)).sum() / batch_num_effect_tokens

        if prediction_loss_only:
            return (loss, None, None)

        # return (loss, logits, labels)
        return (loss, None, None)
    