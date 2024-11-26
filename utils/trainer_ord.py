import math
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction

import numpy as np
from itertools import chain

from .arguments_init import get_data_args, get_training_args
from .utils import to_pt, logger_ids, get_key
from .trainer_base import BaseTrainer
from .trainer_sft import num_fns
from tokenize_ids import get_prompt_ids, get_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def tokenize_data(data, worker_id, args):
    tokenized_data = []
    for ex in tqdm(data, disable=worker_id != args.num_workers - 1):
        prompt_ids = get_prompt_ids(ex, args.tokenizer, args.prompt_type, True, args.system_prompt)
        responses_ids = [get_token_ids(response, args.tokenizer) 
                        for response in ex["response"]]
        responses_ids[-1].append(args.tokenizer.eos_token_id)
        if args.score_scale == [0, 1]:
            score = [[5, 10] if s == 1 else [0, 5] for s in ex["score"]]
        elif args.score_scale == [1, 10]:
            score = [[math.ceil(s) - 1, math.ceil(s)] for s in ex["score"]]
        else:
            raise NotImplementedError

        tokenized_data.append({
                                "prompt_id": ex["prompt_id"],
                                "prompt": prompt_ids, 
                                "response": responses_ids,
                                "score": score
                                })
    return tokenized_data

def get_inputs(examples, name, split):
    training_args = get_training_args()
    data_args = get_data_args()

    inputs = []
    for ex in examples:
        prompt = to_pt(ex["prompt"])
        response = [to_pt(r) for r in ex["response"]]

        if len(prompt) + sum(map(len, response)) > data_args.max_length:
            continue
        
        inputs.append({
            "prompt": ex["prompt"], 
            "response": ex["response"], 
            "score": ex["score"], 
        })

    if training_args.local_rank == 0:
        logger.info(f"{split} {name} dataset samples")
        for ex in inputs[:3]:
            logger_ids("prompt", ex["prompt"])
            responses = ex["response"]
            for idx, response in enumerate(responses):
                logger_ids("response " + str(idx), response)
            logger.info(f"<|score|> {ex["score"]}")

    return inputs

len_fn = lambda ex: len(ex["prompt"]) + sum(map(len, ex["response"]))

def pad_fn(inputs, pad_token_id=0):
    prompts = get_key(inputs, "prompt")
    responses = get_key(inputs, "response")
    scores = get_key(inputs, "score")

    p_lens = np.array(list(map(len, prompts)))
    tot_len = np.array([len(p) + sum(map(len, r)) for p, r in zip(prompts, responses)])
    max_len = tot_len.max()

    batch = {}

    batch["input_ids"] = torch.stack([
                                    F.pad(torch.cat([p, *r]), 
                                    pad=(0, max_len - l), 
                                    value=pad_token_id) 
                                for p, r, l in zip(prompts, responses, tot_len)])
    
    batch["labels"] = torch.stack([
                                    F.pad(torch.cat(list(chain(*zip([torch.ones((len(x)-1, 2)) * (-100) for x in r], [torch.tensor([x]) for x in s])))), 
                                    pad=(0, 0, p, max_len - l), 
                                    value=-100)
                                for p, r, s, l in zip(p_lens, responses, scores, tot_len)]).int()
    
    batch["attention_mask"] = torch.stack([
                                    F.pad(torch.ones(l, dtype=bool), 
                                    pad=(0, max_len - l))
                                for l in tot_len])
    
    batch["position_ids"] = torch.arange(0, max_len)[None].repeat(len(prompts), 1)
    batch["eff_seqlens"] = (batch["labels"][..., 0] != -100).int().sum(dim=1)

    for arg in inputs[0].keys():
        if arg in ["prompt", "response", "score"]:
            continue
        values = get_key(inputs, arg)
        batch[arg] = torch.tensor(values)

    return batch

num_fns = copy.deepcopy(num_fns)
num_fns.update({
    "batch_num_segs": lambda b: len(b["xpo_seqlens"]) // 2
})

class Metric:
    
    def __call__(self, eval_out: EvalPrediction):
        acc = eval_out.label_ids
        return {"acc": acc[:, 0].sum() / max(acc[:, 1].sum(), 1)}

# for deepspeed
class Trainer(BaseTrainer):

    step: int = 0

    def training_init(self):
        self.step = 0

    def training_step_init(self):
        self.step += 1
        self.step_log = {"acc": 0}

    def compute_train_loss(self, model, inputs: Dict[str, Tensor], batch_num_effect_tokens, batch_num_samples, batch_num_tokens):
        labels = inputs["labels"]
        down, up = labels.chunk(2, dim=-1)
        down = down.squeeze(-1)
        up = up.squeeze(-1)
        down = down[down != -100]
        up = up[up != -100]
        outputs = model(**inputs)
        loss = outputs.loss / batch_num_effect_tokens
        # logger.info(f"{outputs}")
        self.step_log["acc"] += ((down <= outputs.pred) & (outputs.pred < up)).float().sum() / batch_num_effect_tokens
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
        batch_num_samples = num_fns["batch_num_samples"](inputs)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            labels = inputs["labels"]
            down, up = labels.chunk(2, dim=-1)
            down = down.squeeze(-1)
            up = up.squeeze(-1)
            down = down[down != -100]
            up = up[up != -100]
            outputs = model(**inputs)
            loss = outputs.loss / batch_num_effect_tokens
            acc = ((down <= outputs.pred) & (outputs.pred < up)).float().sum().item()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, 
                torch.tensor([[acc / batch_num_samples, batch_num_effect_tokens / batch_num_samples]] * batch_num_samples, device=self.args.device), 
                torch.tensor([[acc / batch_num_samples, batch_num_effect_tokens / batch_num_samples]] * batch_num_samples, device=self.args.device))
    