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
from optimized_module.crossentropy import EffCrossEntropy
from tokenize_ids import get_prompt_ids, get_token_ids


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def tokenize_data(data, worker_id, args):
    tokenized_data = []
    for ex in tqdm(data, disable=worker_id != args.num_workers - 1):
        prompt_ids = get_prompt_ids(ex, args.tokenizer, args.prompt_type, True, args.system_prompt)
        responses_ids = [get_token_ids(response, args.tokenizer) 
                        for response in chain(*zip(ex["chosen"], ex["rejected"]))]

        tokenized_data.append({
                                "prompt_id": ex["prompt_id"],
                                "prompt": prompt_ids, 
                                "response": responses_ids,
                                })
    return tokenized_data

def get_inputs(examples, name, split):
    training_args = get_training_args()
    data_args = get_data_args()

    inputs = []
    for ex in examples:
        prompt = to_pt(ex["prompt"])
        response = [to_pt(r) for r in ex["response"]]

        if len(prompt) + sum(map(len, response)) + len(response) // 2 > data_args.max_length * 2:
            continue
        
        inputs.append({
            "prompt": prompt, 
            "response": "response", 
            "prompt_id": ex["prompt_id"]
        })

    if training_args.local_rank == 0:
        logger.info(f"{split} {name} dataset samples")
        for ex in inputs[:3]:
            logger_ids("prompt", ex["prompt"])
            responses = ex["response"]
            for idx, response in enumerate(responses):
                logger_ids("response " + str(idx), response)

    return inputs

len_fn = lambda ex: len(ex["prompt"]) + sum(map(len, ex["response"])) + len(ex["response"]) // 2

def pad_fn(inputs, pad_token_id=0):
    prompts = get_key(inputs, "prompt")
    responses = get_key(inputs, "response")

    p_lens = np.array(list(map(len, prompts)))
    tot_len = np.array([len(p) + sum(map(len, r)) + len(r) // 2 for p, r in zip(prompts, responses)])
    max_len = tot_len.max()

    batch = {}
    ignore_idx = torch.tensor([-100])
    batch["input_ids"] = torch.stack([
                                    F.pad(torch.cat([p, r[1], p[-1:], r[0]] + list(chain(*[[r[i+1], r[i-2][-1:], r[i]] for i in range(2, len(r), 2)]))), 
                                    pad=(0, max_len - l), 
                                    value=pad_token_id) 
                                for p, r, l in zip(prompts, responses, tot_len)])
    batch["labels"] = torch.stack([
                                    F.pad(torch.cat(list(chain(*[[r[i+1], ignore_idx, r[i]] for i in range(0, len(r), 2)])) + [ignore_idx]), 
                                    pad=(p - 1, max_len - l), 
                                    value=pad_token_id) 
                                for p, r, l in zip(p_lens, responses, tot_len)])
    batch["attention_mask"] = torch.stack([
                                    F.pad(torch.ones((l,), dtype=bool), 
                                    pad=(0, max_len - l))
                                for l in tot_len])
    batch["position_ids"] = []
    batch["start_id_q"] = []
    max_len_r = max(map(len, responses))
    for p, r, l in zip(p_lens, responses, tot_len):
        pos = [torch.arange(p)]
        st_id = [torch.zeros(p-1)]

        tot = p
        for i in range(0, len(r), 2):
            c_len = len(r[i])
            r_len = len(r[i+1])
            pos += [torch.arange(tot, tot+r_len), torch.arange(tot-1, tot+c_len)]
            st_id += [torch.ones(r_len+1) * (i+1), torch.ones(c_len) * (i+2)]
            tot += c_len
        
        pad = torch.zeros(max_len - l, dtype=torch.long)
        st_id += [torch.tensor([len(r)]), pad]
        pos.append(pad)
        batch["position_ids"].append(torch.cat(pos))
        batch["start_id_q"].append(torch.cat(st_id))

    batch["position_ids"] = torch.stack(batch["position_ids"])
    batch["start_id_q"] = torch.stack(batch["start_id_q"]).int()
    batch["end_id_k"] = batch["start_id_q"].clone()
    batch["end_id_k"][batch["end_id_k"] & 1 == 0] = max_len_r

    batch["xpo_seqlens"] = torch.stack([torch.tensor(list(map(len, chain(r[1::2], r[::2]))) + [0] * (max_len_r - len(r))) for r in responses])

    for arg in inputs[0].keys():
        if arg in ["prompt", "response"]:
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
num_fns.update({
    "batch_num_segs": lambda b: len(b["xpo_seqlens"]) // 2
})

class Metric:
    
    def __call__(self, eval_out: EvalPrediction):
        seq_preds = eval_out.predictions[eval_out.predictions != -100].reshape((-1, 2))
        seq_log_probs = eval_out.label_ids[eval_out.label_ids != -100].reshape((-1, 2))

        metric = {}
        for stats, name in [(seq_preds, "seq_preds"), (seq_log_probs, "seq_log_probs")]:
            chosen = stats[:, 1]
            rejected = stats[:, 0]
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
        return inputs["xpo_seqlens"].cumsum(0) - 1
    
    def get_eff_seqlens(self, inputs: Dict[str, Tensor]):
        return inputs["xpo_seqlens"]

    def compute_train_loss(self, model, inputs: Dict[str, Tensor], batch_num_effect_tokens, batch_num_samples, batch_num_tokens, batch_num_segs):
        labels = inputs["labels"]
        labels = labels[labels != -100]

        if self.args.with_ref:
            seq_ref_log_probs = inputs.pop("seq_ref_log_probs")
        
        outputs = model(**inputs)

        loss = torch.tensor(0., device=self.args.device)
        neg_log_probs : Tensor = EffCrossEntropy.apply(outputs.logits, labels.unsqueeze(-1)).view(-1)

        seqends = self.get_seqends(inputs)
        seq_log_probs = F.pad(neg_log_probs.cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
        seq_log_probs = - (seq_log_probs[1:] -  seq_log_probs[:-1])

        seq_preds = seq_log_probs

        if self.args.with_ref:
            seq_preds = seq_preds - seq_ref_log_probs

        eff_seqlens = self.get_eff_seqlens(inputs)
        if self.args.norm_by_len:
            seq_preds = seq_preds / eff_seqlens

        seq_preds = seq_preds * self.args.po_beta
        margins = seq_preds[1::2] - seq_preds[::2]
        po_loss = - F.logsigmoid(margins - self.args.po_gamma).sum() / batch_num_segs 
        loss += po_loss
        self.step_log["po_loss"] += po_loss.item()

        for stats, name in [(seq_preds.detach(), "seq_preds"), (seq_log_probs.detach(), "seq_log_probs")]:
            chosen = stats[1::2]
            rejected = stats[::2]
            self.step_log[name+"/accuracies"] += (chosen > rejected).sum().item() / batch_num_segs
            self.step_log[name+"/margins"] += (chosen - rejected).sum().item() / batch_num_segs
            self.step_log[name+"/chosen"] += chosen.sum().item() / batch_num_segs
            self.step_log[name+"/rejected"] += rejected.sum().item() / batch_num_segs

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        
        inputs = self._prepare_inputs(inputs)
        batch_num_segs = num_fns["batch_num_segs"](inputs)
        model.eval()
        labels = inputs["labels"]
        labels = labels[labels != -100]
        if self.args.with_ref:
            seq_ref_log_probs = inputs.pop("seq_ref_log_probs").view(-1)
        with torch.no_grad():
            outputs = model(**inputs)
            neg_log_probs : Tensor = EffCrossEntropy.apply(outputs.logits, labels.unsqueeze(-1))

            seqends = self.get_seqends(inputs)
            seq_log_probs = F.pad(neg_log_probs.cumsum(0, dtype=torch.float)[seqends], (1, 0), value=0)
            seq_log_probs = - (seq_log_probs[1:] -  seq_log_probs[:-1])

            seq_preds = seq_log_probs

            if self.args.with_ref:
                seq_preds = seq_preds - seq_ref_log_probs

            eff_seqlens = self.get_eff_seqlens(inputs)
            if self.args.norm_by_len:
                seq_preds = seq_preds / eff_seqlens

            seq_preds = seq_preds * self.args.po_beta
            margins = seq_preds[1::2] - seq_preds[::2]
            loss = - F.logsigmoid(margins - self.args.po_gamma).sum() / batch_num_segs 
            
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, seq_preds.view(1, -1), seq_log_probs.view(1, -1))
    