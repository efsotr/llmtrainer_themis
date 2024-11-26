import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from transformers.trainer_utils import EvalPrediction

from .trainer_po import Trainer, get_inputs, len_fn, pad_fn, num_fns, tokenize_data

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Metric:
    
    def __call__(self, eval_out: EvalPrediction):
        predseqscore = eval_out.label_ids

        metric = {}
        for stats, name in [(predseqscore, "score")]:
            chosen = stats[:, 0]
            rejected = stats[:, 1]
            metric[name+"/accuracies"] = (chosen > rejected).mean()
            metric[name+"/margins"] = (chosen - rejected).mean()
            metric[name+"/chosen"] = chosen.mean()
            metric[name+"/rejected"] = rejected.mean()

        return metric

# for deepspeed
class Trainer(Trainer):

    step: int = 0

    def training_init(self):
        self.step = 0

    def training_step_init(self):
        self.step += 1
        self.step_log = {}

        self.step_log["score_loss"] = 0.

        for name in ["score"]:
            self.step_log[name+"/accuracies"] = 0.
            self.step_log[name+"/margins"] = 0.
            self.step_log[name+"/chosen"] = 0.
            self.step_log[name+"/rejected"] = 0.

    def get_rm_seqends(self, inputs: Dict[str, Tensor]):
        seqlens = inputs["triple_seqlens"]
        cu_seqlens = torch.column_stack([seqlens[:, 1] + seqlens[:, 0], seqlens[:, 2] + 1]).reshape(-1)
        return cu_seqlens.cumsum(0) - 1

    def compute_train_loss(self, model, inputs: Dict[str, Tensor], batch_num_effect_tokens, batch_num_samples, batch_num_tokens):
        outputs = model(**inputs)

        seqends = self.get_rm_seqends(inputs)
        predseqscore : Tensor = outputs.logits.view(-1)[seqends].float()
        score_loss = - F.logsigmoid(self.args.rm_beta * (predseqscore[::2] - predseqscore[1::2]) - self.args.rm_gamma).sum() / batch_num_samples
        loss = score_loss
        self.step_log["score_loss"] += score_loss.item() 

        for stats, name in [(predseqscore.detach(), "score")]:
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
        with torch.no_grad():
            outputs = model(**inputs)
            seqends = self.get_rm_seqends(inputs)
            predseqscore : Tensor = outputs.logits.view(-1)[seqends].float()
            score_loss = - F.logsigmoid(self.args.rm_beta * (predseqscore[::2] - predseqscore[1::2]) - self.args.rm_gamma).sum() / batch_num_samples
            loss = score_loss

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, torch.empty((batch_num_samples, 1), device=self.args.device), predseqscore.view(-1, 2))
    