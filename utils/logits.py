import logging
from typing import List, Dict, Union
from itertools import chain

import torch
from torch import Tensor

from .arguments_init import get_data_args, get_training_args
from .file_IO import safe_load

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_logits(
    train_datasets: List[Dict[str, Union[List[Dict[str, Tensor]], Dict[str, int]]]], 
    eval_datasets=None
):
    training_args = get_training_args()
    data_args = get_data_args()

    if training_args.with_ref and training_args.training_type in ["po"]:
        logger.info("get ref logits")
        logits_map = safe_load(data_args.ref_cache_path)
        for batch in train_datasets:
            for inputs in batch["batch"]:
                prompt_ids = inputs["prompt_id"]

                ref_log_probs = []
                for prompt_id in prompt_ids:
                    ref_log_probs += list(logits_map[prompt_id])

                inputs["ref_log_probs"] = torch.cat(ref_log_probs)
                assert len(inputs["ref_log_probs"]) == (inputs["labels"] != -100).sum().item()
                
        if eval_datasets is not None and training_args.training_type == "po":
            for eval_dataset in eval_datasets.values():
                for ex in eval_dataset:
                    ref_log_probs = logits_map[ex["prompt"]]
                    ex["ref_log_probs"] = torch.cat(ref_log_probs)
                    assert len(ref_log_probs[0]) == len(ex["chosen"])
                    assert len(ref_log_probs[1]) == len(ex["rejected"])
