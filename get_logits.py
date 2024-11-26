# coding=utf-8

import os
import sys
import torch
import logging
from tqdm import tqdm
from typing import cast, Any, Mapping, Union

import torch.nn.functional as F
import torch.distributed as dist
from transformers import set_seed
from accelerate.utils.other import wait_for_everyone

from .utils import (
    init_args, get_model_args, get_data_args, get_training_args, 
    preprocess_data, load_model, load_tokenizer, get_tokenizer
)
from optimized_module.crossentropy import EffCrossEntropy
from file_IO import safe_load, safe_save
from collections import defaultdict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    init_args()

    training_args = get_training_args()
    data_args = get_data_args()

    assert training_args.training_type in ["po", "vm"]
    cache_path = data_args.ref_cache_path if training_args.training_type == "po" else data_args.adv_cache_path

    if training_args.local_rank == 0 :
        if os.path.exists(cache_path):
            # raise NotImplementedError("file exits")
            logger.info("file exits")
        else:
            torch.save({}, cache_path)
    wait_for_everyone()

    # load tokenizer
    set_seed(training_args.seed)
    load_tokenizer()

    # Get the datasets 
    def get_data():
        train_format_data = safe_load(data_args.formated_train_data_cache_path)
        dev_format_data = safe_load(data_args.formated_dev_data_cache_path)
        if train_format_data is None:
            train_format_data = {}
        for k in dev_format_data.keys():
            train_format_data["<|eval|> "+k] = dev_format_data[k]
        training_args.do_eval = False
        return preprocess_data((train_format_data, {}))
    train_datasets, _, _ = get_data()
    logger.info('Finish loading dataset')

    # Load model
    model = load_model()

    world_size = training_args.world_size
    rank = training_args.local_process_index
    avg_len = (len(train_datasets) + world_size - 1) // world_size
    train_datasets = train_datasets[avg_len*rank: avg_len*(rank+1)]

    def _prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: _prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(_prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(training_args.device)
        return data
    
    def get_cu_seqlens_vm(inputs):
        seqlens = inputs["triple_seqlens"]
        seqlens = torch.column_stack([seqlens[:, 0] + seqlens[:, 1], seqlens[:, 2] + 1]).reshape(-1)
        return F.pad(seqlens.cumsum(0), (1, 0))
    
    def get_cu_seqlens_po(inputs):
        seqlens = inputs["triple_seqlens"]
        seqlens = torch.column_stack([seqlens[:, 1], seqlens[:, 2]]).reshape(-1)
        return F.pad(seqlens.cumsum(0), (1, 0))
    
    logits_map = defaultdict(list)
    with torch.no_grad():
        model.eval()
        for batch in tqdm(train_datasets, disable=rank != 0):
            for inputs in batch["batch"]:
                inputs = _prepare_input(inputs)
                labels = cast(torch.Tensor, inputs["labels"])
                outputs = model(**inputs)

                logits = outputs.logits
                if training_args.training_type == "po":
                    labels = labels[labels != -100]
                    logits = - EffCrossEntropy.apply(logits, labels.unsqueeze(-1)).view(-1)
                else:
                    logits = logits[0]
                    labels = labels[0]

                prompt_id = inputs["prompt_id"]
                if training_args.training_type == "vm":
                    cu_seqlens = get_cu_seqlens_vm(inputs)
                    labels[cu_seqlens[1:] - 1] = 0
                    cu_seqlens = cu_seqlens.tolist()
                    for i in range(len(cu_seqlens) - 1):
                        sub = slice(cu_seqlens[i], cu_seqlens[i+1])
                        adv_logits = logits[sub][labels[sub] != -100]
                        logits_map[prompt_id[i // 2]].append((adv_logits[1:] - adv_logits[:-1]).cpu())

                elif training_args.training_type == "po":
                    cu_seqlens = get_cu_seqlens_po(inputs).tolist()
                    for i in range(len(cu_seqlens) - 1):
                        logits_map[prompt_id[i // 2]].append(logits[cu_seqlens[i]: cu_seqlens[i+1]].cpu())
    
    gathered_map = [None for _ in range(world_size)]
    dist.gather_object(logits_map, gathered_map if rank == 0 else None)
    if rank == 0:
        for i in range(1, world_size):
            logits_map.update(gathered_map[i])
        safe_save(logits_map, cache_path)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()

