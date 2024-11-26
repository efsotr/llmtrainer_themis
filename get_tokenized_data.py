# coding=utf-8

import sys
import json
import logging
import argparse
import importlib
from typing import Optional

from transformers import LlamaTokenizer, AutoTokenizer

from file_IO import can_create_file, safe_load, safe_save
from multi_process_ray import multi_process
from tokenize import get_eos

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

class Namespace(argparse.Namespace):
    model_path : str
    prompt_type : str
    chat_template: str
    training_type : int
    train_dataset_path : str
    train_save_path : str
    dev_dataset_path : str
    dev_save_path : str
    num_workers : int
    tokenizer : Optional[LlamaTokenizer] = None
    system_prompt : Optional[str]
    score_scale: str

    def init(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        tokenizer = self.tokenizer
        if self.prompt_type == "chat" and self.chat_template is not None:
            tokenizer.chat_template = self.chat_template
        
        _, tokenizer.eos_token = get_eos(tokenizer, self.prompt_type)
        logger.info(f"eos token: {tokenizer.eos_token}")

        if self.system_prompt is not None:
            self.system_prompt = json.load(open(self.system_prompt))["system_prompt"]

        if self.score_scale is not None:
            self.score_scale = list(map(float, self.score_scale.split("-")))
            assert self.score_scale in [[0, 1], [1, 10]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, required=True, choices=["completion", "chat"])
    parser.add_argument("--chat_template", type=str, default=None)
    parser.add_argument("--training_type", type=str, required=True, choices=["sft", "po", "xpo", "ord"])
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument("--train_save_path", type=str, default=None)
    parser.add_argument("--dev_dataset_path", type=str, default=None)
    parser.add_argument("--dev_save_path", type=str, default=None)
    parser.add_argument("--num_workers", '-n', type=int, default=32)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--score_scale", type=str, default=None)
    args = parser.parse_args(namespace=Namespace())
    args.init()

    module = importlib.import_module("utils.trainer_" + args.training_type)
    tokenize_data = module.tokenize_data

    if args.train_dataset_path:
        assert args.train_save_path is not None
        assert can_create_file(args.train_save_path)
        data = safe_load(args.train_dataset_path)
        tokenized_data = multi_process(data, tokenize_data, num_workers=args.num_workers, num_cpus=1, worker_ids=list(range(args.num_workers)), extra_args=(args,))
        formated_data = {}
        formated_data[args.training_type] = tokenized_data
        safe_save(formated_data, args.train_save_path)

    if args.dev_dataset_path:
        assert args.dev_save_path is not None
        assert can_create_file(args.dev_save_path)
        data = safe_load(args.dev_dataset_path)
        tokenized_data = multi_process(data, tokenize_data, num_workers=args.num_workers, num_cpus=1, worker_ids=list(range(args.num_workers)), extra_args=(args,))
        formated_data = {}
        formated_data[args.training_type] = tokenized_data
        safe_save(formated_data, args.dev_save_path)

