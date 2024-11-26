import os
import sys
import json
import torch
import logging
import argparse
from typing import cast, List, Tuple
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(training_dir)

from cache import Cache
from file_IO import safe_load, safe_save
from tokenize_ids import get_prompt_ids, get_eos
from multi_process_ray import multi_process_stream, multi_process_decorator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="[%(asctime)s,%(msecs)d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class AllIn:
    def __contains__(self, x):
        return True


class Namespace(argparse.Namespace):
    model: str
    output_dir: str
    test_dirs: List[str]
    prefix_dir: str
    test_files: str
    prompt_type: str
    seed: int
    cache_dir: str
    stop: str = None

    ## vllm config
    sampling_params: str
    batch_size: int
    tp_size: int

    def init(self):        
        self.avail_test_files = set(file for file in self.test_files.split(',')) \
            if self.test_files is not None and self.test_files != "" else AllIn()
        
        self.test_dirs = [test_dir for test_dir in self.test_dirs.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Model name or path")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_dirs', required=True, type=str, help="Testset directory")
    parser.add_argument('--test_files', default=None, type=str, help="Testset filename")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory")
    parser.add_argument('--sampling_params',  default="./eval/gen_params/default.json", type=str, help="Vllm Sampling Parameters")
    parser.add_argument('--tp_size', '-tp', default=1, type=int)
    parser.add_argument('--prompt_type', required=True, type=str, choices=["chat", "completion"])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--prefix_dir', default=None, type=str)
    args = parser.parse_args(namespace=Namespace())
    args.init()

    logger.info(f"get_output_fast.py args: {repr(args)}")
    
    test_files = cast(list[Tuple[str, str]], [])
    for test_dir in args.test_dirs:
        for dirpath, _, files in os.walk(test_dir):
            for file in files:
                if (file.endswith(".json") or file.endswith(".json.gz")) and file in args.avail_test_files \
                    and not file.startswith("completions.") and not file.startswith("stats.") and not file.startswith("prefix."):
                    test_files.append((dirpath, file))

    test_files.sort()
    assert len(set(file for _, file in test_files)) == len(test_files), "File names in test_files must be unique"
    logger.info(f"Test files: {', '.join(file for _, file in test_files)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_test_prompts = []
    all_outputs = {}

    for dirpath, file in test_files:
        test_dataset = safe_load(os.path.join(dirpath, file))
        if args.prefix_dir:
            test_prefix = safe_load(os.path.join(args.prefix_dir, "prefix." + file))
            for ex, prefix in zip(test_dataset, test_prefix):
                ex["prefix"] = prefix
        assert isinstance(test_dataset, list) , "test_dataset must be of type list"
        assert len(test_dataset) > 0, "test_dataset must have a length greater than 0"
        assert len(test_dataset) == len(test_prefix)

        all_test_prompts.extend([
            (test_dataset[i], file, i)
            for i in range(0, len(test_dataset), args.batch_size)
        ])
        all_outputs[file] = [None for _ in range(len(test_dataset))]

    cache = Cache(args.cache_dir, args.batch_size)
    cached_outs, all_test_prompts = cache.get_cache(all_test_prompts, lambda ex: tuple(ex[1:]))

    num_workers = torch.cuda.device_count() // args.tp_size
    logger.info(f"num_workers: {num_workers}")

    @multi_process_decorator(multi_process_stream, num_workers=num_workers, num_gpus=args.tp_size, num_cpus=1, worker_ids=list(range(num_workers)))
    def process(inputs_queue, worker_id):
        from transformers import AutoTokenizer
        from vllm import LLM, EngineArgs, SamplingParams, TokensPrompt
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        eos_token_id, eos_token = get_eos(tokenizer, args.prompt_type)
        logger.info(f"eos token: {eos_token}")

        engine_args = EngineArgs(model=args.model, 
                                 swap_space=8, 
                                 seed=args.seed, 
                                 enable_prefix_caching=True, 
                                 tensor_parallel_size=args.tp_size)
        model = LLM(**engine_args.__dict__)
        eval_params_map = json.load(open(args.sampling_params))
        sampling_params_map = {n: SamplingParams(stop_token_ids=[eos_token_id], **p) 
                               for n, p in eval_params_map.items()}
        
        cache_file = cache.open(worker_id)
        no_input = False
        outs = []

        while not no_input or model.llm_engine.has_unfinished_requests():
            if model.llm_engine.get_num_unfinished_requests() < args.batch_size and not no_input:
                item = inputs_queue.get()
                if item is None:
                    no_input = True
                else:
                    outs.append([None, *item[1:]])
                    input_ids = get_prompt_ids(item[0], tokenizer, args.prompt_type)
                    model.llm_engine.add_request(str(len(outs) - 1), TokensPrompt(prompt_token_ids=input_ids), sampling_params_map[file])
            
            if model.llm_engine.has_unfinished_requests():
                step_outputs = model.llm_engine.step()
                for out in step_outputs:
                    if out.finished:
                        idx = int(out.request_id)
                        out = [ex.text for ex in out.outputs]
                        outs[idx][0] = out
                        cache.write_wcache(cache_file, outs[idx])

        return outs

    cache.nfork(num_workers)
    outs = process(all_test_prompts) + cached_outs

    for batch_outs in outs:
        out, file, i = batch_outs
        all_outputs[file][i] = out

    for dirname, file in test_files:
        safe_save(all_outputs[file], os.path.join(args.output_dir, "completions." + file))

    cache.delete()