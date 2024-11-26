import os
import sys
import copy
import torch
import logging
import argparse
from itertools import chain
from typing import cast, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(training_dir)

from tools import batch_samples
from file_IO import safe_load, safe_save
from tokenize_ids import get_prompt_ids, get_token_ids, get_eos
from multi_process_ray import multi_process, multi_process_decorator

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
    test_files: str
    batch_tot_tokens: int
    seed: int
    tp_size: int = 1
    prefix_dir: str
    output_filename: str

    def init(self):        
        self.avail_test_files = set(file for file in self.test_files.split(',')) \
            if self.test_files is not None and self.test_files != "" else AllIn()
        
        self.test_dirs = [test_dir for test_dir in self.test_dirs.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="./models/ArmoRM-Llama3-8B-v0.1", type=str)
    parser.add_argument('--prefix_dir', default=None, type=str)
    parser.add_argument('--batch_tot_tokens', default=128 * 1024, type=int)
    parser.add_argument('--test_dirs', required=True, type=str, help="Testset directory")
    parser.add_argument('--test_files', default=None, type=str, help="Testset filename")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory")
    parser.add_argument('--output_filename', default="stats.overall.json", type=str, help="Output filename")
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args(namespace=Namespace())
    args.init()
    
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
    all_scores = {}
    avg_lengths = {}

    def get_attr_batch(prompt, prefix, res):
        inputs_ids = prompt + prefix
        position_ids = list(range(len(inputs_ids)))
        score_mask = [0] * len(inputs_ids)
        gate_pos = len(prompt) - 5
        start_q = [0] * len(inputs_ids)
        end_k = [len(res)] * len(inputs_ids)
        pos = len(inputs_ids)
        for i, r in enumerate(res, 1):
            inputs_ids += r
            position_ids += list(range(pos, pos + len(r)))
            score_mask += [0] * (len(r) - 1) + [1]
            start_q += [i] * len(r)
            end_k += [i] * len(r)
        gate_mask = [0] * len(inputs_ids)
        gate_mask[gate_pos] = 1
        return inputs_ids, position_ids, score_mask, gate_mask, start_q, end_k

    for dirpath, file in test_files:
        test_dataset = safe_load(os.path.join(dirpath, file))
        if args.prefix_dir:
            test_prefix = safe_load(os.path.join(args.prefix_dir, "prefix." + file))
            assert len(test_dataset) == len(test_prefix)
        else:
            test_prefix = ["" for _ in range(len(test_dataset))]
            
        test_outputs = safe_load(os.path.join(args.output_dir, "completions." + file))
        assert isinstance(test_dataset, list) , "test_dataset must be of type list"
        assert len(test_dataset) > 0, "test_dataset must have a length greater than 0"
        assert len(test_dataset) == len(test_outputs)

        avg_lengths[file] = sum(sum(map(len, outs)) / len(outs) for outs in test_outputs) / len(test_outputs)
        def process(inputs):
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            _, eos_token = get_eos(tokenizer, "chat")
            rets = []
            for i, (ex, prefix, outs) in inputs:
                prompt = get_prompt_ids(ex, tokenizer, "chat")
                prefix = get_token_ids(prefix, tokenizer)
                responses = []
                for out in outs:
                    responses.append(get_token_ids(out + eos_token, tokenizer))
                batchs = batch_samples(list(enumerate(responses)), len_fn=lambda x: len(x[1]), batch_max_size=args.batch_tot_tokens - len(prompt) - len(prefix), enable_sort=True)
                for batch in batchs:
                    idxs, res = list(zip(*batch))
                    rets.append(get_attr_batch(prompt, prefix, res) + (list(zip([file] * len(idxs), [i] * len(idxs), idxs)), ))
            return rets

        all_test_prompts.extend(multi_process(enumerate(zip(test_dataset, test_prefix, test_outputs)), process, num_workers=32, num_cpus=1))
        all_scores[file] = [[None] * len(outs) for outs in test_outputs]
        
    batchs = batch_samples(all_test_prompts, len_fn=lambda x: len(x[0]), batch_max_size=args.batch_tot_tokens, enable_sort=True)
    def get_batch(batch):
        batch = list(zip(*batch))
        batch[-1] = list(chain(*batch[-1]))
        return batch

    batchs = [get_batch(batch) for batch in batchs]

    num_workers = torch.cuda.device_count() // args.tp_size
    logger.info(f"num_workers: {num_workers}")
    
    @multi_process_decorator(multi_process, num_workers=num_workers, num_gpus=args.tp_size, num_cpus=1)
    def process(inputs):
        if inputs == []:
            return []
        import torch
        from transformers import AutoConfig
        from optimized_module.modeling_armo import LlamaForRewardModelWithGating, do_batch
        config = AutoConfig.from_pretrained(args.model)
        config.use_cache = False
        config.use_tree_attn = True
        model = LlamaForRewardModelWithGating.from_pretrained(args.model,
                                                              config=config,
                                                              device_map="cuda",
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.bfloat16,
                                                              )
        model.eval()

        from tqdm import tqdm
        with torch.no_grad():
            outs = []
            for batch in tqdm(inputs):
                input_ids, position_ids, score_mask, gate_mask, start_q, end_k, pos = batch
                input_batch = do_batch(input_ids, position_ids, score_mask, gate_mask, start_q, end_k)
                batch_score = model(**input_batch).score

                assert len(batch_score) == len(pos)
                outs.extend(zip(batch_score.tolist(), pos))
            
        return outs
    
    outs = process(batchs)

    for ex in outs:
        score, (file, i, j) = ex
        all_scores[file][i][j] = score

    for file in all_scores:
        safe_save(all_scores[file], os.path.join(args.output_dir, "stats." + file))

    def mean(x):
        return sum(x) / len(x)

    avg_mean_scores = {}
    avg_max_scores = {}
    for file in all_scores:
        avg_mean_scores[file] = mean([mean(scores) for scores in all_scores[file]])
        avg_max_scores[file] = mean([max(scores) for scores in all_scores[file]])

    safe_save({"avg_mean_scores": avg_mean_scores, 
               "avg_max_scores": avg_max_scores, 
               "avg_lengths(chars)": avg_lengths}, 
              os.path.join(args.output_dir, args.output_filename))