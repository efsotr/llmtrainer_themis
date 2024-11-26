import os
import sys
import torch
import logging
import argparse
from typing import cast, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(training_dir)

from multi_process_ray import multi_process, multi_process_decorator
from file_IO import safe_load, safe_save

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
    output_dir: str
    test_dirs: List[str]
    test_files: str
    batch_size: int
    seed: int
    tp_size: int = 1
    output_filename: str

    def init(self):        
        self.avail_test_files = set(file for file in self.test_files.split(',')) \
            if self.test_files is not None and self.test_files != "" else AllIn()
        
        self.test_dirs = [test_dir for test_dir in self.test_dirs.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
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
                    and not file.startswith("completions.") and not file.startswith("stats."):
                    test_files.append((dirpath, file))

    test_files.sort()
    assert len(set(file for _, file in test_files)) == len(test_files), "File names in test_files must be unique"
    logger.info(f"Test files: {', '.join(file for _, file in test_files)}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_test_prompts = []
    all_scores = {}
    avg_lengths = {}

    for dirpath, file in test_files:
        test_dataset = safe_load(os.path.join(dirpath, file))
        test_outputs = safe_load(os.path.join(args.output_dir, "completions." + file))
        assert isinstance(test_dataset, list) , "test_dataset must be of type list"
        assert len(test_dataset) > 0, "test_dataset must have a length greater than 0"
        assert len(test_dataset) == len(test_outputs)

        avg_lengths[file] = sum(sum(map(len, outs)) / len(outs) for outs in test_outputs) / len(test_outputs)
        all_test_prompts.extend([
            ([{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": out}], file, i, j)
            for i, (ex, outs) in enumerate(zip(test_dataset, test_outputs))
            for j, out in enumerate(outs)
        ])
        all_scores[file] = [[None] * len(outs) for outs in test_outputs]

    num_workers = torch.cuda.device_count() // args.tp_size
    logger.info(f"num_workers: {num_workers}")

    @multi_process_decorator(multi_process, num_workers=num_workers, num_gpus=args.tp_size, num_cpus=1)
    def process(inputs):
        import gc
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_path = "./models/ArmoRM-Llama3-8B-v0.1"
        model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                                device_map="cuda",
                                                                trust_remote_code=True,
                                                                torch_dtype=torch.bfloat16,
                                                                attn_implementation='flash_attention_2',
                                                                )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        scores = []
        alls = list(zip(*inputs))
        inputs, others = alls[0], alls[1:]
        with torch.no_grad():
            bsz = args.batch_size
            st = 0
            while st < len(inputs):
                while True:
                    try:
                        input_ids = tokenizer.apply_chat_template(inputs[st: st+bsz], padding=True, return_tensors="pt").cuda()
                        scores.extend(model(input_ids).score.tolist())
                        break
                    except:
                        if bsz == 1:
                            raise NotImplementedError("batch_size == 1")
                        torch.cuda.empty_cache()
                        for i in range(10):
                            gc.collect()
                        bsz //= 2
                        logger.info(f"batch size reduce to {bsz}")

                st += bsz

        return list(zip(scores, *others))
    
    outs = process(all_test_prompts)

    for ex in outs:
        score, file, i, j = ex
        all_scores[file][i][j] = score

    # for dirname, file in test_files:
    #     safe_save(all_scores[file], os.path.join(args.output_dir, "stats." + file))

    def mean(x):
        return sum(x) / len(x)

    avg_scores = {}
    for file in all_scores:
        avg_scores[file] = mean([mean(scores) for scores in all_scores[file]])

    safe_save({"avg_scores": avg_scores, "avg_lengths": avg_lengths, "scores": all_scores}, 
              os.path.join(args.output_dir, args.output_filename))