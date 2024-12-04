import os
import re
import sys
import logging
import argparse
from collections import Counter
from typing import cast, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(training_dir)

from file_IO import safe_load, safe_save
from stats import Correlation

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
    output_filename: str

    def init(self):        
        self.avail_test_files = set(file for file in self.test_files.split(',')) \
            if self.test_files is not None and self.test_files != "" else AllIn()
        
        self.test_dirs = [test_dir for test_dir in self.test_dirs.split(',')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dirs', required=True, type=str, help="Testset directory")
    parser.add_argument('--test_files', default=None, type=str, help="Testset filename")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory")
    parser.add_argument('--output_filename', default="stats.overall.json", type=str, help="Output filename")
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

    rating_pattern = re.compile(r"Rating:\s*([1-5])")
    INVALID_RATING = 0
    def parse_rating(response):
        last_line = response.split("\n")[-1]
        match = rating_pattern.search(last_line)
        return int(match.group(1)) if match else INVALID_RATING
    
    def get_score(ex):
        score = ex["score"]
        if isinstance(score, (list, tuple)):
            return score[0]
        return score
    
    all_test_prompts = []
    all_scores = {}

    overall_stats = {}
    for dirpath, file in test_files:
        test_dataset = safe_load(os.path.join(dirpath, file))
        test_outputs = safe_load(os.path.join(args.output_dir, "completions." + file))
        assert isinstance(test_dataset, list) , "test_dataset must be of type list"
        assert len(test_dataset) > 0, "test_dataset must have a length greater than 0"
        assert len(test_dataset) == len(test_outputs)

        ratings = [parse_rating(outs[0]) for outs in test_outputs]
        rating_distribution = Counter(ratings)
        rating_distribution = dict(sorted(rating_distribution.most_common()))
        correlation = {}

        if test_dataset[0].get("sys_id", None) is None:
            num_sys = 1
            exs = [
                (get_score(ex), r, None, None)
                for ex, r in zip(test_dataset, ratings)
            ]
        else:
            num_sys = len(set(ex["sys_id"] for ex in test_dataset))
            exs = [
                (get_score(ex), r, ex['sys_id'], ex['seg_id'])
                for ex, r in zip(test_dataset, ratings)
            ]
            exs.sort(key=lambda x: (x[2], x[3]))
        golds = [ex[0] for ex in exs]
        preds = [ex[1] for ex in exs]
        corr = Correlation(num_sys, golds, preds)

        if num_sys > 1:
            for metric in ["Pearson", "Spearman", "Kendall"]:
                correlation[metric] = getattr(corr, metric)("item")[0]
        else:
            for metric in ["Pearson", "Spearman", "Kendall"]:
                correlation[metric] = getattr(corr, metric)()[0]

        overall_stats[file] = {
            "correlation": correlation,
            "rating_distribution": rating_distribution,
            "dataset_size": len(test_dataset)
        }

    avgs = ["SummEval", "Topical-Chat", "WMT23"]
    for avg in avgs:
        s = {"Pearson": 0, "Spearman": 0, "Kendall": 0}
        n = {"Pearson": 0, "Spearman": 0, "Kendall": 0}
        for k in overall_stats.keys():
            if k.startswith(avg) and k.endswith(".json"):
                for metric in ["Pearson", "Spearman", "Kendall"]:
                    s[metric] += overall_stats[k]["correlation"][metric]
                    n[metric] += 1

        if n["Pearson"] > 0:
            for metric in ["Pearson", "Spearman", "Kendall"]:
                s[metric] /= n[metric]
            overall_stats[avg] = s
        
    safe_save(overall_stats, os.path.join(args.output_dir, args.output_filename))