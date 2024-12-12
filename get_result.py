import os
import sys
import glob
import json
import pandas as pd

ck_dirs = sys.argv[1]
ck_dirs = ck_dirs.split(",")

class Avg:
    
    def __init__(self):
        self.sum = 0
        self.num = 0

    def add(self, x):
        self.sum += x
        self.num += 1

    def result(self):
        return self.sum / self.num

def get_overall(stats, is_in):
    out = {
        "Pearson": Avg(),
        "Spearman": Avg(),
        "Kendall": Avg(),
    }
    for k, item in stats.items():
        if k.endswith(".json") and is_in(k):
            for corr, v in item["correlation"].items():
                out[corr].add(v)
    
    return {corr: v.result() for corr, v in out.items()}

def get_overall2(*items):
    out = {
        "Pearson": Avg(),
        "Spearman": Avg(),
        "Kendall": Avg(),
    }
    for item in items:
        for corr, v in item.items():
            out[corr].add(v)
    
    return {corr: v.result() for corr, v in out.items()}

def add(result_dict, prefix, other_dict):
    for k, v in other_dict.items():
        result_dict[f"{prefix}.{k}"] = v

results = []
for ck_dir in ck_dirs:
    files = glob.glob(os.path.join(ck_dir, "test_result-checkpoint-*/"))
    file_list = []
    for file in files:
        paths = file.split(os.path.sep)
        name = paths[2]
        ck_num = int(paths[3].removeprefix("test_result-checkpoint-"))
        file_list.append((os.path.join(file, "stats.overall.json"), name, ck_num))
    
    file_list.sort(key=lambda ex: ex[2])
    for i, (file, name, _) in enumerate(file_list, 1):
        if "lora32" not in name:
            name = name.replace("lora", "lora-r16")
        else:
            name = name.replace("lora32", "lora-r32")
        stats = json.load(open(file))
        name += f"_{i}epoch"
        result_dict = {"name": name}
        SummEval = get_overall(stats, lambda x: x.startswith("SummEval"))
        Topical = get_overall(stats, lambda x: x.startswith("Topical-Chat"))
        sf = get_overall(stats, lambda x: x.startswith("sfhot") or x.startswith("sfres"))
        add(result_dict, "Overall", get_overall2(*(SummEval, Topical, sf)))
        add(result_dict, "SummEval", SummEval)
        add(result_dict, "Topical-Chat", Topical)
        add(result_dict, "SFHOT&SFRES", sf)
        for k, v in stats.items():
            if k.endswith(".json"):
                k = k.removesuffix(".json")
                add(result_dict, k, v["correlation"])
        results.append(result_dict)

results = pd.DataFrame(results)
results.to_csv("reulst.csv", index=False)