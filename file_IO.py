import os
import json
import gzip
import torch
from typing import Optional

suffixs = [".gz", ".json", ".jsonl", ".pt"]
def get_base(filepath: str):
    for suffix in suffixs:
        filepath = filepath.removesuffix(suffix)
    return filepath
    
def safe_load(filepath: Optional[str]):
    if filepath is None:
        return None
    if filepath.endswith(".jsonl.gz"):
        return [json.loads(line) for line in gzip.open(filepath, "rt").readlines()]
    if filepath.endswith(".json.gz"):
        return json.load(gzip.open(filepath, "rt"))
    if filepath.endswith(".pt.gz"):
        return torch.load(gzip.open(filepath, "rb"))
    if filepath.endswith(".jsonl"):
        return [json.loads(line) for line in open(filepath).readlines()]
    if filepath.endswith(".json"):
        return json.load(open(filepath))
    if filepath.endswith(".pt"):
        return torch.load(filepath)
    raise NotImplementedError(filepath)

def to_json(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "to_list"):
        return obj.to_list()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise NotImplementedError(obj)

def safe_save(obj, filepath: str):
    if filepath.endswith(".jsonl.gz"):
        return gzip.open(filepath, "wt", compresslevel=1).write('\n'.join(json.dumps(ex, default=to_json) for ex in obj))
    if filepath.endswith(".json.gz"):
        return json.dump(obj, gzip.open(filepath, "wt", compresslevel=1), default=to_json)
    if filepath.endswith(".pt.gz"):
        return torch.save(obj, gzip.open(filepath, "wb", compresslevel=1))
    if filepath.endswith(".jsonl"):
        return open(filepath, "w").write('\n'.join(json.dumps(ex, default=to_json) for ex in obj))
    if filepath.endswith(".json"):
        return json.dump(obj, open(filepath, "w"), indent=2, default=to_json)
    if filepath.endswith(".pt"):
        return torch.save(obj, filepath)
    raise NotImplementedError(filepath)

def can_create_file(filepath: str):
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath, exist_ok=True)
        except (OSError, IOError):
            return False
    return os.access(dirpath, os.W_OK)