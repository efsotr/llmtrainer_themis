import json
import shutil
from pathlib import Path
from collections import defaultdict

class Cache:

    def __init__(self, cache_dir: str, batch_size: int = 0):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.batch_size = batch_size

    def get_cache(self, exs, idx_ex_fn=None, idx_cache_fn=None):
        if self.cache_dir is None:
            return [], exs

        if not self.cache_dir.exists():
            self.cache_dir.mkdir()
            (self.cache_dir / "FILE_COUNT").write_text(str(0))

        tot = int((self.cache_dir / "FILE_COUNT").read_text())
        cached_outs = []
        for i in range(tot):
            s = f"cache{i}.jsonl"
            try:
                for line in open(self.cache_dir / s, encoding="utf-8").readlines():
                    try:
                        cached_outs.append(json.loads(line))
                    except:
                        pass
            except:
                pass

        if idx_cache_fn is None:
            idx_cache_fn = idx_ex_fn
        cached_idx = set([idx_cache_fn(out) for out in cached_outs])
        if len(cached_idx) > 0:
            exs = [ex for ex in exs if idx_ex_fn(ex) not in cached_idx]
        return cached_outs, exs

    def nfork(self, n):
        if self.cache_dir is None:
            return 
        tot = int((self.cache_dir / "FILE_COUNT").read_text())
        (self.cache_dir / "FILE_COUNT").write_text(str(tot + n))
        self.cache_files = [self.cache_dir / f"cache{i}.jsonl" for i in range(tot, tot + n)]

    def open(self, idx):
        if self.cache_dir is None:
            return None
        if self.batch_size > 0:
            return open(self.cache_files[idx], "a"), []
        return open(self.cache_files[idx], "a")

    def write(self, file, out):
        if self.cache_dir is None:
            return 
        file.write(json.dumps(out) + "\n")

    def write_multi(self, file, outs):
        if self.cache_dir is None:
            return 
        file.write("".join([json.dumps(out) + "\n" for out in outs]))

    def write_wcache(self, file, out):
        if self.cache_dir is None:
            return 
        file, outs = file
        outs.append(out)
        if len(outs) == self.batch_size:
            file.write("".join([json.dumps(out) + "\n" for out in outs]))
            outs.clear()
    
    def delete(self):
        if self.cache_dir is None:
            return None
        shutil.rmtree(self.cache_dir)
