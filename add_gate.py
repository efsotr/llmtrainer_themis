import os
import sys
from file_IO import safe_load, safe_save

dir = sys.argv[1]
gate_mapping = safe_load("task_aspect_lora.json")
for full_dir, _, files in os.walk(dir):
    for file in files:
        if file.endswith(".json") or file.endswith(".json.gz"):
            path = os.path.join(full_dir, file)
            data = safe_load(path)
            try:
                def add_gate(ex):
                    ex["gate"] = gate_mapping[ex["task"]][ex["aspect"]]
                    return ex
                data = [add_gate(ex) for ex in data]
            except:
                print("no", file)
                continue
            safe_save(data, path)