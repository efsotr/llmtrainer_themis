import os
import sys
import copy
import json
import torch
from safetensors.torch import load_file, save_file

dir = sys.argv[1]
output_dir = dir
peft_config = json.load(open(os.path.join(dir, "adapter_config.json")))
peft_params = load_file(os.path.join(dir, "adapter_model.safetensors"))
peft_config.pop("num_gates")
# peft_config["lora_alpha"] *= 2
# peft_config["r"] *= 2

keys = list(set(['.'.join(k.split('.')[:-2]) for k in peft_params.keys()]))
gate_mapping = {"t1": 0, "t2": 1, "t3": 2, "t4": 3, "a1": 4, "a2": 5, "a3": 6, "a4": 7}
gate_rmapping = dict(zip(gate_mapping.values(), gate_mapping.keys()))
combination_peft_params = {}
ratio = {}
for i in range(0, 4):
    for j in range(4, 8):
        name = f"{gate_rmapping[i]}{gate_rmapping[j]}"
        combination_peft_params[name] = {}
        ratio[name] = 4 if j == 7 else 2
for key in keys:
    new_key = key + ".weight"
    for i in range(0, 4):
        for j in range(4, 8):
            name = f"{gate_rmapping[i]}{gate_rmapping[j]}"
            if j == 6:
                combination_peft_params[name][new_key] = torch.cat([peft_params[f"{key}.{i}.weight"], peft_params[f"{key}.{5}.weight"]], dim=int("lora_B" in key))
            elif j == 7:
                combination_peft_params[name][new_key] = torch.cat([peft_params[f"{key}.{i}.weight"], peft_params[f"{key}.{4}.weight"], peft_params[f"{key}.{5}.weight"], torch.zeros_like(peft_params[f"{key}.{5}.weight"])], dim=int("lora_B" in key))
            else:
                combination_peft_params[name][new_key] = torch.cat([peft_params[f"{key}.{i}.weight"], peft_params[f"{key}.{j}.weight"]], dim=int("lora_B" in key))

for name, params in combination_peft_params.items():
    param_dir = os.path.join(output_dir, name)
    os.makedirs(param_dir, exist_ok=True)
    new_peft_config = copy.deepcopy(peft_config)
    new_peft_config["lora_alpha"] *= ratio[name]
    new_peft_config["r"] *= ratio[name]
    json.dump(new_peft_config, open(os.path.join(param_dir, "adapter_config.json"), "w"), indent=2)
    save_file(params, os.path.join(os.path.join(param_dir, "adapter_model.safetensors")))