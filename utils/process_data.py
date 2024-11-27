import logging
import importlib

from sortedcontainers import SortedList

from .arguments_init import get_training_args
from .utils import do_package
from tools import batch_samples

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def preprocess_data(formated_data):
    train_formated_data, dev_formated_data = formated_data
    training_args = get_training_args()
    module = importlib.import_module("utils.trainer_" + training_args.training_type)
    get_inputs = module.get_inputs
    len_fn = module.len_fn
    pad_fn = module.pad_fn
    num_fns = module.num_fns
    collate_fn = lambda ex: do_package(pad_fn(ex))
    
    def tokenize_function_train_pad(formated_data):
        inputs = []

        for name, examples in formated_data.items():
            inputs.extend(get_inputs(examples, name, "train"))

        if training_args.local_rank == 0:
            logger.info(f"Size of Training Data: {len(inputs)}")
        
        fw_batchs = batch_samples(inputs, len_fn, training_args.per_deivce_train_batch_max_tokens // training_args.batch_tokens_divider)

        group_size = training_args.h_accumulation_steps * training_args.world_size * training_args.batch_tokens_divider
        remainder = len(fw_batchs) % group_size
        if remainder > 0:
            idx = len(fw_batchs) - 1
            for _ in range(group_size - remainder):
                while idx >= 0 and len(fw_batchs[idx]) == 1:
                    idx -= 1
                if idx == - 1:
                    break
                half_size = len(fw_batchs[idx]) // 2
                fw_batchs.append(fw_batchs[idx][half_size:])
                fw_batchs[idx] = fw_batchs[idx][:half_size]
                idx -= 1

            remainder = len(fw_batchs) % group_size
            idx = len(fw_batchs) - 1
            if remainder > 0:
                for _ in range(group_size - remainder):
                    while idx >= 0 and len(fw_batchs[idx]) == 1:
                        idx -= 1
                    if idx == - 1:
                        raise NotImplementedError
                    fw_batchs.append(fw_batchs[idx][-1:])
                    fw_batchs[idx] = fw_batchs[idx][:-1]

        fw_batchs.sort(key=lambda b: len(b), reverse=True)

        if training_args.local_rank == 0:
            logger.info(f"Optimization Steps: {len(fw_batchs) / group_size}")
            logger.info(f"Average Batch Size: {len(inputs) / (len(fw_batchs) / group_size)}")

        if len(fw_batchs) > group_size and group_size > 1:
            n = group_size
            gbatch_w_size = [(0, ()) for _ in range(len(fw_batchs) // n)]
            sorted_gbatch = SortedList([(0, i) for i in range(len(gbatch_w_size))])
            for i, b in enumerate(fw_batchs):
                s = len(b)
                _, idx = sorted_gbatch.pop(0)
                gbatch_w_size[idx] = (_ + s, gbatch_w_size[idx][1] + ((s, i),))
                if len(gbatch_w_size[idx][1]) < n:
                    sorted_gbatch.add((_ + s, idx))

            sorted_gbatch = SortedList(gbatch_w_size, key=lambda x: x[0])
            while True:
                mn = sorted_gbatch.pop(0)
                mx = sorted_gbatch.pop(-1)
                dt = mx[0] - mn[0]

                poss_dt, pi, pj = 0, -1, -1
                for i in range(n):
                    for j in range(n):
                        ex_dt = mx[1][i][0] - mn[1][j][0]
                        if ex_dt < dt and ex_dt > poss_dt:
                            poss_dt, pi, pj = ex_dt, i, j

                if pi == -1:
                    sorted_gbatch.add(mn)
                    sorted_gbatch.add(mx)
                    break
                
                new_mx = (mx[0] - poss_dt, mx[1][:pi] + (mn[1][pj], ) + mx[1][pi+1:])
                new_mn = (mn[0] + poss_dt, mn[1][:pj] + (mx[1][pi], ) + mn[1][pj+1:])
                assert sum(b[0] for b in new_mn[1]) == new_mn[0]
                assert sum(b[0] for b in new_mx[1]) == new_mx[0]
                sorted_gbatch.add(new_mn)
                sorted_gbatch.add(new_mx)

            collected_fw_batchs = []
            for gb in sorted_gbatch:
                collected_fw_batchs.extend([fw_batchs[i[1]] for i in gb[1]])
            
            fw_batchs = collected_fw_batchs

        fw_batchs = [sum(fw_batchs[st: st+training_args.batch_tokens_divider], start=[]) 
                    for st in range(0, len(fw_batchs), training_args.batch_tokens_divider)]
        
        group_size //= training_args.batch_tokens_divider

        if training_args.local_rank == 0:
            logger.info(f"Min Batch Size: {min([sum(len(b) for b in fw_batchs[st: st+group_size]) for st in range(0, len(fw_batchs), group_size)])}")
            logger.info(f"Max Batch Size: {max([sum(len(b) for b in fw_batchs[st: st+group_size]) for st in range(0, len(fw_batchs), group_size)])}")
        
        fw_batchs = [collate_fn(b) for b in fw_batchs]

        world_size = training_args.world_size
        batchs = []
        for st in range(0, len(fw_batchs), group_size):
            sub_batchs = sorted(fw_batchs[st: st+group_size], key=lambda b: b["input_ids"].numel(), reverse=True)
            group_batchs = []
            for i in range(world_size):
                group_batchs.append(sub_batchs[i::world_size*2] + sub_batchs[world_size*2-1-i::world_size*2])
            
            nums = dict()
            for n, fn in num_fns.items():
                nums[n] = sum(fn(b) for b in sub_batchs)
            
            for i, group_batch in enumerate(group_batchs):
                group_batchs[i] = {"nums": nums, "batch": group_batch}
            batchs.extend(group_batchs)
        
        return batchs
    
    def tokenize_function_eval_pad(formated_data):
        inputs = {}
        for name, examples in formated_data.items():
            inputs[name] = get_inputs(examples, name, "eval")
        return inputs
    
    train_datasets, dev_datasets = {}, {}
    
    if training_args.do_train:
        train_datasets = tokenize_function_train_pad(train_formated_data)
        logger.info(f"Size of batched train dataset: {len(train_datasets)}")

    if training_args.do_eval:
        dev_datasets = tokenize_function_eval_pad(dev_formated_data)
        for name, dev_dataset in dev_datasets.items():
            logger.info(f"Size of {name} eval dataset: {len(dev_dataset)}")
    
    return train_datasets, dev_datasets, collate_fn
