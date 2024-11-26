import sys
import math

from sortedcontainers import SortedList

from file_IO import safe_load
from tools import batch_samples

max_length = int(sys.argv[2])
training_type = sys.argv[3]
assert training_type in ["sft", "po", "xpo", "ord"]

def get_data_len(data):
    if training_type == "po":
        data_len = []
        for ex in data["po"]:
            p = len(ex["prompt"])
            c = len(ex["chosen"])
            r = len(ex["rejected"])
            if p + 1 + c + r > max_length * 2 or ex["chosen"] == ex["rejected"]:
                continue
            data_len.append(p + 1 + c + r)
    elif training_type == "sft":
        data_len = []
        for ex in data["sft"]:
            p = len(ex["prompt"])
            if p > max_length - 16:
                continue
            data_len.append(min(p + len(ex["response"]), max_length))
    elif training_type == "xpo":
        data_len = []
        for ex in data["xpo"]:
            tot = len(ex["prompt"]) + sum(map(len, ex["response"])) + len(ex["response"]) // 2
            if tot > max_length * 2:
                continue
            data_len.append(tot)
    elif training_type == "ord":
        data_len = []
        for ex in data["ord"]:
            tot = len(ex["prompt"]) + sum(map(len, ex["response"]))
            if tot > max_length:
                continue
            data_len.append(tot)

    return data_len

if "," not in sys.argv[1]:
    data = safe_load(sys.argv[1])
    data_len = get_data_len(data)
else:
    data_len = []
    for path in sys.argv[1].split(","):
        data = safe_load(path)
        data_len.extend(get_data_len(data))

print("dataset size:", len(data_len))
    
while True:
    print("input group_size(if =0 then exit) and batch size: ", end='', flush=True)
    group_size, batch_size = map(int, input().split())
    if group_size == 0:
        break
    def get_bs(data_len, batch_tokens, group_size):
        cnt = batch_samples(data_len, lambda x: x, batch_tokens)
        cnt = list(map(len, cnt))

        idx = len(cnt) - 1
        for i in range((group_size - len(cnt) % group_size) % group_size):
            cnt.append(cnt[idx] - cnt[idx] // 2)
            cnt[idx] //= 2
            idx -= 1
        
        cnt.sort(reverse=True)

        if len(cnt) > group_size:
            gbatch_w_size = [(0, ()) for _ in range(len(cnt) // group_size)]
            sorted_gbatch = SortedList([(0, i) for i in range(len(gbatch_w_size))], key=lambda x: x[0])
            for i, s in enumerate(cnt):
                _, idx = sorted_gbatch.pop(0)
                gbatch_w_size[idx] = (_ + s, gbatch_w_size[idx][1] + (s,))
                if len(gbatch_w_size[idx][1]) < group_size:
                    sorted_gbatch.add((_ + s, idx))

            sorted_gbatch = SortedList(gbatch_w_size, key=lambda x: x[0])
            while True:
                mn = sorted_gbatch.pop(0)
                mx = sorted_gbatch.pop(-1)
                dt = mx[0] - mn[0]

                poss_dt, pi, pj = 0, -1, -1
                for i in range(group_size):
                    for j in range(group_size):
                        ex_dt = mx[1][i] - mn[1][j]
                        if ex_dt < dt and ex_dt > poss_dt:
                            poss_dt, pi, pj = ex_dt, i, j

                if pi == -1:
                    sorted_gbatch.add(mn)
                    sorted_gbatch.add(mx)
                    break
                
                new_mx = (mx[0] - poss_dt, mx[1][:pi] + (mn[1][pj], ) + mx[1][pi+1:])
                new_mn = (mn[0] + poss_dt, mn[1][:pj] + (mx[1][pi], ) + mn[1][pj+1:])
                sorted_gbatch.add(new_mn)
                sorted_gbatch.add(new_mx)
            
            ret = (sorted_gbatch[0][0], sorted_gbatch[-1][0])
        else:
            ret = (sum(cnt), sum(cnt))

        return len(data_len) / math.ceil(len(cnt) / group_size), ret
    
    l, r, batch_max_tokens, bs = 0, 1 << (22 - 5), 0, None
    while l <= r:
        mid = (l + r) >> 1
        mid_batch_size, mid_bs = get_bs(data_len, mid << 5, group_size)
        if mid_batch_size >= batch_size:
            batch_max_tokens = mid << 5
            bs = mid_bs
            r = mid - 1
        else:
            l = mid + 1
    print("batch_max_tokens", batch_max_tokens, flush=True)
    print("batch size", bs, flush=True)

