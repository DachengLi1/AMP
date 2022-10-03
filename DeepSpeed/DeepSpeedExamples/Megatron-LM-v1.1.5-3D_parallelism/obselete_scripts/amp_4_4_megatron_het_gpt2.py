import math
import time
from collections import defaultdict
import operator
import random
import os
import copy

from tqdm import tqdm

import numpy as np

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import megatron_strategy
from cost import get_cost_c, get_cost_e, rank_loss, AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import simulate, to_float_torch

# cluster information

# number of GPU per node, number of nodes
M = 4
N = 4

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

#TODO: Find DGX-2 box config
cluster_info = {}

# inter-node bandwidth, intra-node bandwidth
cluster_info[0] = [torch.tensor([50 * 1e9 / 32]).float(), torch.tensor([50 * 1e9 / 32]).float()]
for i in range(1, N):
    cluster_info[i] = [torch.tensor([10 * 1e9 / 32]).float(), torch.tensor([170 * 1e9 / 32]).float()]

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([1024]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"gpt2"}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"megatron_het_gpt2"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# save this name to env
os.environ["amp_log_path"] = record_file

global_bs = 32
model = AMP(model_config, exp_name)
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

simulated_settings = [] 
feasible = {}

with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")                
    fp.write(f"gbs:{global_bs}\n")                
known = None
iter_count = 0
while True:
    ret = megatron_strategy(M=M, N=N, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, known = ret
        oth = {"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w,
                       "orig_pp": torch.ones(1,)*(M*N/(h*w))}
        gt_costs = simulate([None], [None], torch.ones(1,)*global_bs, to_float_torch([mbs]), model_config, [oth], exp_name)
        #gt_costs = [np.random.randn()] 
        gt_cost = gt_costs[0]
        with open(record_file, "a") as fp:
            fp.write(f"megatron - mbs: {mbs} degree: {oth}, r_cost: {gt_cost} \n")                
        if gt_cost != float("inf") and (M*N)//(h*w) != 1:
            known_feasible = feasible.get(mbs, float("inf"))
            feasible[mbs] = min(M*N//w, known_feasible)
            cur_remain = known[mbs]
            replace_remain = []
            for (h, w) in cur_remain:
                assert M*N % w == 0
                product =  M*N // w
                if product <= feasible[mbs]:
                    replace_remain.append((h, w))
                else:
                    with open(record_file, "a") as fp:
                        fp.write(f"{h, w} deleted - current min {feasible[mbs]} \n")
            if len(replace_remain) > 0:
                known[mbs] = replace_remain
            else:
                known.pop(mbs, None)
        else:
            with open(record_file, "a") as fp:
                fp.write(f"{mbs, h, w} infinity \n")
                
        simulated_settings.append(((mbs, h, w), gt_cost))
    iter_count += 1

print(f"finish megatron search with {iter_count} iterations")
# sorted simulated settings
sorted_settings = sorted(simulated_settings, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"{item}")
