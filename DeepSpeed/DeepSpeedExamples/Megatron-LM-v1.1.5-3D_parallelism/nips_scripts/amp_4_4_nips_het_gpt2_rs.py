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

from sa import random_strategy
from cost import get_cost_c, get_cost_e, rank_loss, AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import simulate, to_float_torch

time_all_start = time.time()
# cluster information

# number of GPU per node, number of nodes
M = 4
N = 4

data_bs = 2

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([1024]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"gpt2"}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"rs_het_gpt2"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# save this name to env
os.environ["amp_log_path"] = record_file

global_bs = 32
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

simulated_settings = [] 
feasible = {}

known = None
budget = 50

for bud in range(budget):
    ret = random_strategy(M=M, N=N, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, rank_map, known = ret
        oth = {"orig_mp": torch.ones(1,)*h, "orig_dp": torch.ones(1,)*w,
                       "orig_pp": torch.ones(1,)*(M*N/(h*w))}
        gt_costs = simulate([rank_map], [None], torch.ones(1,)*global_bs, to_float_torch([mbs]), model_config, [oth], exp_name)
        #gt_costs = [np.random.randn()] 
        gt_cost = gt_costs[0]
        with open(record_file, "a") as fp:
            fp.write(f"rs - mbs: {mbs} degree: {oth}, r_cost: {gt_cost} \n")                
        if gt_cost == float("inf"):
            # find index of config
            #print(known)
            if (h, w) in list(known.keys()):
                for cs in (known[(h, w)]):
                    for find_c in range(len(known[(h,w)])):
                        if (known[(h,w)][find_c][0] == cs[0]).all() and known[(h, w)][find_c][1] == cs[1]:
                            c = find_c
                    if (known[(h, w)][c][0] == rank_map).all():
                        mbs_list = known[(h, w)][c][1]
                        known_max = mbs_list[-1]
                        known_max = min(known_max, mbs)
                        for mbs_ in mbs_list:
                            if mbs_ > known_max:
                                mbs_list.pop(mbs_list.index(mbs_))
                                if len(mbs_list) == 0:
                                    known[(h, w)].pop(c)
                                    if len(known[(h, w)]) == 0:
                                        known.pop((h, w), None)
                                with open(record_file, "a") as fp:
                                    fp.write(f"{mbs_} deleted - current max {known_max} \n")
        else:
            with open(record_file, "a") as fp:
                fp.write(f"mbs={mbs}, h={h}, w={w}, {rank_map} : {gt_cost}\n")
        simulated_settings.append(((mbs, h, w, rank_map), gt_cost))

time_all_finish = time.time()
time_used = time_all_finish - time_all_start
print(f"finish random search with {budget} iterations in {time_used}")
# sorted simulated settings
sorted_settings = sorted(simulated_settings, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"{item}")

