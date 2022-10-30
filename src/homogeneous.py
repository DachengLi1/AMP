# Script to reproduce homogeneous setting results

import math
import time
from collections import defaultdict
import operator
import shutil
import random
import os
import copy

from tqdm import tqdm

import numpy as np

import torch
from torch import optim as optim
import torch.nn as nn
import torch.nn.functional as F

from sa import amp_no_placement_strategy
from cost_homo import   AMP
from amp_utils import simulate, to_float_torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="Whether to run real trials")
parser.add_argument("--budget", type=int, default=-1, help="how many real trials to launch")

args = parser.parse_args()
# cluster information

time_s = time.time()
# number of GPU per node, number of nodes
M = 4
N = 4

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

cluster_info = {}

# inter-node bandwidth, intra-node bandwidth
for i in range(N):
    cluster_info[i] = [torch.tensor([50 * 1e9 / 32]).float(), torch.tensor([50 * 1e9 / 32]).float()]

model_config = {"hidden_size": torch.tensor([1024]).float(), 
                "sequence_length": torch.tensor([1024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([52256]).float(),
                "type":"gpt2"}

config_h = int((model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"homogeneous"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"
simulate_dir = os.path.join(home_path, "amp_simulate")
if not os.path.exists(simulate_dir):
    os.mkdir(simulate_dir)

# remove cache directory from last run
if os.path.exists(os.path.join(home_path, "tmp")):
    for root, dirs, files in os.walk(os.path.join(home_path, "tmp")):
        for f in files:
            os.unlink(os.path.join(root, f))

# save this name to env
os.environ["amp_log_path"] = record_file

global_bs = 32
model = AMP(model_config, exp_name)
assert (global_bs % M == 0) and (global_bs % N == 0), "global batch size is too irrgular"

want_simulate = [] 
feasible = {}

with open(record_file, "a") as fp:
    fp.write(f"{model_config}\n")                
    fp.write(f"gbs:{global_bs}\n")                
known = None
iter_count = 0

# Estimating best configurations
while True:
    ret = amp_no_placement_strategy(M=M, N=N, gbs=global_bs, known=known)
    if ret is None:
        break
    else:
        h, w, mbs, known = ret
        oth = {"mp_deg": torch.ones(1,)*h, "dp_deg": torch.ones(1,)*w, "pp_deg": torch.ones(1,)*(M*N/(h*w))}
        fake_config = np.ones((M,N)) * (-1)
        model_args = (fake_config, global_bs, mbs, cluster_info, model_config, oth)    
        
        with torch.no_grad():
            rank_map, partition, cost = model(model_args)
        
        want_simulate.append(((mbs, oth, rank_map, partition), cost))
    #if len(want_simulate) > 5:
    #    break
    iter_count += 1
time_e = time.time()
print(f"finish amp search without placement in {iter_count} iterations in {time_e - time_s}")

sorted_settings = sorted(want_simulate, key = lambda kv: kv[1])
with open(record_file, "a") as fp:
    for item in sorted_settings:
        fp.write(f"rank {sorted_settings.index(item)}: {item}")
        fp.write("\n")

# Run real trials to get ground truth runtime
if args.full:
    if args.budget == -1:
        budget = len(sorted_settings)
    else:
        budget = args.budget
    simulate_start = time.time()
    for i in range(budget):
        can = sorted_settings[i][0]
        rmap = None
        mbs = can[0]
        oth = can[1]
        partition = can[3]
        gt_cost = simulate([rmap], [partition], torch.ones(1,)*global_bs, to_float_torch([mbs]), model_config, [oth], exp_name)
        gt_cost = gt_cost[0]
        with open(record_file, "a") as fp:
            fp.write(f"Simulating result: {rmap}, {partition}, {mbs}, {oth}, with p_cost: {sorted_settings[i][1]}, r_cost: {gt_cost} \n")
            fp.write(f"running real trials till iter {i} takes {time.time() - time_s} \n")
