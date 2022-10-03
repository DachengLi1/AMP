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

from sa import generate_initial, neighbor, cool_down, candidate
from cost import get_cost_c, get_cost_e, rank_loss, AMP
from pipe import pipe_dp, pipe_ds
from amp_utils import simulate, to_float_torch

# cluster information

# number of GPU per node, number of nodes
M = 1
N = 1

data_bs = 2

home_path = os.environ['HOME']
dir_path = os.path.join(home_path, 'amp_main_logs')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

#TODO: Find DGX-2 box config
cluster_info = {}

# inter-node bandwidth
for i in range(N):
    cluster_info[i] = torch.tensor([10]).float()

# Model information: 16 layers network, 3 micro-batches
model_config = {"hidden_size": torch.tensor([0.1024]).float(), 
                "sequence_length": torch.tensor([1.024]).float(), 
                "num_layers": torch.tensor([24]).float(), 
                "vocab_size":torch.tensor([0.52256]).float()}

config_h = int((10000 * model_config["hidden_size"]).item())
config_n = int(model_config["num_layers"].item())
time_stamp = int(time.time())
exp_name = f"orca_8_{config_h}_{config_n}"
record_file = f"{os.path.join(dir_path, exp_name)}_{time_stamp}.txt"

# save this name to env
os.environ["amp_log_path"] = record_file

rank_maps=[None] #[ {0:[0,1,8,9], 1:[0,3,10,11], 2:[4,5,12,13], 3:[6,7,14,15]} ]
partitions = [None] #[[0,15,30]]
gbs = torch.tensor([4])
micro_bs_list = [torch.tensor([2])]
oth_list = [{"orig_dp": torch.tensor([1]), "orig_pp": torch.tensor([1]), "orig_mp": torch.tensor([1])}]

simulate(rank_maps, partitions, gbs, micro_bs_list, model_config, oth_list, exp_name)
