from collections import defaultdict
import time
import json
import copy

import subprocess
import sys
import os

import torch
import torch.nn as nn
import numpy as np

from amp_utils import rank2axis, axis2rank, get_host
from pipe import pipe_ds, pipe_ast, pipe_cost, pipe_uniform, pipe_gpt2

home_dir = os.environ['HOME'] 
workdir_path = os.path.join(home_dir, "AMP/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")
example_path = os.path.join(workdir_path, "examples")
sys.path.append(workdir_path)
sys.path.append(example_path)

class AMP(nn.Module):
    def __init__(self, model_config, exp_name):
        
        super().__init__()
        self.model_config = model_config
        self.exp_name = "init_" + exp_name 
        self.model_type = model_config["type"]
        assert self.model_type == "gpt2" 
        self.init_param()
        
    def init_param(self):
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
 
        config_h = int((self.model_config["hidden_size"]).item())
        config_n = int(n)

        json_path = os.path.join(example_path, "ds_config.json")

        self.profile_cost = {}
        for mp_size in [1,2,4]:
            # known_cost directory stores the real forward time with correponding model parallel degree.
            known_record = f"known_cost/{self.model_type}_P3_{mp_size}"
            cur_profile_cost1 = 3 * np.load(f"{known_record}.npy")
            known_record = f"known_cost/{self.model_type}_G4_{mp_size}"
            cur_profile_cost2 = 3 * np.load(f"{known_record}.npy")

            # average between different speed of GPUs
            cur_profile_cost = cur_profile_cost1 * 0.75 + cur_profile_cost2 * 0.25
            self.profile_cost[str(mp_size)] = cur_profile_cost
            print(f"using profile cost with mp_size {mp_size}: {cur_profile_cost}")

        
    def forward(self, args):
        model_type = self.model_type
        config, bs, micro_bs, cluster_info, model_config, oth = args
        amp_config = {"profile_cost" : self.profile_cost}
        rank_map, partition, amp_pred = predict(config, bs, micro_bs, cluster_info, model_config, amp_config, oth)
        return rank_map, partition, amp_pred
        
# pipeline communication cost, return shape: (L-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, amp_config, dp_index=0):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _layer = ["embed2h", "noop"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
  
    _layer.extend(["noop","noop", "embed2v", "noop"])
    _num_layer = len(_layer)
      
    # build layer activation lookup table. Noop exatly has the same activation as the previous op.
    # Leave bs factor outside.
    layer_volume = []
    last_volume = torch.zeros(1,)
    for i in range(_num_layer):
        layer_type = _layer[i]
        if layer_type == "embed2h" or layer_type == "transformer_layer":
            last_volume = bs * s * h
            layer_volume.append(last_volume)
        elif layer_type == "embed2v":
            last_volume = bs * s * v / mp
            layer_volume.append(last_volume)
        elif layer_type == "noop":
            layer_volume.append(last_volume)
        else:
            raise RuntimeError("Unknown layer type.")
            
    # Build communication cost between pipeline stages by looking up the cluster information
    cost_c = torch.zeros((int(dp.item()), _num_layer-1, int(pp.item()-1)))
    for i in range(int(dp.item())):    
        for j in range(int(pp.item()-1)):
            # get the slowest mp gpu connection
            slowest_bandwidth = np.inf
            for k in range(int(mp.item())):    
                rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                rank_peer = axis2rank(axis=(j+1,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                node_peer = rank_node_map[int(rank_peer.item())]
                
                if node_cur != node_peer: 
                    cur_bandwidth = min(cluster_info[node_cur][0], cluster_info[node_peer][0])
                else:
                    cur_bandwidth = cluster_info[node_cur][1]
                if cur_bandwidth < slowest_bandwidth:
                    slowest_bandwidth = cur_bandwidth
            for k in range(_num_layer-1):
                cost_c[i][k][j] = layer_volume[k]  / slowest_bandwidth
            
    cost_c = torch.mean(cost_c, dim=0)
    return cost_c

# execution cost for one layer, return shape (L,)
def get_cost_e(cluster_info, model_config, parallel_config, amp_config):    

    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    profile_cost = amp_config["profile_cost"]
    _layer = ["embed2h", "noop"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    
    _layer.extend(["noop","noop", "embed2v", "noop"])
    _num_layer = len(_layer)
            
    cost_e = np.zeros((int(dp.item()), _num_layer))

    # Avegrage across pipelines
    # We have two choices here:
    # (1) Estimate execution cost and model parallelism cost ourselves;
    # (2) Directly Use profile cost, which includes model parallelism cost.
    for i in range(int(dp.item())):
        assert _num_layer == len(profile_cost["1"]), "predicted number of layers not equal to actual"
         
        mp_avg = 0 # TODO: clean SA code to update mp_avg
        for layer_id in range(_num_layer):
            layer_type = _layer[layer_id]
            cur_layer = bs * profile_cost[str(int(mp.item()))][layer_id]
                
            if layer_type == "embed2h":
                pass
            elif layer_type == "embed2v":
                cur_layer += (v * h / mp * mp_avg).item()
            elif layer_type == "transformer_layer":
                cur_layer += ((7*h**2/mp + 2*bs*s*h) * mp_avg).item()
            elif layer_type == "noop":
                pass
            else:
                raise RuntimeError("Unknown layer type.")
            cost_e[i][layer_id] = cur_layer
    
    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = torch.mean(cost_e, dim=0)
    return cost_e

def dp_cost(config, cluster_info,model_config, parallel_config, amp_config, partition):
    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]
    
    _layer = ["embed2h", "noop"]
    for i in range(int(n.item())):
        _layer.append("transformer_layer")
    
    _layer.extend(["noop","noop", "embed2v", "noop"])
    _num_layer = len(_layer)
        
    # First translate to deepspeed partition form
    ds_partition = [0]
    print(f"partition: {partition}")
    for i in range(len(partition)):
        ds_partition.append(ds_partition[-1]+partition[i])
    print(ds_partition, _num_layer)
    assert ds_partition[-1] == _num_layer
    assert len(ds_partition) == pp + 1
                
    # should be per-dp_group time
    max_dp = torch.zeros(1,)
    for i in range(int(pp.item())):
        for j in range(int(mp.item())):
            
            slowest = float("inf")
            for k in range(int(dp.item())):
                rank_cur = axis2rank(axis=(i,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_cur = rank_node_map[int(rank_cur.item())]
                    
                    
                rank_next = axis2rank(axis=(i,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                node_next = rank_node_map[int(rank_next.item())]
                    
                    
                if node_cur == node_next:
                    connectivity = cluster_info[node_cur][1]
                else:
                    connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                        
            slowest = min(slowest, connectivity)
            dp_const = 2 * (dp-1) / (dp * slowest)
            dp_const = torch.tensor([dp_const])
                
            param_count = torch.zeros(1,)
            counted = False
            for layer_id in range(ds_partition[i], ds_partition[i+1]):
                layer_type = _layer[layer_id]
                if layer_type == "embed2h" or layer_type == "embed2v":
                    if not counted:
                        counted = True
                        param_count += h * v / mp
                elif layer_type == "transformer_layer":
                    param_count += 12 * h ** 2 / mp
                elif layer_type == "noop":
                    pass
                else:
                    raise RuntimeError("Unknown layer type.")
                        
            cur_dp = dp_const * param_count
            if cur_dp > max_dp:
                max_dp = cur_dp
                
    return ds_partition, max_dp

def predict(config, bs, mbs, cluster_info, model_config, amp_config, oth):
    L = model_config["num_layers"]
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        m = oth["mp_deg"]
        n = oth["dp_deg"]
        pp = oth["pp_deg"]                   
        
        # infer a GPU rank map                
        counter = 0    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1
                
        print(f"AMP estimate default to {rank_map}")
    
    # valid config, inferred from sa 
    else:
        config = torch.from_numpy(config)
        pp = torch.max(config).float()
        
        # infer rank_map: given node name, returns the global mapped rank(int) in (pp, dp, mp) order
        # rank_node_map: given rank, returns the node
        rank_map = defaultdict(list)
        rank_node_map = dict()
    
        if pp >= (L + 2):
            print(f"early return with pp={pp}, L={L}")
            return None, None, torch.tensor([float("inf")])
           
        m = oth["mp_deg"]
        n = oth["dp_deg"]
        assert pp == oth["pp_deg"]                   
        
        rank_counter = np.zeros(int(pp.item()))
            
        # infer a GPU rank map                    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                cur_pp = int(config[k][j] - 1)
                rank_map[j].append(int((rank_counter[cur_pp] + cur_pp * m * n).item()))
                rank_node_map[int((rank_counter[cur_pp] + cur_pp * m * n).item())] = j
                rank_counter[cur_pp] += 1
            
    # infer number of micro-batch size B
    B = bs / (n * mbs)
            
    parallel_config = {"mp" : m, "dp" : n, "pp" : pp, "micro_bs" : mbs, "rank_map" : rank_map, "rank_node_map": rank_node_map}
        
    cost_e = get_cost_e(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)
    cost_c = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config)
           
    if int(B.item()) == 1:
        partition, _ = pipe_uniform(int(L.item()), int(pp.item()))
        partition[0] += 2
        partition[-1] += 4
    else:
        partition, _ = pipe_ast(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
    
    print(f"amp gives partition: {partition}")
    cost = pipe_cost(L, cost_e, cost_c, pp, B, partition)
        
    # translate to ds form, add data parallelism cost
    ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition)
       
    cost += dp_side_cost
    print(ds_partition, cost, dp_side_cost)
    return rank_map, ds_partition, cost
