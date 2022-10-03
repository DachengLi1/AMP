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
from pipe import pipe_ds, pipe_dp, pipe_cost, pipe_uniform, pipe_gpt2

home_dir = os.environ['HOME'] 
sys.path.append(os.path.join(home_dir, "/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/"))
#sys.path.append(os.path.join(home_dir, "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism"))
sys.path.append(os.path.join(home_dir, "/home/ubuntu/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism"))

class AMP(nn.Module):
    def __init__(self, model_config, exp_name, estimate=True):
        
        super().__init__()
        self.model_config = model_config
        self.exp_name = "init_" + exp_name 
        self.estimate = estimate
        self.model_type = model_config["type"]
        assert self.model_type == "gpt2" or self.model_type == "transgan" 
        #print(self.model_type, "haha")
        self.init_param()
        
    def init_param(self):
        home_path = os.environ['HOME']
        h = float(self.model_config["hidden_size"].item())
        n = float(self.model_config["num_layers"].item())
        s = float(self.model_config["sequence_length"].item())
        v = float(self.model_config["vocab_size"].item())
 
        config_h = int((self.model_config["hidden_size"]).item())
        config_n = int(n)

        json_path = os.path.join(home_dir, 'DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_config.json')
        
        if self.model_type == "gpt2":
            alpha_path = os.path.join(home_dir,"DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_alpha.sh")          
            alpha_conf = f" {config_n} {config_h} {self.exp_name}"
        else:
            alpha_path = os.path.join(home_dir,"DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_transgan_alpha.sh")            
            alpha_conf = f" {config_h} {self.exp_name}"
        
        dir_path = os.path.join(home_path, "amp_simulate")
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        record_path = os.path.join(dir_path, f"{self.exp_name}")
        # last run
        if os.path.exists(record_path):
            os.remove(record_path)
        # recreate the record file
        with open(record_path, "w") as tf:
            pass
        
        self.cost_single = {}

        time_profile_start = time.time()

        #if self.estimate:
            #for mp_size in [1,2,4]:
            #    known_record = f"known_cost/{self.model_type}_P3_{mp_size}"
                # if we have profile this setting
                #if not os.path.exist(known_record):
                # No need to run several mp for gpt2 model
                #if self.model_type == "gpt2" and mp_size > 1:
                #    continue
                # last run
            #    if os.path.exists(record_path):
            #        os.remove(record_path)
                # recreate the record file
            #    with open(record_path, "w") as tf:
            #        pass
                
            #    with open(json_path) as json_file:
            #        dict = json.load(json_file)
            #        dict["train_micro_batch_size_per_gpu"] = 1
            #        dict["gradient_accumulation_steps"] = 1

            #    with open(json_path, 'w') as outfile:
            #        json.dump(dict, outfile)
        
            #    os.environ["amp_record_path"] = record_path
                #os.environ["amp_mp_value"] = None
                #os.environ["amp_profile"] = "True"
               
           #     alpha_cmd = "bash " + alpha_path + alpha_conf + " " + str(mp_size)
            #    print(alpha_cmd)
                #subprocess.run(alpha_cmd, shell=True)
                # forward + backward
            #    cur_cost = 3 * np.load(f"{record_path}.npy")
            #    np.save(known_record, cur_cost / 3)
            #    self.cost_single[str(mp_size)] = cur_cost
       #         print(f"using single cost with mp_size {mp_size}: {cur_cost} with sum {np.sum(cur_cost)} std {np.std(cur_cost)}")
        if self.estimate:
            for mp_size in [1,2,4]:
                known_record = f"known_cost/{self.model_type}_P3_{mp_size}"
                cur_cost = 3 * np.load(f"{known_record}.npy")
                self.cost_single[str(mp_size)] = cur_cost
                print(f"using single cost with mp_size {mp_size}: {cur_cost} with sum {np.sum(cur_cost)} std {np.std(cur_cost)}")
        else:
            self.cost_single = None

        time_profile_end = time.time()
        time_profile_used = time_profile_end - time_profile_start
        print(f"profile time: {time_profile_used}")
        print(f"execution cost: {self.cost_single}")
        
    def forward(self, args, model_type="gpt2"):
        #os.environ["amp_profile"] = "False"
        model_type = self.model_type
        configs, bs_list, micro_bs_list, cluster_info, model_config, oth = args
        amp_config = {"cost_single" : self.cost_single}
        if isinstance(bs_list, list):
            rank_maps, partitions, amp_preds = predict_multi(configs, bs_list, micro_bs_list, cluster_info, model_config, amp_config, oth, model_type)
        else:
            assert isinstance(bs_list, int)
            rank_maps, partitions, amp_preds = predict_single(configs, bs_list, micro_bs_list, cluster_info, model_config, amp_config, oth, model_type)
        return rank_maps, partitions, amp_preds
        
def rank_loss(preds, target):
    ret = []
    for i in range(len(preds)):
        for j in range(len(preds)):
            if target[i] > target[j]:
                diff = preds[i]-preds[j]
                ret.append(torch.relu(-diff) + torch.log1p(torch.exp(-torch.abs(diff))))
    ret_ = torch.zeros(1,)
    for i in ret:
        if not torch.isinf(i):
            ret_ += i
    #print(preds, target)
    return ret_

# return shape: (L-1, pp-1)
def get_cost_c(cluster_info, model_config, parallel_config, amp_config, dp_index=0, model_type="GPT2"):
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
    
    if model_type == "gpt2":
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
              # TODO 
              last_volume = bs * s * h
              layer_volume.append(last_volume)
          elif layer_type == "embed2v":
              last_volume = bs * s * v / mp
              layer_volume.append(last_volume)
          elif layer_type == "noop":
              layer_volume.append(last_volume)
          else:
              raise RuntimeError("Unknown layer type.")
    else:
        assert model_type == "transgan"
        cost_single = amp_config["cost_single"]
        depth = model_config["depth"]
        bottom = model_config["bottom"]
        _num_layer = 1+depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]

        layer_volume = [bs * h * bottom ** 2] * _num_layer
            
            
    cost_c = torch.zeros((int(dp.item()), _num_layer-1, int(pp.item()-1)))
    # TODO: loop through all dp, and get the average
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
            #print(slowest_bandwidth)
            for k in range(_num_layer-1):
                cost_c[i][k][j] = layer_volume[k]  / slowest_bandwidth
            
    cost_c = torch.mean(cost_c, dim=0)
    print(f"using cost_c: {cost_c}")
    return cost_c

# execution cost, return shape (L,)
def get_cost_e(cluster_info, model_config, parallel_config, amp_config, model_type):    

    h = model_config["hidden_size"]
    s = model_config["sequence_length"]
    n = model_config["num_layers"]
    v = model_config["vocab_size"]
    orig_bs = parallel_config["micro_bs"]
    rank_map = parallel_config["rank_map"]
    rank_node_map = parallel_config["rank_node_map"]
    mp = parallel_config["mp"]
    dp = parallel_config["dp"]
    pp = parallel_config["pp"]

    #print("I am using", alpha)
    
    if model_type == "gpt2":
        cost_single = amp_config["cost_single"]
        _layer = ["embed2h", "noop"]
        for i in range(int(n.item())):
            _layer.append("transformer_layer")
    
        _layer.extend(["noop","noop", "embed2v", "noop"])
        _num_layer = len(_layer)
            
        cost_e = np.zeros((int(dp.item()), _num_layer))
        # TODO: loop through all dp, and get the average
        for i in range(int(dp.item())):
            # TODO: first find on average how many cross node (we dont know which layer in which pp)
            # Compute the constant along the pipeline: 2*(N-1)/(NB)
            mp_avg = torch.zeros(1,)
            for j in range(int(pp.item())):
                slowest = float("inf")
                for k in range(int(mp.item())):
                    rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_cur = rank_node_map[int(rank_cur.item())]
                    print(rank_cur, node_cur)
                    
                    rank_next = axis2rank(axis=(j,i,(k+1)%(mp.item())), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_next = rank_node_map[int(rank_next.item())]
                    
                    if node_cur == node_next:
                        connectivity = cluster_info[node_cur][1]
                    else:
                        connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                slowest = min(slowest, connectivity)
                mp_avg += 2 * (mp-1) / (mp * slowest)
            
            mp_avg /= pp
            assert _num_layer == len(cost_single["1"]), "predicted number of layers not equal to actual"
            mp_total = 0
            mp_total_volume = 0
            for layer_id in range(_num_layer):
                layer_type = _layer[layer_id]
                cur_layer = orig_bs * cost_single["1"][layer_id] / mp.item()
                
                if layer_type == "embed2h":
                    pass
                elif layer_type == "embed2v":
                    cur_layer += (v * h / mp * mp_avg).item()
                    mp_total += (v * h / mp * mp_avg).item()
                    mp_total_volume += (v * h / mp).item()
                    print(f"debug: embed2v predicted mp cost:{v * h / mp * mp_avg}")
                    print(f"debug: embed2v predicted mp volume:{v * h / mp}")
                elif layer_type == "transformer_layer":
                    cur_layer += ((7*h**2/mp + 2*orig_bs*s*h) * mp_avg).item()
                    mp_total += ((7*h**2/mp + 2*orig_bs*s*h) * mp_avg).item()
                    mp_total_volume += ((7*h**2/mp + 2*orig_bs*s*h)).item()
                    print(((7*h**2/mp + 2*orig_bs*s*h)).item())
                elif layer_type == "noop":
                    pass
                else:
                    raise RuntimeError("Unknown layer type.")
                cost_e[i][layer_id] = cur_layer
                #print(cost_e[i])
            print(f"using cost_e i : -------- -------- {cost_e[i]}")
            print(f"debug: total execution cost:{np.sum(cost_e[i])} with predicted mp cost:{mp_total} volume: {mp_total_volume} and avg: {mp_avg}")
    else:
        assert model_type == "transgan"
        # (bs, s, h)
        cost_single = amp_config["cost_single"]
        depth = model_config["depth"]
        bottom = model_config["bottom"]
        
        cost_e = [None] * int(dp.item())#torch.zeros((int(dp.item()), 28))
        for i in range(int(dp.item())):
            cost_e[i] = []
        #print(f"using bs: {bs}")
        # TODO: loop through all dp, and get the average
        for i in range(int(dp.item())):
        # TODO: first find on average how many cross node (we dont know which layer in which pp)
        # Compute the constant along the pipeline: 2*(N-1)/(NB)
            mp_avg = torch.zeros(1,)
            for j in range(int(pp.item())):
                slowest = float("inf")
                for k in range(int(mp.item())):
                    rank_cur = axis2rank(axis=(j,i,k), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_cur = rank_node_map[int(rank_cur.item())]
                    print(rank_cur, node_cur)
                    
                    rank_next = axis2rank(axis=(j,i,(k+1)%(mp.item())), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_next = rank_node_map[int(rank_next.item())]
                    
                    if node_cur == node_next:
                        connectivity = cluster_info[node_cur][1]
                    else:
                        connectivity = min(cluster_info[node_cur][0], cluster_info[node_next][0])
                slowest = min(slowest, connectivity)
                mp_avg += 2 * (mp-1) / (mp * slowest)
                
            mp_avg /= pp
            cost_e[i].append(cost_single[str(int(mp.item()))][0])
            
            if mp == 0: 
                assert mp_avg == 0
                
            bs = orig_bs
           
            mp_total = 0
            s = bottom**2
            first_layer_id = 3
            for inc in range(depth[0]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (cost_single[str(int(mp.item()))][cur_layer_id])
                #print(cur_layer)
                cur_layer += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                mp_total += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                #print(((7*h**2/mp + 2*s*bs*h) * mp_avg).item())
                cost_e[i].append(cur_layer)
            
            s = 4*bottom**2
            first_layer_id = 7 + depth[0]
            for inc in range(depth[1]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (cost_single[str(int(mp.item()))][cur_layer_id])
                cur_layer +=  ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                mp_total += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                cost_e[i].append(cur_layer) 
            
            s = 16*bottom**2
            first_layer_id = 9 + depth[0] + depth[1]
            for inc in range(depth[2]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (cost_single[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                mp_total += ((7*h**2/mp + 2*s*bs*h) * mp_avg).item()
                cost_e[i].append(cur_layer)
            
            bs *= 16
            s = 4*bottom**2
            first_layer_id = 15 + depth[0] + depth[1] + depth[2]
            for inc in range(depth[3]):
                cur_layer_id = first_layer_id + inc
                cur_layer =  orig_bs* (cost_single[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*(h/4)**2/mp + 2*s*bs*(h/4)) * mp_avg).item()
                mp_total += ((7*(h/4)**2/mp + 2*s*bs*(h/4)) * mp_avg).item()
                cost_e[i].append(cur_layer)
              
            bs *= 4
            s = 4*bottom**2
            first_layer_id = 22 + depth[0] + depth[1] + depth[2] + depth[3]
            for inc in range(depth[4]):
                cur_layer_id = first_layer_id + inc
                cur_layer =  orig_bs * (cost_single[str(int(mp.item()))][cur_layer_id])
                cur_layer += ((7*(h/16)**2/mp + 2*s*bs*(h/16)) * mp_avg).item()
                mp_total += ((7*(h/16)**2/mp + 2*s*bs*(h/16)) * mp_avg).item()
                cost_e[i].append(cur_layer)
             
            bs *= 4
            s = 4*bottom**2
            first_layer_id = 29 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4]
            for inc in range(depth[5]):
                cur_layer_id = first_layer_id + inc
                cur_layer = orig_bs * (cost_single[str(int(mp.item()))][cur_layer_id])
                #print(f"now {cur_layer} {}{cost_single[cur_layer_id] / (mp).item()}")
                cur_layer += ((7*(h/64)**2/mp + 2*s*bs*(h/64)) * mp_avg).item()
                mp_total +=  ((7*(h/64)**2/mp + 2*s*bs*(h/64)) * mp_avg).item()
                cost_e[i].append(cur_layer) 
                       
            #print(f"debug: {len(cost_e[i])} {len(cost_single)}")
            assert len(cost_e[i]) == 1 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]
            cost_e[i] = np.asarray(cost_e[i])
            print(f"debug: total execution cost:{np.sum(cost_e[i])} with predicted mp cost:{mp_total}")
    
    cost_e = torch.from_numpy(np.stack(cost_e, axis=0))            
    cost_e = torch.mean(cost_e, dim=0)
    print(f"using cost_e: {cost_e} with sum {torch.sum(cost_e)}" )
    return cost_e

def dp_cost(config, cluster_info,model_config, parallel_config, amp_config, partition, model_type):
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
    
    if model_type == "gpt2":
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
        # embed2h and noop at the beginning
        #for i in range(1, len(ds_partition)):
        #    ds_partition[i] += 2
        # embed2v and 3 noop at the end
        #ds_partition[-1] += 4
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
                    
                    print("dp: ", rank_cur, node_cur)
                    
                    rank_next = axis2rank(axis=(i,(k+1)%(dp.item()),j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_next = rank_node_map[int(rank_next.item())]
                    
                    print("dp next: ", rank_next, node_next)
                    
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
                        print(f"embed size {h * v / mp}")
                    elif layer_type == "transformer_layer":
                        param_count += 12 * h ** 2 / mp
                    elif layer_type == "noop":
                        pass
                    else:
                        raise RuntimeError("Unknown layer type.")
                        
                print(f"dp: {dp_const} and param {param_count}")
                cur_dp = dp_const * param_count
                if cur_dp > max_dp:
                    max_dp = cur_dp
                
                #print(max_dp, beta, dp_const, param_count)
    else:
        assert model_type == "transgan"
        depth = model_config["depth"]
        # First translate to deepspeed partition form
        work_layers = [0]
        h_map = dict()
        #work_layers = [0, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17, 18, 19, 20, 27, 28, 29, 30, 38, 39, 40, 41, 49, 50, 51, 52]
        #h_map = {"3": h, "4": h, "5": h, "6": h, "7": h,
        #         "11": h,"12": h,"13": h,"14": h,
        #         "17": h, "18":h, "19": h, "20": h,
        #         "27": h/4, "28": h/4,  "29": h/4, "30": h/4, 
        #         "38": h/16, "39": h/16, "40": h/16, "41": h/16, 
        #         "49": h/64, "50": h/64, "51": h/64, "52": h/64}
                 
        
        first_layer_id = 3
        for inc in range(depth[0]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h
        
        first_layer_id = 7 + depth[0]
        for inc in range(depth[1]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h
          
        first_layer_id = 9 + depth[0] + depth[1]
        for inc in range(depth[2]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h
            
        first_layer_id = 15 + depth[0] + depth[1] + depth[2]
        for inc in range(depth[3]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h/4
              
        first_layer_id = 22 + depth[0] + depth[1] + depth[2] + depth[3]
        for inc in range(depth[4]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h/16
             
        first_layer_id = 29 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4]
        for inc in range(depth[5]):
            cur_layer_id = first_layer_id + inc
            work_layers.append(cur_layer_id)
            h_map[str(cur_layer_id)] = h/64 
                
        ptr = -1
        ds_partition = [0]
        print(partition, work_layers, len(work_layers))
        for i in partition:
            ptr += i
            last_layer_id = work_layers[ptr] 
            ds_partition.append(last_layer_id + 1)
        ds_partition[-1] = 32 + depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]
        
      #  print(ds_partition)    
        assert len(ds_partition) == pp + 1
        
        # should be per-dp_group time
        max_dp = torch.zeros(1,)
        for i in range(int(pp.item())):
            for j in range(int(mp.item())):
                slowest = float("inf")
                node_counter = defaultdict(set)
               
                first_node = None
                dp_const = 0
                for k in range(int(dp.item())):
                    rank_cur = axis2rank(axis=(i,k,j), mp_deg=mp, dp_deg=dp, pp_deg=pp)
                    node_cur = rank_node_map[int(rank_cur.item())]
                    if first_node is None:
                        first_node = node_cur
                        assert first_node is not None
                    else:
                        if node_cur == first_node:
                            dp_const += 2 / (dp * cluster_info[node_cur][1])
                        else:
                            dp_const += 2 / (dp * min(cluster_info[node_cur][0], cluster_info[first_node][0]))
                 
                dp_const = torch.tensor([dp_const])
                
                param_count = torch.zeros(1,)
                for layer_id in range(ds_partition[i], ds_partition[i+1]):
                    if str(layer_id) in h_map:
                        param_count += 12 * h_map[str(layer_id)] ** 2 / mp
                
                print(f"dp: {dp_const} and param {param_count}")
                cur_dp = dp_const * param_count
                if cur_dp > max_dp:
                    max_dp = cur_dp
                
                #print(max_dp, beta, dp_const, param_count)
    
            
    return ds_partition, max_dp

def predict_multi(configs, bs_list, micro_bs_list, cluster_info, model_config, amp_config, oth_list, model_type):
    costs = torch.zeros(len(configs))
    rank_maps = []
    partitions = []
    for i in range(len(configs)):
        rank_map, partition, cost = predict_single(configs[i], bs_list[i], micro_bs_list[i], cluster_info, model_config, amp_config, oth_list[i], model_type)
        costs[i] = cost
        rank_maps.append(rank_map)
        partitions.append(partition)
    
    return rank_maps, partitions, costs

def predict_single(config, bs, mbs, cluster_info, model_config, amp_config, oth, model_type):
    # whether this is model fitting or inference
    
    L = model_config["num_layers"]
         
    cost = torch.zeros(1,)
    M, N = config.shape
    config = np.asarray(config)
    #config = torch.from_numpy(np.asarray(config))
       
    if np.all(config == -1):
        rank_map = defaultdict(list)
        rank_node_map = dict()

        m = oth["orig_mp"]
        n = oth["orig_dp"]
        pp = oth["orig_pp"]                   
        
        #config = np.transpose(np.asarray(list(range(M*N))).reshape((N, M)))
        # infer a GPU rank map                
        counter = 0    
        for j in range(N):
            for k in range(M):
                # TODO: bad code here, config counts from 1
                rank_map[j].append(counter)
                rank_node_map[counter] = j
                counter += 1
                
        print(f"---------------------------------------------- AMP estimate default to {rank_map} --------------------------------")
        
    
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
           
        m = oth["orig_mp"]
        n = oth["orig_dp"]
        print(mbs, oth)
        assert pp == oth["orig_pp"]                   
        
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
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config, model_type=model_type)
    cost_c = get_cost_c(cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, amp_config=amp_config, model_type=model_type)
           
    #partition, _ = pipe_dp(int(L.item()), np.asarray(cost_e.detach()), np.asarray(cost_c.detach()), int(pp.item()), int(B.item()))
    if model_type == "gpt2":
        if int(B.item()) == 1:
            partition, _ = pipe_uniform(int(L.item()), int(pp.item()))
            partition[0] += 2
            partition[-1] += 4
        else:
            partition, _ = pipe_dp(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
        #partition, _ = pipe_gpt2(int(L.item()), int(pp.item()))
    else:
        # balance layer if no need to balance time
        if int(B.item()) == 1:
            partition, _ = pipe_uniform(1+sum(model_config["depth"]), int(pp.item()))
        else:
            partition, _ = pipe_dp(len(cost_e), np.asarray(cost_e), np.asarray(cost_c), int(pp.item()), int(B.item()))
    print(f"amp gives partition: {partition}")
    cost = pipe_cost(L, cost_e, cost_c, pp, B, partition)
        
    # translate to ds form, add data parallelism cost
    ds_partition, dp_side_cost = dp_cost(config, cluster_info=cluster_info, 
                        model_config=model_config, parallel_config=parallel_config, 
                        amp_config=amp_config, partition=partition, model_type=model_type)
       
    cost += dp_side_cost
    print(ds_partition, cost, dp_side_cost)
    return rank_map, ds_partition, cost
