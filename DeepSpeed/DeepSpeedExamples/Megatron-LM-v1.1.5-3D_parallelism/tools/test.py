from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from amp_utils import rank2axis, axis2rank
from pipe import pipe_ds, pipe_dp

import subprocess
import sys
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/")
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")

class AMP(nn.Module):
    def __init__(self, model_config):
        
        super().__init__()
        self.model_config = model_config
        self.init_param()
        
        
    def init_param(self):
        h = self.model_config["hidden_size"]
        n = self.model_config["num_layers"]
        exp_name = f"orca_8_{h}_{n}"
        
        # guess alpha by running locally
        json_path = '/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_config.json'
        
        alpha_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_alpha.sh"
        
        beta_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_beta.sh"
        
        #gamma_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_gamma.sh"
        
        record_file = f"/users/dacheng2/{exp_name}.txt"
        
        alpha_conf = f"{n} {h} {exp_name}"
        
        with open(json_path) as json_file:
            dict = json.load(json_file)
            dict["train_micro_batch_size_per_gpu"] = 1
            dict["gradient_accumulation_steps"] = 1

        with open(json_path, 'w') as outfile:
            json.dump(dict, outfile)
        alpha_cmd = "bash " + alpha_path + alpha_conf
        
        subprocess.run(alpha_cmd, shell=True)
        
        
        beta_conf = f"{n} {h} {exp_name}"
        
        with open(json_path) as json_file:
            dict = json.load(json_file)
            dict["train_micro_batch_size_per_gpu"] = 1
            dict["gradient_accumulation_steps"] = 1

        with open(json_path, 'w') as outfile:
            json.dump(dict, outfile)
            
        beta_cmd = "bash " + beta_path + beta_conf
        
        subprocess.run(beta_cmd, shell=True)
        
        flop_count = flop(n=n, bs=1, h=h, s=1024, v=50304)
        comm_count = comm(n=n, bs=1, h=h, s=1024, v=50304, p=2)
        
        with open(record_file, "r") as f:
            lines = f.readlines()
            assert len(lines) == 2, "cannot run basic config"
            
            setting_1 = float(lines[0].rstrip())
            setting_2 = float(lines[1].rstrip())
            #setting_3 = float(lines[2].rstrip())
            
            alpha = setting_1 / flop_count
            self.alpha = torch.nn.Parameter(torch.ones(1,) * alpha)
            beta = (setting_2 / (setting_1 / 2)) / comm_count  
            self.beta = torch.nn.Parameter(torch.ones(1,) * beta)
            #gamma = 
        
        #self.alpha = torch.nn.Parameter(torch.ones(1,))
        #self.beta = torch.nn.Parameter(torch.ones(1,))

model = Amp()
print("hi", model.alpha, model.beta)
