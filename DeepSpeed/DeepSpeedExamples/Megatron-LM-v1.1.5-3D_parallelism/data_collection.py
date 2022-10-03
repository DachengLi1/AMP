import subprocess
import sys
import os
import json

sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/")
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")

world_size = 8
record_file = "/users/dacheng2/amp_orca_8_data.txt"

if not os.path.exists(record_file):
    with open(record_file, 'w') as fp:
        pass
done = []

f = open(record_file, "r")
for line in f.readlines():
    done.append(line) 
f.close()

nlayer = 24
hidden = 1024
global_bs_list = [8,16,32, 64, 128]
#local_bs = 1
#B = 1

def factors(num):
    ret = []
    for i in range(1,num+1):
        if num % i == 0:
            ret.append(i)
    return ret

for global_bs in global_bs_list:
    for mp_size in factors(world_size):
       # if world_size % mp_size == 0:
        if hidden % mp_size == 0:
            for pp_size in factors(world_size // mp_size): #range(1, (world_size // mp_size)+1):
                if nlayer % pp_size == 0:

                    dp_size = world_size // (mp_size * pp_size)
                    #print(mp_size, dp_size, pp_size)
                    
                    if global_bs % dp_size == 0:
                  

                        for local_bs in factors(global_bs // dp_size):
                 #           local_bs = 1024
                            file_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh"

#                    subprocess.run("bash ~/clean_all.sh", shell=True)
                 
                            gas = int((global_bs / dp_size) / local_bs)
                            conf = f" {mp_size} {pp_size} {nlayer} {hidden} {local_bs} {gas}"
                            record = f"mp: {mp_size}, dp: {dp_size}, pp: {pp_size}, nlayer: {nlayer}, hidden: {hidden}, global_bs: {global_bs}, local_bs: {local_bs}, gas: {gas} \n"
                 
                            if record in done:
                                 continue
                
                    # Chenge the ds_config
                            json_path = '/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_config.json'
                            with open(json_path) as json_file:
                                dict = json.load(json_file)
                                dict["train_micro_batch_size_per_gpu"] = local_bs
                                dict["gradient_accumulation_steps"] = gas

                            with open(json_path, 'w') as outfile:
                                json.dump(dict, outfile)                               
 
                            cmd = "bash " + file_path + conf
                
                            f = open(record_file, "a")
                
                            f.write(record)
                            f.close()
                            
                            #try: 
                            process = subprocess.run(cmd, shell=True)
                            #if process.returncode == 0:
                            #    print(f"{conf} successed!")
                            #except subprocess.TimeoutExpired:
                            #    clean_cmd = "bash ~/clean_all.sh"
                            #    subprocess.run(clean_cmd, shell=True)
                            #    print(f"{conf} feiled!")
                            #    f = open(record_file, "a")
                            #    f.write("timeout \n")
                            #    f.close()
                            #    continue
