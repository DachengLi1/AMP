import subprocess
import sys
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/")
sys.path.append("/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism")

world_size = 16

for mp_size in range(1, world_size+1):
    if world_size % mp_size == 0:
        for pp_size in range(1, (world_size // mp_size)+1):
            if world_size % (mp_size * pp_size) == 0:
                dp_size = world_size // (mp_size * pp_size)
                #print(mp_size, dp_size, pp_size)
                file_path = "/users/dacheng2/DeepSpeed/DeepSpeedExamples/Megatron-LM-v1.1.5-3D_parallelism/examples/ds_pretrain_gpt2_pipe.sh"
                conf = f" {mp_size} {pp_size} 24 1024"
                cmd = "bash " + file_path + conf
                subprocess.run(cmd, shell=True)
                print(f"{conf} successed!")
