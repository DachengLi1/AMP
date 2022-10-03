from deepspeed.amp import query_amp_topo

orca_cluster = "../orca_resource_32.yml"
gpt2_model_config = {"name": "gpt2", "sequence_length": 1024, "hidden_size": 1024, "num_layers": 24, "vocab_size": 50512}

ret = query_amp_topo(orca_cluster, gpt2_model_config, 50)



