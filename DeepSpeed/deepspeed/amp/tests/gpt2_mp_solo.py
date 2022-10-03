from deepspeed.amp import query_amp_topo

orca_cluster = "../orca_resource_1.yml"
gpt2_model_config = {"name": "gpt2", "sequence_length": 1024, "hidden_size": 768, "num_layers": 12, "vocab_size": 50512}

ret = query_amp_topo(orca_cluster, gpt2_model_config, 50)



