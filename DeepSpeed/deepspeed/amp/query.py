import argparse
from .AmpModel import gpt2
from .optimizer import optimizer, MCMCOptimizer
from .cluster import cluster as cl
def parse_args():

    parser = argparse.ArgumentParser(description='Runtime analysis.')
    parser.add_argument('--name', type=str, help='model name, currently supports ["bert"]')
    parser.add_argument('--num_layers', type=int, help='number of layers for transformer')
    parser.add_argument('--hidden_size', type=int, help='hidden size for transformer')
    parser.add_argument('--num_attn_heads', type=int, help='number of attention heads for transformer')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--seq_length', type=int, help='sequence length for transformer')
    parser.add_argument('--pos_embedding', type=int, help='position embedding size for transformer')


    return parser.parse_args()

"""
Entry point of Amp logic. 

Receive the configuration of a cluster and model, and return:
    (1) The rank of each GPU
    (2) The placement group lists
"""

def query_amp_topo(cluster_resource, model_config, bs, budget=50):
    
    #test_ranks = [2,0,3,5,7,1,4,6]
    #ret = dict()
    #count = 0
    #for k, v in cluster.items():
    #    ret[k] = [test_ranks[count]]
    #    count += 1
    
    #return ret, 4, 2

    # The cluster information is global w.r.t our search.

    cluster = cl(cluster_resource)
    print(cluster.get_info()) 
    name = model_config["name"]
    if name == "gpt2":
        model = gpt2(model_config)
    else:
        raise NotImplementedError()

    #optim = MCMCOptimizer(model, cluster, bs, budget)
    optim = HeuristicOptimizer(model, cluster, bs, budget)
    return optim.optimize()

