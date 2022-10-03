"""
Utility functions.
"""

def comm_table(type, volume, bandwidths):
    """
        type: collective operation types: "allreduce" or "allgather"
        volume(int): communication volumne
        sb (int): slowest bandwidth link
        nsb (int): number of node that contains GPU with sb link
    """
    
    # TODO: Double check this formula.
    if type == "allreduce":
        n = len(bandwidths)
        bandwidth_list = list(bandwidths.values())

        time = 0
        for i in range(n-1):
            time += 1 / min(bandwidth_list[i], bandwidth_list[(i+1)%n])
        return time * volume / n
    else:
        raise NotImplementedError()

    """

    slowest_link = float('inf')
    slowest_link_nodes = set()
    num_slowest_link = 0
    for k, v in bandwidth_dict.items():
        for _k, _v in bandwidth_dict.items():
        # Assuming intra-node does not take time
            if k != _k:
                bandwidth = min(v, _v)
                if bandwidth < slowest_link:
                    slowest_link = bandwidth
                    num_slowest_link = 1
                elif bandwidth
        #return 2 * (nsb - 1)* volume / (nsb * sb)
    """

"""
Reverse mapping rank_map. For example, rank_map: {node1: [0,2], node2: [1,3]}
results in {0 : node1, 2 : node1, 1 : node2, 3 : node2}
"""
def rank2node(rank_map):
    ret = dict()
    for k, v in rank_map.items():
        for i in v:
            ret[i] = k
    return ret

def factor(integer):
    factors = []
    for i in range(1,integer+1):
        if integer % i == 0:
            factors.append(i)
    return factors

def estimate_alpha():
    pass

def estimate_beta():
    pass

def estimate_gamma():
    return 0
