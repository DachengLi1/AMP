from deepspeed.amp import cluster

cluster = cluster()
print(cluster.get_info())
