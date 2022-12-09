import torch
import dgl
import pandas as pd
import numpy as np

dataset_name = 'GDELT'
# slice_end_idx = 100000000

num_partitions = 4
undirected = False

# df = pd.read_csv('/data/tgl/{}/edges.csv'.format(dataset_name)) # HKU Lab Machine
df = pd.read_csv('/home/ubuntu/data/{}/edges.csv'.format(dataset_name)) # AWS

print('df length before slicing is {}\n'.format(len(df)))


# GDELT NO SLICE!
# df = df[:slice_end_idx]

print('df length after slicing is {}\n'.format(len(df)))

src_nodes = df['src'].values.astype(np.int64)
dst_nodes = df['dst'].values.astype(np.int64)

if undirected:
	src_nodes_ext = np.concatenate([src_nodes, dst_nodes])
	dst_nodes_ext = np.concatenate([dst_nodes, src_nodes])
	src_nodes = torch.from_numpy(src_nodes_ext)
	dst_nodes = torch.from_numpy(dst_nodes_ext)
else:
	src_nodes = torch.from_numpy(src_nodes)
	dst_nodes = torch.from_numpy(dst_nodes)

max_node_id = int(torch.max(torch.max(src_nodes), torch.max(dst_nodes)))

visited = torch.zeros(max_node_id + 1, dtype=torch.int32)

visited[src_nodes] = 1

gte = dgl.graph((src_nodes, dst_nodes))
g = dgl.to_simple(gte)

print(len(src_nodes))

b_ntype = torch.zeros(g.num_nodes(), dtype=torch.int8)

pt = dgl.metis_partition_assignment(g, num_partitions, b_ntype, True, "k-way", "cut")

print(max_node_id, len(pt))

cnt = 0
for i in range(len(pt)):
	if visited[i].item() == 0:
		pt[i] = -1
		cnt += 1

print('Partition Finished and Find {} single nodes. Saving...\n'.format(cnt))
# torch.save(pt, '/home/yczhong/repos/GNNFlow/partition_data/{}_metis_partition.pt'.format(dataset_name)) # HKU Lab Machine
torch.save(pt, '/home/ubuntu/repos/GNNFlow/partition_data/{}_metis_partition.pt'.format(dataset_name)) # AWS





