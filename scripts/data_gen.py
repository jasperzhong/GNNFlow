import torch

rand_de = 172

edge_feats = torch.randn(1293103, rand_de)

torch.save(edge_feats, '/home/ubuntu/data/LASTFM/edge_features.pt')