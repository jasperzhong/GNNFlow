import os
import matplotlib.pyplot as plt
from glob import glob

import numpy as np
import torch

model = 'TGN'
dataset = 'REDDIT'
layer = 1

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

# same iter -> different epochs
node_emd_kv = {}

def load_node_embeds(list_iters):
    files = glob('node_embeddings_{}_{}_layer{}_*'.format(model, dataset, layer))
    assert len(files) > 0
    for file in files:
        iter = file.split('layer{}'.format(layer))[-1].split('_')[2]
        iter = int(iter.split('.')[0])
        if iter not in node_emd_kv:
            node_emd_kv[iter] = []
        node_emd_kv[iter].append(file)        
    
    for k, v in node_emd_kv.items():
        node_emd_kv[k] = sorted(node_emd_kv[k], key=lambda x: int(x.split('layer{}'.format(layer))[-1].split('_')[1]))[::10][:2]

    cos_sim_lists = []
    for iter in list_iters:
        cos_sim_list = []
        for x, y in zip(node_emd_kv[iter], node_emd_kv[iter+20]):
            node_embed_1 = torch.from_numpy(np.load(x)).cuda()
            node_embed_2 = torch.from_numpy(np.load(y)).cuda()

            # compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(node_embed_1, node_embed_2, dim=1)

            # sort by cosine similarity
            cos_sim = cos_sim.cpu().numpy()
            cos_sim = np.sort(cos_sim)

            cos_sim_list.append(cos_sim)
        cos_sim_lists.append(cos_sim_list)

    return cos_sim_lists


if __name__ == '__main__':
    list_iters = [0, 300, 600, 700]
    cos_sim_lists = load_node_embeds(list_iters)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, cos_sim_list in enumerate(cos_sim_lists):
        for e, cos_sim in enumerate(cos_sim_list):
            ax.plot(np.arange(len(cos_sim)), cos_sim, label='epoch {} iter {}'.format(e*10, list_iters[i]))

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.legend()
    ax.set_xlim((0, len(cos_sim)))
    ax.set_ylim((0, 1))
    ax.grid(True, color='gray', linestyle='--')
    ax.set_title("Cos similarities of node embeddings of {} on {} (layer {})".format(model, dataset, layer))
    plt.savefig('cos_sim_{}_{}_layer{}.png'.format(model, dataset, layer), dpi=400, bbox_inches='tight')
