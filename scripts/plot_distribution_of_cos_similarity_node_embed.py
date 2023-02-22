import os
import matplotlib.pyplot as plt

import numpy as np
import torch

model = 'TGAT'
dataset = 'REDDIT'
layer = 2

def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

def load_node_embeds():
    files = os.listdir()
    files = [f for f in files if f.startswith('node_embeddings_{}_{}_layer{}'.format(model, dataset, layer))]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    cos_sim_list = []
    for x, y in pairwise(files[:20]):
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(y)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim)

        cos_sim_list.append(cos_sim)

    return cos_sim_list


if __name__ == '__main__':
    cos_sim_list = load_node_embeds()

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, cos_sim in enumerate(cos_sim_list):
        ax.plot(np.arange(len(cos_sim)), cos_sim, label='iter {}'.format((i+1)*500))

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.legend()
    ax.set_xlim((0, len(cos_sim)))
    ax.set_ylim((-1, 1))
    ax.grid(True, color='gray', linestyle='--')
    ax.set_title("Cos similarities of node embeddings of {} on {} (layer {})".format(model, dataset, layer))
    plt.savefig('cos_sim_{}_{}_layer{}.png'.format(model, dataset, layer), dpi=400, bbox_inches='tight')
