import os
import matplotlib.pyplot as plt

import numpy as np
import torch


def load_node_embeds():
    files = os.listdir()
    files = [f for f in files if f.startswith('node_embeddings_')]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[::25]
    node_embed = torch.from_numpy(np.load(files[0])).cuda()
    cos_sim_list = []
    for f in files[1:]:
        node_embed_2 = torch.from_numpy(np.load(f)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(node_embed, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim)

        cos_sim_list.append(cos_sim)

        node_embed = node_embed_2

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
    plt.savefig('cos_sim.png', dpi=400, bbox_inches='tight')
