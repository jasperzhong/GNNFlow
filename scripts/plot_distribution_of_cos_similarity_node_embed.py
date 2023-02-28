import os
import matplotlib.pyplot as plt
from itertools import tee
import numpy as np
import torch

model = 'TGN'
dataset = 'REDDIT'
layer = 1


# def pairwise(iterable):
#     a = iter(iterable)
#     return zip(a, a)
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_node_embeds():
    files = os.listdir()
    files = [f for f in files if f.startswith(
        'node_embeddings_{}_{}_layer{}'.format(model, dataset, layer))]
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    cos_sim_list = []
    for x, y in pairwise(files[:20]):
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(y)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim)

        cos_sim_list.append(cos_sim)

    return cos_sim_list


def load_node_memory():
    files = os.listdir()
    files = [f for f in files if f.startswith(
        'node_memory_{}_{}_'.format(model, dataset))]
    files = sorted(files, key=lambda x: (
        int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))

    def extract_epoch_iter(x): return (int(x.split('_')[-2]),
                                       int(x.split('_')[-1].split('.')[0]))
    # epoch_iter_list = [(lambda x: (int(x.split('_')[-2]),
    #                     int(x.split('_')[-1].split('.')[0])))(x) for x in files]
    cos_sim_list = []
    ep_it_list = []
    ep_it_list2 = []
    print(files[-40:])
    for x, y in pairwise(files[-41:]):
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(y)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim)

        cos_sim_list.append(cos_sim)
        ep_it_list.append(extract_epoch_iter(x))
        ep_it_list2.append(extract_epoch_iter(y))

    print(ep_it_list)
    print(ep_it_list2)
    return cos_sim_list, ep_it_list


if __name__ == '__main__':
    # cos_sim_list = load_node_embeds()
    cos_sim_list, epoch_iter_list = load_node_memory()
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i, (cos_sim, epoch_iter) in enumerate(zip(cos_sim_list, epoch_iter_list)):
        ax.plot(np.arange(len(cos_sim)), cos_sim,
                label='epoch {} iter {}'.format(*epoch_iter))

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(loc='center left')
    ax.set_xlim((0, len(cos_sim)))
    ax.set_ylim((-1, 1))
    ax.grid(True, color='gray', linestyle='--')
    ax.set_title("Cos similarities of node embeddings of {} on {} (layer {})".format(
        model, dataset, layer))
    plt.savefig('cos_sim_{}_{}.png'.format(
        model, dataset), dpi=400, bbox_inches='tight')
