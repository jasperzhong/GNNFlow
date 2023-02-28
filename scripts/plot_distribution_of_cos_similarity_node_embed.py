import os
import matplotlib.pyplot as plt
from itertools import tee
import numpy as np
import torch

model = 'TGN'
dataset = 'GDELT'
layer = 1
epoch = 1


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)

# def pairwise(iterable):
#     # pairwise('ABCDEFG') --> AB BC CD DE EF FG
#     a, b = tee(iterable)
#     next(b, None)
#     return zip(a, b)


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
    files = os.listdir('memory/')
    files = [f for f in files if f.startswith(
        'node_memory_{}_{}_{}'.format(model, dataset, epoch))]
    files = sorted(files, key=lambda x: (
        int(x.split('_')[-2]), int(x.split('_')[-1].split('.')[0])))

    def extract_epoch_iter(x): return (int(x.split('_')[-2]),
                                       int(x.split('_')[-1].split('.')[0]))
    # epoch_iter_list = [(lambda x: (int(x.split('_')[-2]),
    #                     int(x.split('_')[-1].split('.')[0])))(x) for x in files]
    cos_sim_list = []
    ep_it_list = []
    ep_it_list2 = []
    for i, (x, y) in enumerate(pairwise(files[:])):
        # plot interval
        if i % 2 == 1:
            continue
        x = 'memory/' + x
        y = 'memory/' + y
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(y)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim, kind='stable')

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
        # ax.scatter(np.arange(len(cos_sim)), cos_sim,
        #            label='iter {}'.format(epoch_iter[1]), linewidths=1)
        # eliminate cos = 1 and cos = 0
        cos_sim = cos_sim[cos_sim != 0]
        cos_sim = cos_sim[cos_sim < 0.999]
        # cos_sim = cos_sim - 1
        # cos_sim = cos_sim[cos_sim != 0]
        # cos_sim = cos_sim + 1
        print(cos_sim)
        ax.plot(np.arange(len(cos_sim)), cos_sim,
                label='iter {}'.format(epoch_iter[1]))

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(ncol=4)
    ax.set_xlim((0, len(cos_sim)))
    ax.set_ylim((-1, 1))
    ax.grid(True, color='gray', linestyle='--')
    ax.set_title("Cos similarities of node memory of {} on {} (epoch {})".format(
        model, dataset, epoch))
    # plt.savefig('nid_cos_sim_{}_{}_epoch{}.png'.format(
    #     model, dataset, epoch), dpi=400, bbox_inches='tight')
    plt.savefig('target_cos_sim_{}_{}_epoch{}.png'.format(
        model, dataset, epoch), dpi=400, bbox_inches='tight')
