import os
import matplotlib.pyplot as plt
from itertools import tee
import numpy as np
import torch

model = 'TGN'
dataset = 'GDELT'
layer = 1
epoch = 0

# plt.rcParams.update({'font.size': 10, 'font.family': 'Myriad Pro'})


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a, a, a)

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
    for x, y, z, k in pairwise(files[:40]):
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(z)).cuda()

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim)

        cos_sim_list.append(cos_sim)

    return cos_sim_list


def count_zeros(x):
    # find the rows where all elements are zero using boolean indexing
    zero_rows = (x == 0).all(dim=1)

    # count the number of rows where all elements are zero
    num_zero_rows = zero_rows.sum().item()
    return num_zero_rows


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
    num_first_time_update_list = []
    for i, (x, y, z, k) in enumerate(pairwise(files[:])):
        # plot interval
        if i % 10 != 0:
            continue
        x = 'memory/' + x
        z = 'memory/' + z
        node_embed_1 = torch.from_numpy(np.load(x)).cuda()
        node_embed_2 = torch.from_numpy(np.load(z)).cuda()

        num_first_time_update = abs(count_zeros(
            node_embed_1) - count_zeros(node_embed_2))
        num_first_time_update_list.append(num_first_time_update)

        # compute cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            node_embed_1, node_embed_2, dim=1)

        # sort by cosine similarity
        cos_sim = cos_sim.cpu().numpy()
        cos_sim = np.sort(cos_sim, kind='stable')

        cos_sim_list.append(cos_sim)
        ep_it_list.append(extract_epoch_iter(x))
        ep_it_list2.append(extract_epoch_iter(z))

    print(ep_it_list)
    print(ep_it_list2)
    print(num_first_time_update_list)
    return cos_sim_list, ep_it_list, num_first_time_update_list


if __name__ == '__main__':
    # cos_sim_list = load_node_embeds()
    cos_sim_list, epoch_iter_list, num_first_time_update_list = load_node_memory()
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    all_updated_nodes_list = []
    update_percent_list = []
    all_percent_list = []
    for i, (cos_sim, epoch_iter, num_first_time_update) in enumerate(zip(cos_sim_list, epoch_iter_list, num_first_time_update_list)):
        # ax.scatter(np.arange(len(cos_sim)), cos_sim,
        #            label='iter {}'.format(epoch_iter[1]), linewidths=1)
        # eliminate cos = 1 and cos = 0
        all_nodes = len(cos_sim)
        cos_sim = cos_sim[cos_sim != 0]
        cos_sim = cos_sim[cos_sim < 0.99]
        # updated_nodes = len(cos_sim)
        # all_updated_nodes = updated_nodes + num_first_time_update
        # all_updated_nodes_list.append(all_updated_nodes)

        # # >0.75 <0.95
        # cos_sim = cos_sim[cos_sim < 0.99]
        # cos_sim = cos_sim[cos_sim > 0.75]
        # update_percent = len(cos_sim) / all_updated_nodes
        # update_percent_list.append(update_percent)
        # all_percent = len(cos_sim) / all_nodes
        # all_percent_list.append(all_percent)
        ax.plot(np.arange(len(cos_sim)), cos_sim,
                label='iter {}'.format(epoch_iter[1]))
    # ax.plot(np.arange(0, 8100, 400),
    #         all_updated_nodes_list, label='all updated nodes num')
    # ax.plot(np.arange(0, 8100, 400),
    #         num_first_time_update_list, label='first updated nodes num')

    # ax.plot(np.arange(0, 8100, 400),
    #         update_percent_list, label='percent of cos >0.75 <0.95 in updated nodes')
    # ax.plot(np.arange(0, 8100, 400),
    #         all_percent_list, label='percent of cos >0.75 <0.95 in all nodes')

    # print('all noeds: {}'.format(all_nodes))
    # print('all updated nodes: {}'.format(all_updated_nodes_list))
    # print('all updated nodes percent: {}'.format(
    #     np.array(all_updated_nodes_list) / all_nodes))
    # print('>0.75 <0.95 in updated nodes percent: {}'.format(update_percent_list))
    # print('>0.75 <0.95 in all nodes percent: {}'.format(all_percent_list))

    # ax.set_xlabel('iteration')
    # ax.set_ylabel('Num nodes')
    # print('len{}'.format(len(all_updated_nodes_list)))
    # ax.legend(ncol=4)
    # # ax.set_xlim((0, len(all_percent_list)))
    # # print(len(np.arange(0, 8100, 400)))
    # ax.set_xticks(np.arange(0, 8100, 400))
    # # ax.set_ylim((-1, 1))
    # ax.grid(True, color='gray', linestyle='--')
    # # ax.set_title("Cos similarities of node memory of {} on {} (epoch {})".format(
    # #     model, dataset, epoch))
    # # plt.savefig('nid_cos_sim_{}_{}_epoch{}.png'.format(
    # #     model, dataset, epoch), dpi=400, bbox_inches='tight')
    # plt.savefig('percent_{}_{}_epoch{}.png'.format(
    #     model, dataset, epoch), dpi=400, bbox_inches='tight')

    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(ncol=4)
    ax.set_xlim((0, len(cos_sim)))
    ax.set_ylim((-1, 1))
    ax.grid(True, color='gray', linestyle='--')
    ax.set_title("Cos similarities of node memory of {} on {} (epoch {}, diff {} iters)".format(
        model, dataset, epoch, 200))
    # plt.savefig('nid_cos_sim_{}_{}_epoch{}.png'.format(
    #     model, dataset, epoch), dpi=400, bbox_inches='tight')
    plt.savefig('difference_cos_sim_{}_{}_epoch{}_all.png'.format(
        model, dataset, epoch), dpi=400, bbox_inches='tight')
