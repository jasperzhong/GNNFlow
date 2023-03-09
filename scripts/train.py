
import logging
from queue import Queue
from threading import Lock
from typing import List
import torch

from gnnflow.utils import mfgs_to_cuda


def training_batch(model, sampler, cache, target_nodes, ts, eid, device, distributed, optimizer, criterion, Stream: torch.cuda.Stream, queue: Queue, lock_pool: List[Lock], i: int, rank: int):
    with torch.cuda.stream(Stream):
        with lock_pool[0]:
            mfgs = sampler.sample(target_nodes, ts)
        # lock_pool[0].release()
        with lock_pool[1]:
            mfgs_to_cuda(mfgs, device)
            mfgs = cache.fetch_feature(
                mfgs, eid, target_edge_features=True)  # because all use memory
        with lock_pool[2]:
            b = mfgs[0][0]  # type: DGLBlock
            if distributed:
                model.module.memory.prepare_input(b)
            else:
                model.memory.prepare_input(b)
        with lock_pool[3]:
            # if rank == 0:
            #     logging.info("gnn iter: {}".format(i))
            with torch.cuda.stream(torch.cuda.default_stream()):
                if distributed:
                    last_updated = model.module.memory_updater(mfgs[0][0])
                else:
                    last_updated = model.memory_updater(mfgs[0][0])

                optimizer.zero_grad()

                pred_pos, pred_neg = model(mfgs)
                #     logging.info('pred_pos{}'.format(pred_pos))

                loss = criterion(pred_pos, torch.ones_like(pred_pos))

                loss += criterion(pred_neg, torch.zeros_like(pred_neg))
                # logging.info("iter: {} loss: {}".format(i, loss))
            # total_loss += float(loss) * num_target_nodes
                loss.backward()
                optimizer.step()
        # if rank == 0:
        #     logging.info("gnn iter: {}".format(i))
        with lock_pool[4]:
            # if rank == 0:
            #     logging.info("update iter: {}".format(i))
            with torch.no_grad():
                # use one function
                if distributed:
                    model.module.memory.update_mem_mail(
                        **last_updated, edge_feats=cache.target_edge_features.get(),
                        neg_sample_ratio=1)
                else:
                    model.memory.update_mem_mail(
                        **last_updated, edge_feats=cache.target_edge_features.get(),
                        neg_sample_ratio=1)
    # queue.get(1)
