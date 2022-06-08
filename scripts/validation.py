import torch
import pandas as pd
from dgnn.build_graph import get_batch
from dgnn.model.memory_updater import MailBox
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input, mfgs_to_cuda, node_to_dgl_blocks
from sklearn.metrics import average_precision_score, roc_auc_score

def val(df: pd.DataFrame, sampler: TemporalSampler, mailbox: MailBox, model: torch.nn.Module
        , node_feats: torch.Tensor, edge_feats: torch.Tensor, creterion: torch.nn.Module, 
            mode='val', neg_samples=1, no_neg=False, identity=False, deliver_to_neighbor=False, prop_time=False):
    model.eval()
    val_losses = list()
    aps = list()
    aucs_mrrs = list()
        
    with torch.no_grad():
        total_loss = 0
        
        mfgs_reverse = False
        if deliver_to_neighbor:
            mfgs_reverse = True

        mfgs = None
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df=df, mode=mode)):
            
            if sampler is not None:
                if no_neg:
                    pos_root_end = target_nodes.shape[0] * 2 // 3
                    mfgs = sampler.sample(target_nodes[:pos_root_end], ts[:pos_root_end], prop_time=prop_time, reverse=mfgs_reverse)
                    # mfgs = sampler.sample(target_nodes[584:585], ts[584:585])
                    # mfgs = sampler.sample(torch.Tensor([242]), torch.Tensor([48066.859]))
                else:
                    mfgs = sampler.sample(target_nodes, ts, prop_time=prop_time, reverse=mfgs_reverse)
            # if identity
            mfgs_deliver_to_neighbors = None
            if identity:
                mfgs_deliver_to_neighbors = mfgs
                mfgs = node_to_dgl_blocks(target_nodes, ts)

            mfgs_to_cuda(mfgs)
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
            if mailbox is not None:
                mailbox.prep_input_mails(mfgs[0])

            pred_pos, pred_neg = model(mfgs, neg_samples)

            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)
            if neg_samples > 1:
                aucs_mrrs.append(torch.reciprocal(torch.sum(pred_pos.squeeze() < pred_neg.squeeze().reshape(neg_samples, -1), dim=0) + 1).type(torch.float))
            else:
                aucs_mrrs.append(roc_auc_score(y_true, y_pred))

            aps.append(average_precision_score(y_true, y_pred))
            
            if mailbox is not None:
                mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
                block = None
                if deliver_to_neighbor:
                    block = mfgs_deliver_to_neighbors[0][0]
                mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
                mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)

        if mode == 'val':
            val_losses.append(float(total_loss))
            
    ap = float(torch.tensor(aps).mean())
    if neg_samples > 1:
        auc_mrr = float(torch.cat(aucs_mrrs).mean())
    else:
        auc_mrr = float(torch.tensor(aucs_mrrs).mean())
    return ap, auc_mrr