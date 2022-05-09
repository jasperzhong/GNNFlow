import torch
import pandas as pd
from dgnn.build_graph import get_batch
from dgnn.model.memory_updater import MailBox
from dgnn.temporal_sampler import TemporalSampler
from dgnn.utils import prepare_input
from sklearn.metrics import average_precision_score, roc_auc_score

def val(df: pd.DataFrame, sampler: TemporalSampler, mailbox: MailBox, model: torch.nn.Module
        , node_feats: torch.Tensor, edge_feats: torch.Tensor, creterion: torch.nn.Module, mode='val'):
    model.eval()
    val_losses = list()
    aps = list()
    aucs = list()
        
    with torch.no_grad():
        total_loss = 0
        for i, (target_nodes, ts, eid) in enumerate(get_batch(df=df, mode=mode)):
            mfgs = sampler.sample(target_nodes, ts)
            mfgs[0][0] = mfgs[0][0].to('cuda:0')
            mfgs = prepare_input(mfgs, node_feats, edge_feats, combine_first=False)
            mailbox.prep_input_mails(mfgs[0])
            pred_pos, pred_neg = model(mfgs)
            total_loss += creterion(pred_pos, torch.ones_like(pred_pos))
            total_loss += creterion(pred_neg, torch.zeros_like(pred_neg))
            y_pred = torch.cat([pred_pos, pred_neg], dim=0).sigmoid().cpu()
            y_true = torch.cat([torch.ones(pred_pos.size(0)), torch.zeros(pred_neg.size(0))], dim=0)

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))
            
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            mailbox.update_mailbox(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
            mailbox.update_memory(model.memory_updater.last_updated_nid, model.memory_updater.last_updated_memory, model.memory_updater.last_updated_ts)
        if mode == 'val':
            val_losses.append(float(total_loss))
            
    ap = float(torch.tensor(aps).mean())
    auc = float(torch.tensor(aucs).mean())
    return ap, auc