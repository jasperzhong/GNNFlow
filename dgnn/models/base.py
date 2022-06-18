import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mailbox = None
        self.memory_updater = None

    def train(self, mode = True):
        # TODO: where to put this if?
        # learn AOP!!
        if self.mailbox is not None:
            self.mailbox.reset()
            self.memory_updater.last_updated_nid = None
        super().train()

    def forward(self, mfgs, neg_samples=1):
        if self.mailbox is not None:
            self.mailbox.prep_input_mails(mfgs[0])
            self.memory_updater(mfgs[0])

    def update(self, target_nodes, ts, edge_feats, eid, mfgs_deliver_to_neighbors = None, deliver_to_neighbors = False):
        if self.mailbox is not None:
            mem_edge_feats = edge_feats[eid] if edge_feats is not None else None
            block = None
            if deliver_to_neighbors:
                block = mfgs_deliver_to_neighbors[0][0]
            self.mailbox.update_mailbox(self.memory_updater.last_updated_nid, self.memory_updater.last_updated_memory, target_nodes, ts, mem_edge_feats, block)
            self.mailbox.update_memory(self.memory_updater.last_updated_nid, self.memory_updater.last_updated_memory, self.memory_updater.last_updated_ts)