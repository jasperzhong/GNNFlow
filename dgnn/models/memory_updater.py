import torch

from .layers import TimeEncode


class MailBox():

    def __init__(self, memory_dim_out, mailbox_size, mail_combine, num_nodes, dim_edge_feat,
                 deliver_to_neighbors=False,
                 _node_memory=None, _node_memory_ts=None, _mailbox=None,
                 _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):

        self.memory_dim_out = memory_dim_out
        self.mail_combine = mail_combine
        self.mailbox_size = mailbox_size
        self.dim_edge_feat = dim_edge_feat
        self.deliver_to_neighbors = deliver_to_neighbors

        self.node_memory = torch.zeros(
            (num_nodes, memory_dim_out),
            dtype=torch.float32) if _node_memory is None else _node_memory
        self.node_memory_ts = torch.zeros(
            num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
        self.mailbox = torch.zeros(
            (num_nodes, mailbox_size,
             2 * memory_dim_out + dim_edge_feat),
            dtype=torch.float32) if _mailbox is None else _mailbox
        self.mailbox_ts = torch.zeros(
            (num_nodes, mailbox_size),
            dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
        self.next_mail_pos = torch.zeros(
            (num_nodes),
            dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')

    def reset(self):
        self.node_memory.fill_(0)
        self.node_memory_ts.fill_(0)
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')

    def allocate_pinned_memory_buffers(
            self, sample_neighbor, sample_history, batch_size):
        limit = int(batch_size * 3.3)
        if sample_neighbor is not None:
            for i in sample_neighbor:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_history):
            self.pinned_node_memory_buffs.append(
                torch.zeros(
                    (limit, self.node_memory.shape[1]),
                    pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(
                torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(
                torch.zeros(
                    (limit, self.mailbox.shape[1],
                     self.mailbox.shape[2]),
                    pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(
                torch.zeros(
                    (limit, self.mailbox_ts.shape[1]),
                    pin_memory=True))

    def prep_input_mails(self, mfg, use_pinned_buffers=False):
        for i, b in enumerate(mfg):
            if use_pinned_buffers:
                idx = b.srcdata['ID'].cpu().long()
                dst_idx = idx[:b.num_dst_nodes()]
                torch.index_select(
                    self.node_memory, 0, idx,
                    out=self.pinned_node_memory_buffs[i][: idx.shape[0]])
                b.srcdata['mem'] = self.pinned_node_memory_buffs[i][
                    : idx.shape[0]].cuda(
                    non_blocking=True)
                torch.index_select(
                    self.node_memory_ts, 0, dst_idx,
                    out=self.pinned_node_memory_ts_buffs[i]
                    [: dst_idx.shape[0]])
                b.dstdata['mem_ts'] = self.pinned_node_memory_ts_buffs[i][
                    : dst_idx.shape[0]].cuda(
                    non_blocking=True)
                torch.index_select(
                    self.mailbox, 0, dst_idx,
                    out=self.pinned_mailbox_buffs[i][: dst_idx.shape[0]])
                b.dstdata['mem_input'] = self.pinned_mailbox_buffs[i][:dst_idx.shape[0]].reshape(
                    dst_idx.shape[0], -1).cuda(non_blocking=True)
                torch.index_select(
                    self.mailbox_ts, 0, dst_idx,
                    out=self.pinned_mailbox_ts_buffs[i][: dst_idx.shape[0]])
                b.dstdata['mail_ts'] = self.pinned_mailbox_ts_buffs[i][
                    : dst_idx.shape[0]].cuda(
                    non_blocking=True)
            else:
                dst_id = b.srcdata['ID'][:b.num_dst_nodes()].long()
                b.srcdata['mem'] = self.node_memory[b.srcdata['ID'].long()
                                                    ].cuda()
                b.dstdata['mem_ts'] = self.node_memory_ts[dst_id].cuda()
                b.dstdata['mem_input'] = self.mailbox[dst_id].cuda().reshape(
                    dst_id.shape[0], -1)
                b.dstdata['mail_ts'] = self.mailbox_ts[dst_id].cuda()

    def update_memory(self, nid, memory, ts, neg_samples=1):
        if nid is None:
            return
        num_true_src_dst = nid.shape[0] // (neg_samples + 2) * 2
        # num_true_src_dst = nid.shape[0]
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(self.device)
            memory = memory[:num_true_src_dst].to(self.device)
            ts = ts[:num_true_src_dst].to(self.device)
            self.node_memory[nid.long()] = memory
            self.node_memory_ts[nid.long()] = ts

    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, 
                       neg_samples=1):
        with torch.no_grad():
            num_true_edges = nid.shape[0] // (neg_samples + 2)
            memory = memory.to(self.device)
            if edge_feats is not None:
                edge_feats = edge_feats.to(self.device)

            src = torch.from_numpy(
                root_nodes[: num_true_edges]).to(
                self.device)
            dst = torch.from_numpy(
                root_nodes[num_true_edges:num_true_edges * 2]).to(self.device)
            mem_src = memory[:num_true_edges]
            mem_dst = memory[num_true_edges:num_true_edges * 2]
            if self.dim_edge_feat > 0:
                src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
            else:
                src_mail = torch.cat([mem_src, mem_dst], dim=1)
                dst_mail = torch.cat([mem_dst, mem_src], dim=1)
            mail = torch.cat([src_mail, dst_mail],
                             dim=1).reshape(-1, src_mail.shape[1])
            nid = torch.cat([src.unsqueeze(1),
                             dst.unsqueeze(1)],
                            dim=1).reshape(-1)
            mail_ts = torch.from_numpy(
                ts[:num_true_edges * 2]).to(self.device)

            # find unique nid to update mailbox
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(
                inv.size(0),
                dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = nid[perm]
            mail = mail[perm]
            mail_ts = mail_ts[perm]
            if self.mail_combine == 'last':
                self.mailbox[nid.long(),
                             self.next_mail_pos[nid.long()]] = mail
                self.mailbox_ts[nid.long(),
                                self.next_mail_pos[nid.long()]] = mail_ts
                if self.mailbox_size > 1:
                    self.next_mail_pos[nid.long()] = torch.remainder(
                        self.next_mail_pos[nid.long()] + 1, self.mailbox_size)

    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(
                self.next_mail_pos[nid] + 1, self.mailbox_size)
            self.update_mail_pos.fill_(0)


class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, combine_node_feature, dim_in,
                 dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.combine_node_feature = combine_node_feature
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combine_node_feature:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(
                    b.srcdata['ts'][: b.num_dst_nodes()] - b.dstdata
                    ['mem_ts'])
                b.dstdata['mem_input'] = torch.cat(
                    [b.dstdata['mem_input'], time_feat], dim=1)
            # updater take two inputs: input_size = msg_dim & hidden_size = mem_dim
            # msg_dim comes from msg_function it maybe an mlp or identity
            # if is identity: msg_dim = raw_msg_dim = 2 * mem_dim + n_edge_dim + time_encoder_dim
            # updater(dim_in + dim_time -> msg_dim, dim_hid == mem_dim == dim_out)
            # b.dstdata['mem_input'] ==> message ;
            # b.srcdata['mem][:b.num_dst_nodes()] ==> target nodes' memory
            updated_memory = self.updater(
                b.dstdata['mem_input'],
                b.srcdata['mem'][: b.num_dst_nodes()])
            self.last_updated_ts = b.srcdata['ts'][
                : b.num_dst_nodes()].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'][
                : b.num_dst_nodes()].detach().clone()
            new_memory = torch.cat(
                [updated_memory, b.srcdata['mem'][b.num_dst_nodes():]], dim=0)
            if self.combine_node_feature:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += new_memory
                    else:
                        b.srcdata['h'] = new_memory + \
                            self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = new_memory


class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, combine_node_feature, dim_in,
                 dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.combine_node_feature = combine_node_feature
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if combine_node_feature:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(
                    b.srcdata['ts'][: b.num_dst_nodes()] - b.dstdata
                    ['mem_ts'])
                b.dstdata['mem_input'] = torch.cat(
                    [b.dstdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(
                b.dstdata['mem_input'],
                b.srcdata['mem'][: b.num_dst_nodes()])
            self.last_updated_ts = b.srcdata['ts'][
                : b.num_dst_nodes()].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'][
                : b.num_dst_nodes()].detach().clone()
            new_memory = torch.cat(
                [updated_memory, b.srcdata['mem'][b.num_dst_nodes():]], dim=0)
            if self.combine_node_feature:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += new_memory
                    else:
                        b.srcdata['h'] = new_memory + \
                            self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = new_memory


class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, mailbox_size, attention_head, dim_in,
                 dim_out, dim_time, dropout, attn_dropout):
        super(TransformerMemoryUpdater, self).__init__()

        self.dim_time = dim_time
        self.mailbox_size = mailbox_size
        self.att_h = attention_head
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(attn_dropout)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape(
                (b.num_src_nodes(), self.att_h, -1))
            mails = b.dstdata['mem_input'].reshape(
                (b.num_src_nodes(), self.mailbox_size, -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.dstdata['mail_ts']).reshape(
                    (b.num_src_nodes(), self.mailbox_size, -1))
                mails = torch.cat([mails, time_feat], dim=2)
            K = self.w_k(mails).reshape(
                (b.num_src_nodes(),
                 self.mailbox_size,
                 self.att_h, -1))
            V = self.w_v(mails).reshape(
                (b.num_src_nodes(),
                 self.mailbox_size,
                 self.att_h, -1))
            att = self.att_act((Q[:, None, :, :] * K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:, :, :, None] * V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
