import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.heterograph import DGLBlock


class TimeEncode(torch.nn.Module):
    """
    Time encoding layer
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: dimension of time features
        """
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = torch.nn.Linear(1, dim)
        self.w.weight = nn.parameter.Parameter((torch.from_numpy(
            1 / 10 ** np.linspace(0, 9, dim, dtype=np.float32))).reshape(dim, -1))
        self.w.bias = nn.parameter.Parameter(torch.zeros(dim))

    def forward(self, delta_time: torch.Tensor):
        output = torch.cos(self.w(delta_time.reshape((-1, 1))))
        return output


class EdgePredictor(torch.nn.Module):
    """
    Edge prediction layer
    """

    def __init__(self, dim: int):
        """
        Args:
            dim_in: dimension of embedding
        """
        super(EdgePredictor, self).__init__()
        self.dim_in = dim
        self.src_fc = torch.nn.Linear(dim, dim)
        self.dst_fc = torch.nn.Linear(dim, dim)
        self.out_fc = torch.nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: embeddings of source, destination and negative sampling nodes
        """
        num_edge = h.shape[0] // 3
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src + h_neg_dst)
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)


class TransfomerAttentionLayer(torch.nn.Module):
    """
    Transfomer attention layer
    """

    def __init__(self, dim_node: int, dim_edge: int, dim_time: int,
                 dim_out: int, num_head: int, dropout: float, att_dropout: float):
        """
        Args:
            dim_node: dimension of node features/embeddings
            dim_edge: dimension of edge features
            dim_time: dimension of time features
            dim_out: dimension of output embeddings
            num_head: number of heads
            dropout: dropout rate
            att_dropout: dropout rate for attention
        """
        super(TransfomerAttentionLayer, self).__init__()
        assert dim_node > 0 or dim_edge > 0, \
            "either dim_node or dim_edge should be positive"

        self.use_node_feat = dim_node > 0
        self.use_edge_feat = dim_edge > 0
        self.use_time_enc = dim_time > 0

        self.num_head = num_head
        self.dim_time = dim_time
        self.dim_out = dim_out
        self.dropout = torch.nn.Dropout(dropout)
        self.att_dropout = torch.nn.Dropout(att_dropout)
        self.att_act = torch.nn.LeakyReLU(0.2)

        if self.use_time_enc:
            self.time_enc = TimeEncode(dim_time)

        if self.use_node_feat or self.use_time_enc:
            self.w_q = torch.nn.Linear(dim_node + dim_time, dim_out)
        else:
            self.w_q = torch.nn.Identity()

        self.w_k = torch.nn.Linear(
            dim_node + dim_edge + dim_time, dim_out)
        self.w_v = torch.nn.Linear(
            dim_node + dim_edge + dim_time, dim_out)

        self.w_out = torch.nn.Linear(dim_node + dim_out, dim_out)

        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, b: DGLBlock):
        """
        Args:
            b: sampled message flow graph (mfg), where
                `b.num_dst_nodes()` is the number of target nodes to sample,
                `b.srcdata['h']` is the embedding of all nodes,
                `b.edge['f']` is the edge features of sampled edges, and
                `b.edata['dt']` is the delta time of sampled edges.

        Returns:
            output: output embedding of target nodes (shape: (num_dst_nodes, dim_out))
        """
        num_edges = b.num_edges()
        num_dst_nodes = b.num_dst_nodes()
        device = b.device

        # sample nothing (no neighbors)
        if num_edges == 0:
            return torch.zeros((num_dst_nodes, self.dim_out), device=device)

        target_node_embeddings = b.srcdata['h'][:num_dst_nodes]
        source_node_embeddings = b.srcdata['h'][num_dst_nodes:]
        edge_feats = b.edata['f']
        delta_time = b.edata['dt']
        assert isinstance(edge_feats, torch.Tensor)

        # determine Q, K, and V (whether to use node features/embeddings,
        # edge features and time encoding
        if self.use_time_enc:
            if self.use_node_feat and self.use_edge_feat:
                Q = target_node_embeddings
                K = V = torch.cat((source_node_embeddings, edge_feats), dim=1)
            elif self.use_node_feat:
                Q = target_node_embeddings
                K = V = source_node_embeddings
            else:
                Q = torch.ones((num_edges, self.dim_out), device=device)
                K = V = edge_feats
        else:
            time_feats = self.time_enc(delta_time)
            zero_time_feats = self.time_enc(torch.zeros(
                num_dst_nodes, dtype=torch.float32, device=device))

            if self.use_node_feat and self.use_edge_feat:
                Q = torch.cat((target_node_embeddings, zero_time_feats), dim=1)
                K = V = torch.cat(
                    (source_node_embeddings, edge_feats, time_feats), dim=1)
            elif self.use_node_feat:
                Q = torch.cat((target_node_embeddings, zero_time_feats), dim=1)
                K = V = torch.cat((source_node_embeddings, time_feats), dim=1)
            else:
                Q = zero_time_feats
                K = V = torch.cat((edge_feats, time_feats), dim=1)

        Q = self.w_q(Q)[b.edges()[1]]
        K = self.w_k(K)
        V = self.w_v(V)

        Q = torch.reshape(Q, (Q.shape[0], self.num_head, -1))
        K = torch.reshape(K, (K.shape[0], self.num_head, -1))
        V = torch.reshape(V, (V.shape[0], self.num_head, -1))

        # compute attention scores
        att = dgl.ops.edge_softmax(b, self.att_act(torch.sum(Q*K, dim=2)))
        att = self.att_dropout(att)
        V = torch.reshape(V*att[:, :, None], (V.shape[0], -1))

        b.srcdata['v'] = torch.cat([torch.zeros(
            (num_dst_nodes, V.shape[1]), device=device), V], dim=0)
        b.update_all(fn.copy_src('v', 'm'), fn.sum('m', 'h'))

        if self.use_node_feat:
            rst = torch.cat((b.dstdata['h'], source_node_embeddings), dim=1)
        else:
            rst = b.dstdata['h']

        rst = self.w_out(rst)
        rst = F.relu(self.dropout(rst))
        return self.layer_norm(rst)
