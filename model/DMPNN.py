# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : MPNN.py
@Project  : ChiralityGNN_HLM
@Time     : 2023/8/19 10:53
@Author   : Pu Chengtao
@Contact_2: 2319189860@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2023/8/19 10:53        1.0             None
"""
import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_scatter import scatter_sum

def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3, graph_pool='mean'):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth
        self.graph_pool = graph_pool

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float() #  节点v的节点特征和（from_v)的边特征进行拼接，shape : [edge_index[0],node_feat+edge_feat
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        return self.pool(node_attr, batch)


class DMPNN(nn.Module):
    def __init__(self,DMPNN_hidden_size,num_node_features,num_edge_features,depth,mlp_hidden_size,out_dim=1,drop_rate=0,graph_pool='mean'):
        super().__init__()
        self.encoder = DMPNNEncoder(DMPNN_hidden_size,num_node_features,num_edge_features,depth,graph_pool=graph_pool)
        self.mlp = nn.Sequential(
            nn.Dropout(p=drop_rate,inplace=False),
            nn.Linear(DMPNN_hidden_size,mlp_hidden_size,bias=True),
            nn.ReLU(),
            nn.Dropout(p=drop_rate,inplace=False),
            nn.Linear(mlp_hidden_size,out_dim,bias=True)
        )

    def forward(self,data):
        gnn_out = self.encoder(data)
        output = self.mlp(gnn_out)

        return output