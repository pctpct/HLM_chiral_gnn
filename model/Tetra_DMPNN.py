from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool,global_mean_pool, global_max_pool, GlobalAttention, Set2Set


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class TetraDMPNN(nn.Module):
    def __init__(self, gnn_hidden_size, num_node_features, num_edge_features, depth, drop_rate, graph_pool, tetra_message):
        super(TetraDMPNN, self).__init__()

        self.depth = depth
        self.hidden_size = gnn_hidden_size
        self.dropout = drop_rate
        self.graph_pool = graph_pool
        self.tetra_message = tetra_message

        self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size) # initialize edge hidee states
        self.edge_to_node = DMPNNConv(self.hidden_size, self.tetra_message)

        # layers
        self.convs = torch.nn.ModuleList()

        for _ in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_size, self.tetra_message))

        # graph pooling
        self.tetra_update = get_tetra_update(self.tetra_message, self.hidden_size)

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                            torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        self.mult = 2 if self.graph_pool == "set2set" else 1
        self.ffn = nn.Linear(self.mult * self.hidden_size, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch, parity_atoms = data.x, data.edge_index, data.edge_attr, data.batch, data.parity_atoms

        row, col = edge_index
        edge_attr = torch.cat([x[row], edge_attr], dim=1)
        edge_attr = F.relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for l in range(self.depth):

            x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms)
            h = edge_attr_h

            if l == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            h = h + edge_attr_h
            edge_attr_list.append(h)


        # dmpnn edge -> node aggregation
        h, _ = self.edge_to_node(x_list[-1], edge_index, h, parity_atoms)

        return self.ffn(self.pool(h, batch)).squeeze(-1)


class DMPNNConv(MessagePassing):
    def __init__(self, hidden_size, tetra_message):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU())

        self.tetra_update = get_tetra_update(tetra_message, hidden_size)

    def forward(self, x, edge_index, edge_attr, parity_atoms):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)

        tetra_ids = parity_atoms.nonzero().squeeze(1)
        # if tetra_ids.nelement() != 0:
        #     a_message[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)
        if tetra_ids.nelement() > 1: # avoid batch norm with one element
            a_message[tetra_ids] = self.tetra_message(x, edge_index, edge_attr, tetra_ids, parity_atoms)

        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)
        return a_message, self.mlp(a_message[row] - rev_message)

    def message(self, x_j, edge_attr):
        return F.relu(self.lin(edge_attr))

    def tetra_message(self, x, edge_index, edge_attr, tetra_ids, parity_atoms):

        row, col = edge_index
        tetra_nei_ids = torch.cat([row[col == i].unsqueeze(0) for i in range(x.size(0)) if i in tetra_ids]) # shape (num_tetra_ids,4)

        # switch entries for -1 rdkit labels
        ccw_mask = parity_atoms[tetra_ids] == -1
        tetra_nei_ids[ccw_mask] = tetra_nei_ids.clone()[ccw_mask][:, [1, 0, 2, 3]]

        # calculate reps
        edge_ids = torch.cat([tetra_nei_ids.view(1, -1), tetra_ids.repeat_interleave(4).unsqueeze(0)], dim=0) # row:chiral_atom neis id ; col: chiral_atom_ids
        # dense_edge_attr = to_dense_adj(edge_index, batch=None, edge_attr=edge_attr).squeeze(0)
        # edge_reps = dense_edge_attr[edge_ids[0], edge_ids[1], :].view(tetra_nei_ids.size(0), 4, -1)
        attr_ids = [torch.where((a == edge_index.t()).all(dim=1))[0] for a in edge_ids.t()] # index of chiral atom in edge_index
        edge_reps = edge_attr[attr_ids, :].view(tetra_nei_ids.size(0), 4, -1)

        return self.tetra_update(edge_reps)


class TetraPermuter(nn.Module):

    def __init__(self, hidden):
        super(TetraPermuter, self).__init__()

        self.W_bs = nn.ModuleList([copy.deepcopy(nn.Linear(hidden, hidden)) for _ in range(4)])
        self.drop = nn.Dropout(p=0.2)
        self.reset_parameters()
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        self.tetra_perms = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

    def reset_parameters(self):
        gain = 0.5
        for W_b in self.W_bs:
            nn.init.xavier_uniform_(W_b.weight, gain=gain)
            gain += 0.5

    def forward(self, x):

        nei_messages_list = [self.drop(F.tanh(l(t))) for l, t in zip(self.W_bs, torch.split(x[:, self.tetra_perms, :], 1, dim=-2))]
        nei_messages = torch.sum(self.drop(F.relu(torch.cat(nei_messages_list, dim=-2).sum(dim=-2))), dim=-2)

        return self.mlp_out(nei_messages / 3.)


class ConcatTetraPermuter(nn.Module):

    def __init__(self, hidden):
        super(ConcatTetraPermuter, self).__init__()

        self.W_bs = nn.Linear(hidden*4, hidden)
        torch.nn.init.xavier_normal_(self.W_bs.weight, gain=1.0)
        self.hidden = hidden
        self.drop = nn.Dropout(p=0.2)
        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

        self.tetra_perms = torch.tensor([[0, 1, 2, 3],
                                         [0, 2, 3, 1],
                                         [0, 3, 1, 2],
                                         [1, 0, 3, 2],
                                         [1, 2, 0, 3],
                                         [1, 3, 2, 0],
                                         [2, 0, 1, 3],
                                         [2, 1, 3, 0],
                                         [2, 3, 0, 1],
                                         [3, 0, 2, 1],
                                         [3, 1, 0, 2],
                                         [3, 2, 1, 0]])

    def forward(self, x):

        nei_messages = self.drop(F.relu(self.W_bs(x[:, self.tetra_perms, :].flatten(start_dim=2))))
        return self.mlp_out(nei_messages.sum(dim=-2) / 3.)


class TetraDifferencesProduct(nn.Module):

    def __init__(self, hidden):
        super(TetraDifferencesProduct, self).__init__()

        self.mlp_out = nn.Sequential(nn.Linear(hidden, hidden),
                                     nn.BatchNorm1d(hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, hidden))

    def forward(self, x):

        indices = torch.arange(4).to(x.device)
        message_tetra_nbs = [x.index_select(dim=1, index=i).squeeze(1) for i in indices]
        message_tetra = torch.ones_like(message_tetra_nbs[0])

        # note: this will zero out reps for chiral centers with multiple carbon neighbors on first pass
        for i in range(4):
            for j in range(i + 1, 4):
                message_tetra = torch.mul(message_tetra, (message_tetra_nbs[i] - message_tetra_nbs[j]))
        message_tetra = torch.sign(message_tetra) * torch.pow(torch.abs(message_tetra) + 1e-6, 1 / 6)
        return self.mlp_out(message_tetra)


def get_tetra_update(tetra_message,hidden_size):

    if tetra_message == 'tetra_permute':
        return TetraPermuter(hidden_size)
    elif tetra_message == 'tetra_permute_concat':
        return ConcatTetraPermuter(hidden_size)
    elif tetra_message == 'tetra_pd':
        return TetraDifferencesProduct(hidden_size)
    else:
        raise ValueError("Invalid message type.")