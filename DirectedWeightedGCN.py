from torch_geometric import nn as gnn
import torch_geometric
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch import nn
from utils import get_norm_adj
import wandb
import numpy as np

class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, bias = True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim, bias = bias)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim, bias = bias)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )
        
class DirWGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, bias = True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = nn.Linear(input_dim, output_dim, bias = bias)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim, bias = bias)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index, edge_weight):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, value = edge_weight, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, value = edge_weight, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return self.alpha * self.lin_src_to_dst(self.adj_norm @ x) + (1 - self.alpha) * self.lin_dst_to_src(
            self.adj_t_norm @ x
        )


class DirWGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=3,
        dropout=0,
        jumping_knowledge='max',
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
        bias = True,
        activation = nn.ReLU,
        head_num = 1
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        if num_layers == 1:
            self.convs = nn.ModuleList([DirWGCNConv(in_channels, output_dim, self.alpha, bias)])
        else:
            self.convs = nn.ModuleList([DirWGCNConv(in_channels, hidden_channels, self.alpha, bias)])
            for _ in range(num_layers - 2):
                self.convs.append(DirWGCNConv(hidden_channels, hidden_channels, self.alpha, bias))
            self.convs.append(DirWGCNConv(hidden_channels, hidden_channels, self.alpha, bias))

        if jumping_knowledge is not None:
            input_dim = hidden_channels * num_layers if jumping_knowledge == "cat" else hidden_channels
            self.jump = gnn.JumpingKnowledge(mode=jumping_knowledge, channels=hidden_channels, num_layers=num_layers)
            #expand to an arbiterary FC?
            if head_num == 1:
                self.lin = nn.Linear(input_dim, out_channels) 
            else:
                self.lin = nn.ModuleList([nn.Linear(input_dim,out_channels), nn.Linear(input_dim,out_channels)])

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize
        self.head_num = head_num
        self.activation = activation()

    def forward(self, x, edge_index, edge_weight):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]
        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            if self.head_num == 1:
                x = self.lin(x)
                return x
            else:
                mu, log_sigma = self.lin[0](x), self.lin[1](x)
                return mu, log_sigma