from torch_geometric import nn as gnn
import torch_geometric
from torch_geometric.loader import LinkNeighborLoader
import torch
import torch.nn.functional as F
from torch import nn
import wandb
import numpy as np


class Node2Vec:
    def __init__(self, dataset, project, name, embedding_dim, walk_length, context_size, walks_per_node, p, q, num_negative_samples, device = 'cuda'):
        self.dataset = dataset
        self.project, self.name = project, name
        self.config = {
            'embedding_dim' : embedding_dim,
            'walk_length' : walk_length,
            'context_size' : context_size,
            'walks_per_node' : walks_per_node,
            'p' : p,
            'q' : q,
            'num_negative_samples' : num_negative_samples
        }
        self.device = device
        
    def fit(self, epochs_num = 40, lr=0.01, optimizer = torch.optim.Adam, optimizer_args = {}, 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args = {}, batch_size = 64):
        
        self.config['epochs_num'] = epochs_num
        self.config['lr'] = lr
        self.config['optimizer'] = optimizer.__name__
        self.config['scheduler'] = scheduler.__name__
        self.config['optimizer_args'] = optimizer_args
        self.config['scheduler_args'] = scheduler_args
        self.config['batch_size'] = batch_size
        
        with wandb.init(project=self.project, name=self.name, config=self.config):
            config = wandb.config
            
            self.node2vec = gnn.Node2Vec(edge_index = self.dataset.edge_index,
                                    embedding_dim = config.embedding_dim,
                                    walk_length = config.walk_length,
                                    context_size = config.context_size,
                                    walks_per_node = config.walks_per_node,
                                    p = config.p, q = config.q, num_negative_samples = config.num_negative_samples
                                   ).to(self.device)
            
            loader = self.node2vec.loader(batch_size=config.batch_size, shuffle=True)
            criterion = self.node2vec.loss
            optim = optimizer(self.node2vec.parameters(), lr = config.lr, **config.optimizer_args)
            scheduler = scheduler(optim, **config.scheduler_args)
            batch_count = 0
            
            wandb.watch(self.node2vec, criterion, log="all", log_freq=10)
            
            for epoch in range(1, config.epochs_num+1):
                for pos_rw, neg_rw in loader:
                    pos_rw, neg_rw = pos_rw.to(self.device), neg_rw.to(self.device)
                    
                    loss = criterion(pos_rw, neg_rw)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    batch_count+=1
                    if ((batch_count + 1) % 5) == 0:
                         wandb.log({"epoch": epoch, "loss": loss, "step" : batch_count})
                try:
                    scheduler.step(loss)
                except:
                    scheduler.step()       
        return self  
            
    def __call__(self, nodes):
        nodes = nodes.to(self.device)
        with torch.no_grad():
            return self.node2vec(nodes)

class GraphSage:
    def __init__(self, dataset, project, name, in_channels, hidden_channels, num_layers, out_channels,
                 dropout = 0, 
                 act = "relu",
                 act_first = False,
                act_kwargs = None,
                norm = None,
                norm_kwargs = None,
                jk = None,
                device = 'cuda', #add GraphSageConv kargs
                ):
        self.dataset = dataset
        self.project, self.name = project, name
        self.config = {
            'in_channels' : in_channels, 
            'hidden_channels' : hidden_channels, 
            'num_layers' : num_layers, 
            'out_channels' : out_channels,
            'dropout' : dropout, 
            'act' : act,
            'act_first' : act_first,
            'act_kwargs' : act_kwargs,
            'norm' : norm,
            'norm_kwargs' : norm_kwargs,
            'jk' : jk,
            'device' : device,
            # 'graphsage_conv_args' : graphsage_conv_kargs
        }
        self.device = device

        
    def fit(self, epochs_num = 100, lr=0.005, optimizer = torch.optim.Adam, optimizer_args = {}, 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args = {}, batch_size = 64, num_neighbors = [30] * 2, neg_sampling_ratio = 1.0,
            compiled = False):
            
        self.config['epochs_num'] = epochs_num
        self.config['lr'] = lr
        self.config['optimizer'] = optimizer.__name__
        self.config['scheduler'] = scheduler.__name__
        self.config['optimizer_args'] = optimizer_args
        self.config['scheduler_args'] = scheduler_args
        self.config['batch_size'] = batch_size
        self.config['num_neighbors'] = num_neighbors
        self.config['neg_sampling_ratio'] = neg_sampling_ratio
        
        with wandb.init(project=self.project, name=self.name, config=self.config):
            config = wandb.config
            
            self.graphsage = gnn.GraphSAGE(
                                in_channels = config.in_channels, 
                                hidden_channels = config.hidden_channels, 
                                num_layers = config.num_layers, 
                                out_channels = config.out_channels,
                                dropout = config.dropout, 
                                act = config.act,
                                act_first = config.act_first,
                                act_kwargs = config.act_kwargs,
                                norm = config.norm,
                                norm_kwargs = config.norm_kwargs,
                                jk = config.jk,
                                # **config.graphsage_conv_args
                            ).to(self.device)
            
            if compiled:
                self.graphsage = torch.compile(self.graphsage)
                
            self.graphsage.train()
            
            loader = LinkNeighborLoader(self.dataset, batch_size = config.batch_size, shuffle=True,
                            neg_sampling_ratio=config.neg_sampling_ratio, num_neighbors=config.num_neighbors,
                            num_workers=6, persistent_workers=True)
            
            optim = optimizer(self.graphsage.parameters(), lr = config.lr, **config.optimizer_args)
            scheduler = scheduler(optim, **config.scheduler_args)
            batch_count = 0
            
            wandb.watch(self.graphsage, log="gradients", log_freq=10)
    
            for epoch in range(1, config.epochs_num+1):
                total_loss = total_examples = 0
                for data in loader:
                    data = data.to(self.device)
                    h = self.graphsage(data.x, data.edge_index)

                    h_src = h[data.edge_label_index[0]]
                    h_dst = h[data.edge_label_index[1]]
                    link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
            
                    loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    batch_count+=1
                    total_loss += float(loss) * link_pred.numel()
                    total_examples += link_pred.numel()
            
                loss_final = total_loss / total_examples
                wandb.log({"epoch": epoch, "loss GS": loss_final, "step" : batch_count})
                try:
                    scheduler.step(loss_final)
                except:
                    scheduler.step()       
        return self  
            
    def __call__(self, x, edge_index):
        self.graphsage.eval()
        x, edge_index =  x.to(self.device), edge_index.to(self.device)
        with torch.no_grad():
            return self.graphsage(x, edge_index)

################################# Directed GraphSAGE #####################################################################

# Directed Sage Convolution

class DirSageConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = gnn.SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = gnn.SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )

# Directed GraphSAGE model

class DirSAGE(nn.Module):
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
    ):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        if num_layers == 1:
            self.convs = nn.ModuleList([DirSageConv(in_channels, output_dim, self.alpha)])
        else:
            self.convs = nn.ModuleList([DirSageConv(in_channels, hidden_channels, self.alpha)])
            for _ in range(num_layers - 2):
                self.convs.append(DirSageConv(hidden_channels, hidden_channels, self.alpha))
            self.convs.append(DirSageConv(hidden_channels, hidden_channels, self.alpha))

        if jumping_knowledge is not None:
            input_dim = hidden_channels * num_layers if jumping_knowledge == "cat" else hidden_channels
            self.lin = nn.Linear(input_dim, out_channels)
            self.jump = gnn.JumpingKnowledge(mode=jumping_knowledge, channels=hidden_channels, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]
        if self.jumping_knowledge is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return x #torch.nn.functional.log_softmax(x, dim=1)

class DirGraphSage:
    def __init__(self, dataset, project, name, in_channels, hidden_channels, out_channels,
                 num_layers,
                 dropout = 0, 
                #  act = "relu",
                #  act_first = False,
                # act_kwargs = None,
                norm = None,
                # norm_kwargs = None,
                jumping_knowledge = 'max',
                normalize=False,
                alpha=1 / 2,
                learn_alpha = False,
                device = 'cuda', #add GraphSageConv kargs
                ):
        self.dataset = dataset
        self.project, self.name = project, name
        self.config = {
            'in_channels' : in_channels, 
            'hidden_channels' : hidden_channels, 
            'num_layers' : num_layers, 
            'out_channels' : out_channels,
            'dropout' : dropout, 
            # 'act' : act,
            # 'act_first' : act_first,
            # 'act_kwargs' : act_kwargs,
            'normalize' : normalize,
            'alpha' : alpha,
            'learn_alpha' : learn_alpha,
            # 'norm_kwargs' : norm_kwargs,
            'jumping_knowledge' : jumping_knowledge,
            'device' : device,
            # 'graphsage_conv_args' : graphsage_conv_kargs
        }
        self.device = device

        
    def fit(self, epochs_num = 100, lr=0.005, optimizer = torch.optim.Adam, optimizer_args = {}, 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args = {}, batch_size = 64, num_neighbors = [30] * 2, neg_sampling_ratio = 1.0,
            compiled = False):
            
        self.config['epochs_num'] = epochs_num
        self.config['lr'] = lr
        self.config['optimizer'] = optimizer.__name__
        self.config['scheduler'] = scheduler.__name__
        self.config['optimizer_args'] = optimizer_args
        self.config['scheduler_args'] = scheduler_args
        self.config['batch_size'] = batch_size
        self.config['num_neighbors'] = num_neighbors
        self.config['neg_sampling_ratio'] = neg_sampling_ratio
        
        with wandb.init(project=self.project, name=self.name, config=self.config):
            config = wandb.config
            
            self.dirgraphsage = DirSAGE(
                                in_channels = config.in_channels, 
                                hidden_channels = config.hidden_channels, 
                                num_layers = config.num_layers, 
                                out_channels = config.out_channels,
                                dropout = config.dropout, 
                                # act = config.act,
                                # act_first = config.act_first,
                                # act_kwargs = config.act_kwargs,
                                normalize = config.normalize,
                                alpha = config.alpha,
                                learn_alpha = config.learn_alpha,
                                # norm_kwargs = config.norm_kwargs,
                                jumping_knowledge = config.jumping_knowledge,
                                # **config.graphsage_conv_args
                            ).to(self.device)
            
            if compiled:
                self.dirgraphsage = torch.compile(self.dirgraphsage)
                
            self.dirgraphsage.train()
            
            loader = LinkNeighborLoader(self.dataset, batch_size = config.batch_size, shuffle=True,
                            neg_sampling_ratio=config.neg_sampling_ratio, num_neighbors=config.num_neighbors,
                            num_workers=6, persistent_workers=True)
            
            optim = optimizer(self.dirgraphsage.parameters(), lr = config.lr, **config.optimizer_args)
            scheduler = scheduler(optim, **config.scheduler_args)
            batch_count = 0
            
            wandb.watch(self.dirgraphsage, log="gradients", log_freq=10)
    
            for epoch in range(1, config.epochs_num+1):
                total_loss = total_examples = 0
                for data in loader:
                    data = data.to(self.device)
                    h = self.dirgraphsage(data.x, data.edge_index)

                    h_src = h[data.edge_label_index[0]]
                    h_dst = h[data.edge_label_index[1]]
                    link_pred = (h_src * h_dst).sum(dim=-1)  # Inner product.
            
                    loss = F.binary_cross_entropy_with_logits(link_pred, data.edge_label)
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    batch_count+=1
                    total_loss += float(loss) * link_pred.numel()
                    total_examples += link_pred.numel()
            
                loss_final = total_loss / total_examples
                wandb.log({"epoch": epoch, "loss GS": loss_final, "step" : batch_count})
                try:
                    scheduler.step(loss_final)
                except:
                    scheduler.step()       
        return self  
            
    def __call__(self, x, edge_index):
        self.dirgraphsage.eval()
        x, edge_index =  x.to(self.device), edge_index.to(self.device)
        with torch.no_grad():
            return self.dirgraphsage(x, edge_index)