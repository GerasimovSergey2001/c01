from torch_geometric import nn as gnn
import torch_geometric
from torch_geometric.loader import LinkNeighborLoader
import torch
import torch.nn.functional as F
from torch import nn
import wandb
import numpy as np
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import Module
from pipeline import DirSAGE
from AutoEncoders import GAE, VGAE
from DirectedWeightedGCN import DirWGCN

class AutoEncoderWrapper:
    def __init__(self, project, name, config, encoder, decoder = None, model = 'VGAE', loss_type = 'MSE', device = 'cuda'):
        self.project, self.name = project, name
        self.config = config
        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.model = model
        self.loss_type = loss_type
        if model == 'VGAE':
            assert self.encoder.head_num == 2
            self.autoencoder = VGAE(self.encoder, self.decoder, loss_type).to(self.device)
        elif model == 'GAE':
            assert self.encoder.head_num == 1
            self.autoencoder = GAE(self.encoder, self.decoder, loss_type).to(self.device)
        else:
            raise NotImplementedError("Such AutoEncoder is not Implemented")
    
    def fit(self, full_data, epochs_num = 100, lr=0.005, optimizer = torch.optim.Adam, optimizer_args = {}, 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args = {}, compiled = False, lambda_ = None):
        
        if lambda_ is None:
             lambda_ = 1/full_data.num_nodes
            
        self.config['epochs_num'] = epochs_num
        self.config['model'] = self.model
        self.config['lr'] = lr
        self.config['optimizer'] = optimizer.__name__
        self.config['scheduler'] = scheduler.__name__
        self.config['optimizer_args'] = optimizer_args
        self.config['scheduler_args'] = scheduler_args
        self.config['lambda'] = lambda_

        
        with wandb.init(project=self.project, name=self.name, config=self.config):
            config = wandb.config
            kl_loss = None
            if compiled:
                self.autoencoder = torch.compile(self.autoencoder)
                
            self.autoencoder.train()
            
            optim = optimizer(self.autoencoder.parameters(), lr = config.lr, **config.optimizer_args)
            scheduler = scheduler(optim, **config.scheduler_args)
            batch_count = 0
            
            wandb.watch(self.autoencoder, log="gradients", log_freq=10)
            data = full_data.to(self.device)
            for epoch in range(1, config.epochs_num+1):
                total_loss = 0
                
                z = self.autoencoder.encode(data.x, data.edge_index, data.edge_weight)

                loss = self.autoencoder.recon_loss(z, data.edge_index, data.edge_weight) #such configuration supposes that loader returns positive edge_indexes
                
                if self.model == 'VGAE':
                    kl_loss = self.autoencoder.kl_loss()
                    loss = loss + lambda_ * kl_loss
                
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                batch_count+=1
                
                wandb.log({"epoch": epoch, f"Train Loss ({self.loss_type})": loss, "KL Loss": kl_loss, "step" : batch_count})
                try:
                    scheduler.step(loss)
                except:
                    scheduler.step()       
        return self  
            
    def __call__(self, x, edge_index, edge_weight):
        self.autoencoder.eval()
        x, edge_index, edge_weight =  x.to(self.device), edge_index.to(self.device), edge_weight.to(self.device)
        with torch.no_grad():
            return self.autoencoder.encode(x, edge_index, edge_weight)

class DirGraphSageWrapper:
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