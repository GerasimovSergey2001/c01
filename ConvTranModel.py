import numpy as np
from torch import nn
import torch
from AbsolutePositionalEncoding import tAPE, AbsolutePositionalEncoding, LearnablePositionalEncoding
from Attention import Attention, Attention_Rel_Scl, Attention_Rel_Vec
from torch import nn
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import warnings
from IPython.display import clear_output
import wandb
from sklearn.metrics import classification_report, accuracy_score

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Permute(nn.Module):
    def forward(self, x):
        return x.permute(1, 0, 2)


def model_factory(config):
    if config['Net_Type'][0] == 'T':
        model = Transformer(config, num_classes=config['num_labels'])
    elif config['Net_Type'][0] == 'CC-T':
        model = CasualConvTran(config, num_classes=config['num_labels'])
    else:
        model = ConvTran(config, num_classes=config['num_labels'])
    return model


class Transformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2] # number of time series in an obs./ length of a time series
        emb_size = config['emb_size']  # size of attention output
        num_heads = config['num_heads']  # nomber of attention's heads
        dim_ff = config['dim_ff'] # hidden layer size in feed forward
        self.Fix_pos_encode = config['Fix_pos_encode'] # type of fixed (absolute) positional encoder
        self.Rel_pos_encode = config['Rel_pos_encode'] # type of relative positional encoder
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(
            nn.Linear(channel_size, emb_size),
            nn.LayerNorm(emb_size, eps=1e-5)
        )

        if self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        self.LayerNorm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)
        if self.Rel_pos_encode == 'Scalar':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x_src = self.embed_layer(x.permute(0, 2, 1))
        if self.Fix_pos_encode != 'None':
            x_src = self.Fix_Position(x_src)
        att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm1(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)

        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        # out = out.permute(1, 0, 2)
        # out = self.out(out[-1])

        return out


class ConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.embed_layer = nn.Sequential(nn.Conv2d(1, emb_size*4, kernel_size=[1, 8], padding='same'),
                                         nn.BatchNorm2d(emb_size*4),
                                         nn.GELU())

        self.embed_layer2 = nn.Sequential(nn.Conv2d(emb_size*4, emb_size, kernel_size=[channel_size, 1], padding='valid'),
                                          nn.BatchNorm2d(emb_size),
                                          nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = AbsolutePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out


class CasualConvTran(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Parameters Initialization -----------------------------------------------
        channel_size, seq_len = config['Data_shape'][1], config['Data_shape'][2]
        emb_size = config['emb_size']
        num_heads = config['num_heads']
        dim_ff = config['dim_ff']
        self.Fix_pos_encode = config['Fix_pos_encode']
        self.Rel_pos_encode = config['Rel_pos_encode']
        # Embedding Layer -----------------------------------------------------------
        self.causal_Conv1 = nn.Sequential(CausalConv1d(channel_size, emb_size, kernel_size=8, stride=2, dilation=1),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv2 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=5, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        self.causal_Conv3 = nn.Sequential(CausalConv1d(emb_size, emb_size, kernel_size=3, stride=2, dilation=2),
                                          nn.BatchNorm1d(emb_size), nn.GELU())

        if self.Fix_pos_encode == 'tAPE':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif self.Fix_pos_encode == 'Sin':
            self.Fix_Position = tAPE(emb_size, dropout=config['dropout'], max_len=seq_len)
        elif config['Fix_pos_encode'] == 'Learn':
            self.Fix_Position = LearnablePositionalEncoding(emb_size, dropout=config['dropout'], max_len=seq_len)

        if self.Rel_pos_encode == 'eRPE':
            self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, config['dropout'])
        elif self.Rel_pos_encode == 'Vector':
            self.attention_layer = Attention_Rel_Vec(emb_size, num_heads, seq_len, config['dropout'])
        else:
            self.attention_layer = Attention(emb_size, num_heads, config['dropout'])

        self.LayerNorm = nn.LayerNorm(emb_size, eps=1e-5)
        self.LayerNorm2 = nn.LayerNorm(emb_size, eps=1e-5)

        self.FeedForward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(config['dropout']))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x_src = self.embed_layer(x)
        x_src = self.embed_layer2(x_src).squeeze(2)
        x_src = x_src.permute(0, 2, 1)
        if self.Fix_pos_encode != 'None':
            x_src_pos = self.Fix_Position(x_src)
            att = x_src + self.attention_layer(x_src_pos)
        else:
            att = x_src + self.attention_layer(x_src)
        att = self.LayerNorm(att)
        out = att + self.FeedForward(att)
        out = self.LayerNorm2(out)
        out = out.permute(0, 2, 1)
        out = self.gap(out)
        out = self.flatten(out)
        out = self.out(out)
        return out

################################################### Wrapper for training and testing with ConvTran ##################################################

class ConvTranWrapper:
    def __init__(self, project, name,
                 data_shape,                     # (batch_num, num_of_ts, ts_length)
                 emb_size,                       # size of embeddings in Attention Layer
                 dim_ff,                         # dimension of hidden layer in fully connected layer
                 fix_pos_encode,                 # type of absolute positional encoding (tAPE, Sin or Learn)
                 rel_pos_encode,                 # type of relative positional encoding (eRPE, Vector, None)
                 num_heads,                      # number of heads in multiheaded attention
                 dropout = 0.0,
                 num_classes = 2,
                 device = 'cuda'
                ):
        self.project, self.name = project, name
        self.config = {
            'Data_shape' : data_shape, 
            'emb_size' : emb_size, 
            'dim_ff' : dim_ff, 
            'Fix_pos_encode' : fix_pos_encode,
            'Rel_pos_encode' : rel_pos_encode,
            'num_classes' : num_classes,
            'num_heads': num_heads,
            'dropout' : dropout, 
            'device' : device,
        }
        self.device = device
        self.convtran = ConvTran(self.config, num_classes).to(self.device)
            
    def fit(self, train_loader, val_loader, epochs_num = 100, lr=0.005, optimizer = torch.optim.Adam, optimizer_args = {}, 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler_args = {}, compiled = False):
            
        self.config['epochs_num'] = epochs_num
        self.config['lr'] = lr
        self.config['optimizer'] = optimizer.__name__
        self.config['scheduler'] = scheduler.__name__
        self.config['optimizer_args'] = optimizer_args
        self.config['scheduler_args'] = scheduler_args
        self.config['batch_size'] = train_loader.batch_size
        
        with wandb.init(project=self.project, name=self.name, config=self.config):
            config = wandb.config
            
            if compiled:
                self.convtran = torch.compile(self.convtran)
            
            optim = optimizer(self.convtran.parameters(), lr = config.lr, **config.optimizer_args)
            scheduler = scheduler(optim, **config.scheduler_args)
            criterion = F.cross_entropy
            batch_count = 0
            
            wandb.watch(self.convtran, log="gradients", log_freq=10)
    
            for epoch in range(1, config.epochs_num+1):
                total_loss = 0
    
                # clear_output()
                
                self.convtran.train()
                for x, y in tqdm(train_loader, desc = f'Train (Epoch {epoch})'):
            
                    loss = self.train(x, y, criterion)

                    optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.convtran.parameters(), max_norm=4.0)
                    optim.step()
                    
                    batch_count+=1
                    total_loss += loss.detach().cpu()
            
                wandb.log({"epoch": epoch, "Total Cross Entropy Loss (Train)": total_loss, "step" : batch_count})     
                self.convtran.eval()
                total_val_loss = 0
                val_accuracy = []
                for x, y in tqdm(val_loader, desc = f'Val (Epoch {epoch})'):
                    y_val_pred, y_val, loss = self.evaluate(x, y, criterion)
                    total_val_loss += loss.cpu()
                    val_accuracy.append(accuracy_score( y_val.cpu().numpy(), F.softmax(y_val_pred).argmax(axis=1).cpu().numpy() ) )
                try:
                    scheduler.step(total_val_loss)
                except:
                    scheduler.step()
                wandb.log({"Total Cross Entropy Loss (Validation)": total_loss, "Mean Validation Accuracy (per Epoch)" : np.mean(val_accuracy)})
        return self
        
    def train(self, x, y, criterion):
         x, y = x.to(self.device), y.to(self.device)
         y_pred = self.convtran(x)
         return criterion(y_pred, y)
         
    def evaluate(self, x, y, criterion):
        with torch.no_grad():
            y = y.to(self.device)
            y_pred = self.convtran(x.to(self.device))
            return y_pred, y, criterion(y_pred, y)
            
    def __call__(self, x):
        x =  x.to(self.device)
        return self.convtran(x)
    
    def test(self, test_loader):
        y_test_full = []
        y_pred_full = []
        self.convtran.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                y_pred_full.append(F.softmax(self.convtran(x)).argmax(axis=1))
                y_test_full.append(y)
        y_pred_full = torch.cat(y_pred_full).cpu().numpy()
        y_test_full = torch.cat(y_test_full).cpu().numpy()
        report = classification_report(y_test_full, y_pred_full)
        print(report)
        return accuracy_score(y_test_full, y_pred_full), report