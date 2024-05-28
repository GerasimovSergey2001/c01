from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch_geometric.nn.models.autoencoder import InnerProductDecoder

EPS = 1e-15
MAX_LOGSTD = 10

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models adapted to weighted graphs.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
        loss_type (str): MSE or MAE.
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None, loss_type = 'MSE'):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        self.loss_type = loss_type
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)


    def forward(self, x, edge_index, edge_weight) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(x, edge_index, edge_weight)


    def encode(self, x, edge_index, edge_weight) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(x, edge_index, edge_weight)


    def decode(self, z, edge_index) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(z, edge_index)


    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, positive_edge_weight: Tensor) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the MSE or Cross Entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges:

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        positive_edge_weight_hat = self.decoder(z, pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, z.shape[0])
        neg_edge_weight_hat = self.decoder(z, neg_edge_index)
        pos_loss =  positive_edge_weight - positive_edge_weight_hat
        loss_tensor = torch.cat([pos_loss, neg_edge_weight_hat], dim=0)
        if self.loss_type == 'MSE':
            return torch.mean(torch.pow(loss_tensor,2))
        else:
            return torch.mean(torch.abs(loss_tensor))

    def test(self, z: Tensor, pos_edge_index: Tensor, positive_edge_weight: Tensor,
             neg_edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes MSE and MAE scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred_diff = torch.cat([positive_edge_weight - pos_pred, neg_pred], dim=0)
        
        mse, mae = torch.mean(torch.pow(pred_diff, 2)), torch.mean(torch.abs(pred_diff))

        return mse, mae

class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper adapted to weighted graphs.

    Args:
        encoder (torch.nn.Module): The encoder module to compute :math:`\mu`
            and :math:`\log\sigma^2`.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None, loss_type = 'MSE'):
        super().__init__(encoder, decoder, loss_type)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs) -> Tensor:
        """"""  # noqa: D419
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))