from __future__ import unicode_literals, print_function, division
'''
    Adopt from PNAConv from https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
'''
from typing import Optional, List, Dict
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
from torch_scatter import scatter
from torch.nn import ModuleList, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import degree
from torch_geometric.nn.inits import reset


class PNAConv(MessagePassing):
    r"""
    Modified PNAConv.
        1. Configure directed graph.
        2. Allow local degree bincount. 
        In the case, attenuation scalar is same as Identity scalar.

    The Principal Neighbourhood Aggregation graph convolution operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        aggregators (list of str): Set of aggregation function identifiers,
            namely :obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`,
            :obj:`"var"` and :obj:`"std"`.
        scalers: (list of str): Set of scaling function identifiers, namely
            :obj:`"identity"`, :obj:`"amplification"`,
            :obj:`"attenuation"`, :obj:`"linear"` and
            :obj:`"inverse_linear"`.
        deg (Tensor): Histogram of in-degrees of nodes in the training set,
            used by scalers to normalize.
        edge_dim (int, optional): Edge feature dimensionality (in case
            there are any). (default :obj:`None`)
        towers (int, optional): Number of towers (default: :obj:`1`).
        pre_layers (int, optional): Number of transformation layers before
            aggregation (default: :obj:`1`).
        post_layers (int, optional): Number of transformation layers after
            aggregation (default: :obj:`1`).
        divide_input (bool, optional): Whether the input features should
            be split between towers or not (default: :obj:`False`).
        directed_msg (bool, optional): Specify whether it behaves as a
            directed graph. (default: :obj:`True`).
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor = None,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 divide_input: bool = False, directed_msg: bool = True,
                 **kwargs):

        kwargs.setdefault('aggr', None)
        super(PNAConv, self).__init__(node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.directed_msg = directed_msg

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if deg is not None:
            deg = deg.to(torch.float)
            self.avg_deg: Dict[str, float] = {
                'lin': deg.mean().item(),
                'log': (deg + 1).log().mean().item(),
                'exp': deg.exp().mean().item(),
            }
        else:
            self.avg_deg: Dict[str, float] = {
                'lin': 1.,
                'log': 1.,
                'exp': 1.,
            }


        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            if self.directed_msg:
                modules = [Linear((2 if edge_dim else 1) * self.F_in, self.F_in)]
            else:
                modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [ReLU()]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None,
                node_weight: OptTensor = None) -> Tensor:
        """"""

        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None,
                             node_weight=node_weight)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            if self.directed_msg:
                h = torch.cat([x_j, edge_attr], dim=-1)
            else:
                h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            if self.directed_msg:
                h = x_j
            else:
                h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  dim_size: Optional[int] = None,
                  node_weight: OptTensor = None) -> Tensor:

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == "depth" and node_weight is not None:
                out = out * node_weight
            elif scaler == 'amplification':
                deg, deg_bincount = self._get_degree(index, dim_size)
                avg_deg = (deg_bincount + 1).log().mean().item() \
                                if self.avg_deg["log"] <= 1. \
                                else self.avg_deg['log']
                out = out * (torch.log(deg + 1) / avg_deg)
            elif scaler == 'attenuation':
                deg, deg_bincount = self._get_degree(index, dim_size)
                avg_deg = (deg_bincount + 1).log().mean().item() \
                                if self.avg_deg["log"] <= 1. \
                                else self.avg_deg['log']
                out = out * (avg_deg / torch.log(deg + 1))
            elif scaler == 'linear':
                deg, deg_bincount = self._get_degree(index, dim_size)
                avg_deg = deg_bincount.mean().item() \
                                if self.avg_deg["lin"] <= 1. \
                                else self.avg_deg['lin']
                out = out * (deg / avg_deg)
            elif scaler == 'inverse_linear':
                deg, deg_bincount = self._get_degree(index, dim_size)
                avg_deg = deg_bincount.mean().item() \
                                if self.avg_deg["lin"] <= 1. \
                                else self.avg_deg['lin']
                out = out * (avg_deg / deg)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)
        return torch.cat(outs, dim=-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')

    def _get_degree(self, index, dim_size):
        deg = degree(index, dim_size, dtype=torch.long)
        deg_bincount = torch.bincount(deg, minlength=deg.numel()).to(torch.float)
        deg = deg.clamp_(1).view(-1, 1, 1)
        return deg, deg_bincount

'''
A high level module utilises PNAConv
'''
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
class PNANet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        n_layers,
        scalers,
        directed_msg=True,
    ):
        super().__init__()
        aggregators = ['mean', 'min', 'max']
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(n_layers):
            conv = PNAConv(in_channels=in_channels, out_channels=out_channels,
                           aggregators=aggregators, scalers=scalers, deg=None,
                           edge_dim=edge_dim, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False, directed_msg=directed_msg)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(out_channels))

    def forward(self, x, edge_index, edge_attr, node_weight=None):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr, node_weight)))
        return x
