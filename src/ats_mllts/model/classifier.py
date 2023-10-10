from __future__ import unicode_literals, print_function, division
import torch.nn as nn


class POSTagClassifier(nn.Module):
    def __init__(self, opts):
        super().__init__()
        layer_dims = eval(opts['layer_ndims'])
        n_class = opts['n_class']
        n_layers = len(layer_dims)
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            hidden_in = layer_dims[i]
            if i < n_layers - 1:
                hidden_out = layer_dims[i+1]
                layers.append(nn.Linear(hidden_in, hidden_out, bias=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(opts['dropout']))
            else:
                hidden_out = n_class
                layers.append(nn.Linear(hidden_in, hidden_out, bias=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.mlp(inputs)
        return x
