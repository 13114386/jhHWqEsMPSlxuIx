from __future__ import unicode_literals, print_function, division
import math
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_size):
        super().__init__()
        self.convd1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same")
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=output_size, return_indices=False)

    def forward(self, inputs):
        x = inputs.transpose(1,2) # [nb, nl, nd] -> [nb, nd, nl]
        x = self.convd1(x)
        nb, nd, nl = x.size()
        x = x.transpose(1,2) # [nb, nd, nl] -> [nb, nl, nd]
        x = x.reshape(-1, nd)
        x = self.layer_norm(x)
        x = x.view(nb, nl, nd)
        x = self.activation(x)
        x = x.transpose(1,2) # [nb, nl, nd] -> [nb, nd, nl]
        x = self.pool(x)
        return x.transpose(1,2)

class CnnModel(nn.Module):
    def __init__(self, n_channels, kernel_size, output_size_factor, max_length):
        super().__init__()
        self.layers = nn.ModuleList()
        n_blocks = math.floor(math.log(max_length, output_size_factor))
        for i in range(n_blocks):
            output_size = max_length//output_size_factor**(i+1)
            block = ConvBlock(in_channels=n_channels,
                              out_channels=n_channels,
                              kernel_size=kernel_size,
                              output_size=output_size)
            self.layers.append(block)
        self.last_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = x.transpose(1,2)
        x = self.last_pool(x)
        x = x.transpose(1,2)
        return x
