import torch.nn as nn

from .fft import FFTBias, FFTBias2d
from .naive import NaiveBias, NaiveBias2d

__all__ = ["PositionalBias"]


class PositionalBias(nn.Module):
    def __init__(self, **kwargs):
        super(PositionalBias, self).__init__()
        if kwargs["pos_bias_type"] == "naive":
            self.bias = NaiveBias(**kwargs)
        elif kwargs["pos_bias_type"] == "naive_2d":
            self.bias = NaiveBias2d(**kwargs)
        elif kwargs["pos_bias_type"] == "fft":
            self.bias = FFTBias(**kwargs)
        elif kwargs["pos_bias_type"] == "fft_2d":
            self.bias = FFTBias2d(**kwargs)

    def forward(self, v):
        return self.bias(v)
