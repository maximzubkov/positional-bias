import flax.nn as nn

from .fft import FFTBias, FFTBias2d
from .naive import NaiveBias, NaiveBias2d

__all__ = ["PositionalBias"]


class PositionalBias(nn.Module):
    def __call__(self, **kwargs):
        if kwargs["pos_bias_type"] == "naive":
            return NaiveBias()(**kwargs)
        elif kwargs["pos_bias_type"] == "naive_2d":
            return NaiveBias2d()(**kwargs)
        elif kwargs["pos_bias_type"] == "fft":
            return FFTBias()(**kwargs)
        elif kwargs["pos_bias_type"] == "fft_2d":
            return FFTBias2d()(**kwargs)
