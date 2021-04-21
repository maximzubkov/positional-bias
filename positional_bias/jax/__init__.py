import functools

import flax.nn as nn
import jax
import jax.numpy as jnp

from .fft import FFTBias, FFTBias2d
from .naive import NaiveBias, NaiveBias2d

__all__ = [
    "create_model",
    "name2model"
]


name2model = {
    "naive": NaiveBias,
    "naive_2d": NaiveBias2d,
    "fft": FFTBias,
    "fft_2d": FFTBias2d
}


def create_model(flax_module: nn.Module, input_shape: tuple, model_kwargs: dict, key):
    """Creates and initializes the model."""

    @functools.partial(jax.jit, backend='cpu')
    def _create_model(key_):
        module = flax_module.partial(**model_kwargs)
        with nn.stochastic(key_):
            _, initial_params = module.init_by_shape(key_, [(input_shape, jnp.float32)])
            model = nn.Model(module, initial_params)
        return model

    return _create_model(key)
