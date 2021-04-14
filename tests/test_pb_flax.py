import functools

import jax
import jax.numpy as jnp
from flax import nn

from positional_bias.jax import FFTBias2d, FFTBias, NaiveBias2d, NaiveBias


seq_len = 28 * 28 + 2
num_heads = 10
batch_size = 16
embed_dim = 64

config1 = dict(
    full_seq_len=seq_len,
    n_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)

config2 = dict(
    full_seq_len=seq_len,
    n_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)

name2model = {
    "naive": NaiveBias,
    "naive_2d": NaiveBias2d,
    "fft": FFTBias,
    "fft_2d": FFTBias2d
}


def create_model(key, flax_module, input_shape, model_kwargs):
    """Creates and initializes the model."""

    @functools.partial(jax.jit, backend='cpu')
    def _create_model(key):
        module = flax_module.partial(**model_kwargs)
        with nn.stochastic(key):
            _, initial_params = module.init_by_shape(key, [(input_shape, jnp.float32)])
            model = nn.Model(module, initial_params)
        return model

    return _create_model(key)


def _test_flax(naive_config: dict, fft_config: dict):
    key = jax.random.PRNGKey(9)
    v = jax.random.uniform(key, shape=[batch_size, seq_len, num_heads, embed_dim])

    fft_pb = create_model(key, name2model[fft_config["pos_bias_type"]], v.shape, fft_config)
    orig_pb = create_model(key, name2model[naive_config["pos_bias_type"]], v.shape, naive_config)

    print(v.shape)
    ppb_fft, z_pb_fft = fft_pb(v, **fft_config)
    print(ppb_fft.shape, z_pb_fft.shape)
    print(v.shape)
    ppb_orig, z_pb_orig = orig_pb(v, **naive_config)
    print(ppb_orig.shape, z_pb_orig.shape)

    assert jnp.allclose(z_pb_orig, z_pb_fft, atol=1e-3), "Z not equal"
    assert jnp.allclose(ppb_orig, ppb_fft, atol=1e-3), "PPB not equal"


def test_pos_bias_full_flax():
    config1["pos_bias_type"] = "naive"
    config2["pos_bias_type"] = "fft"
    config1["bias_base_type"] = "full"
    config2["bias_base_type"] = "full"
    _test_flax(config1, config2)


def test_pos_bias_2d_full_flax():
    config1["pos_bias_type"] = "naive_2d"
    config2["pos_bias_type"] = "fft_2d"
    config1["bias_base_type"] = "full"
    config2["bias_base_type"] = "full"
    _test_flax(config1, config2)


def test_pos_bias_sym_flax():
    config1["pos_bias_type"] = "naive"
    config2["pos_bias_type"] = "fft"
    config1["bias_base_type"] = "symmetric"
    config2["bias_base_type"] = "symmetric"
    _test_flax(config1, config2)


def test_pos_bias_2d_sym_flax():
    config1["pos_bias_type"] = "naive_2d"
    config2["pos_bias_type"] = "fft_2d"
    config1["bias_base_type"] = "symmetric"
    config2["bias_base_type"] = "symmetric"
    _test_flax(config1, config2)
