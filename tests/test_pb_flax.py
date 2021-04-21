import jax
import jax.numpy as jnp

from positional_bias.jax import create_model, name2model

seq_len = 28 * 28 + 2
num_heads = 3
batch_size = 4
embed_dim = 12

config1 = dict(
    max_seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)

config2 = dict(
    max_seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)


def _test_flax(naive_config: dict, fft_config: dict):
    key = jax.random.PRNGKey(9)
    v = jax.random.uniform(key, shape=[batch_size, seq_len, num_heads, embed_dim])

    fft_model = name2model[fft_config["pos_bias_type"]]
    naive_model = name2model[naive_config["pos_bias_type"]]

    fft_pb = create_model(fft_model, v.shape, fft_config, key)
    orig_pb = create_model(naive_model, v.shape, naive_config, key)

    ppb_fft, z_pb_fft = fft_pb(v, **fft_config)
    ppb_orig, z_pb_orig = orig_pb(v, **naive_config)

    assert jnp.allclose(z_pb_orig, z_pb_fft, atol=0.3e-1), "Z not equal"
    assert jnp.allclose(ppb_orig, ppb_fft, atol=0.3e-1), "PPB not equal"


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
