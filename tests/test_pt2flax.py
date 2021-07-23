import jax
import numpy as np
import torch

from positional_bias.jax import create_model, name2model
from positional_bias.pytorch import PositionalBias

seq_len = 28 * 28 + 2
num_heads = 3
batch_size = 4
embed_dim = 12

config = dict(
    max_seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_bos=True,
    has_eos=True,
)


def _test_pt2flax(config_: dict):
    key = jax.random.PRNGKey(9)
    v_jax = jax.random.uniform(key, shape=[batch_size, seq_len, num_heads, embed_dim])
    v_pt = torch.Tensor(np.array(v_jax))

    flax_model = name2model[config_["pos_bias_type"]]

    flax_pb = create_model(flax_model, v_jax.shape, config_, key)
    pytorch_pb = PositionalBias(**config_)

    w_np = np.array(flax_pb.params["w"])
    pytorch_pb.bias.w.data = torch.Tensor(w_np)

    ppb_flax = flax_pb(v_jax, **config_)
    ppb_pytorch = pytorch_pb(v_pt)

    ppb_flax_np = np.array(ppb_flax)
    ppb_pytorch_np = ppb_pytorch.detach().cpu().numpy()

    assert np.allclose(ppb_flax_np, ppb_pytorch_np, atol=1e-5), "PPB not equal"


def test_pt2flax_naive_full():
    config["pos_bias_type"] = "naive"
    config["bias_base_type"] = "full"
    _test_pt2flax(config)


def test_pt2flax_naive_sym():
    config["pos_bias_type"] = "naive"
    config["bias_base_type"] = "symmetric"
    _test_pt2flax(config)


def test_pt2flax_fft_full():
    config["pos_bias_type"] = "fft"
    config["bias_base_type"] = "full"
    _test_pt2flax(config)


def test_pt2flax_fft_sym():
    config["pos_bias_type"] = "fft"
    config["bias_base_type"] = "symmetric"
    _test_pt2flax(config)


def test_pt2flax_fft2d_full():
    config["pos_bias_type"] = "fft_2d"
    config["bias_base_type"] = "full"
    _test_pt2flax(config)


def test_pt2flax_fft2d_sym():
    config["pos_bias_type"] = "fft_2d"
    config["bias_base_type"] = "symmetric"
    _test_pt2flax(config)


def test_pt2flax_naive2d_full():
    config["pos_bias_type"] = "naive_2d"
    config["bias_base_type"] = "full"
    _test_pt2flax(config)


def test_pt2flax_naive2d_sym():
    config["pos_bias_type"] = "naive_2d"
    config["bias_base_type"] = "symmetric"
    _test_pt2flax(config)
