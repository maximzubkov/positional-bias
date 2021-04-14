import torch

from positional_bias.pytorch import PositionalBias

torch.manual_seed(9)

seq_len = 28 * 28 + 2
num_heads = 10
batch_size = 16
embed_dim = 64

config1 = dict(
    seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)

config2 = dict(
    seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="",
    bias_base_type="",
    lm=False,
    has_specials=True,
)


@torch.no_grad()
def _test_pt(naive_config: dict, fft_config: dict):
    v = torch.rand(batch_size, seq_len, num_heads, embed_dim)
    fft_pos_bias = PositionalBias(**fft_config)
    fft_pos_bias.eval()

    naive_pos_bias = PositionalBias(**naive_config)
    naive_pos_bias.eval()

    w_ = torch.rand(1, num_heads, naive_pos_bias.bias.w_shape)
    naive_pos_bias.bias.w.data = w_
    fft_pos_bias.bias.w.data = w_

    ppb_fft, z_pb_fft = fft_pos_bias(v)
    ppb_orig, z_pb_orig = naive_pos_bias(v)
    assert torch.allclose(z_pb_orig, z_pb_fft, atol=1e-3), "Z not equal"
    assert torch.allclose(ppb_orig, ppb_fft, atol=1e-3), "PPB not equal"


def _test_flax(naive_config: dict, fft_config: dict):
    pass


def test_pos_bias_full():
    config1["pos_bias_type"] = "naive"
    config2["pos_bias_type"] = "fft"
    config1["bias_base_type"] = "full"
    config2["bias_base_type"] = "full"
    _test_pt(config1, config2)
    _test_flax(config1, config2)


def test_pos_bias_2d_full():
    config1["pos_bias_type"] = "naive_2d"
    config2["pos_bias_type"] = "fft_2d"
    config1["bias_base_type"] = "full"
    config2["bias_base_type"] = "full"
    _test_pt(config1, config2)
    _test_flax(config1, config2)


def test_pos_bias_sym():
    config1["pos_bias_type"] = "naive"
    config2["pos_bias_type"] = "fft"
    config1["bias_base_type"] = "symmetric"
    config2["bias_base_type"] = "symmetric"
    _test_pt(config1, config2)
    _test_flax(config1, config2)


def test_pos_bias_2d_sym():
    config1["pos_bias_type"] = "naive_2d"
    config2["pos_bias_type"] = "fft_2d"
    config1["bias_base_type"] = "symmetric"
    config2["bias_base_type"] = "symmetric"
    _test_pt(config1, config2)
    _test_flax(config1, config2)
