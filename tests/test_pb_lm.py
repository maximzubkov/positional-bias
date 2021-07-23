import torch
from einops import repeat

from positional_bias.pytorch import PositionalBias

seq_len = 28 * 28
num_heads = 10
batch_size = 16
embed_dim = 64

config = dict(
    max_seq_len=seq_len,
    num_attention_heads=num_heads,
    pos_bias_type="naive",
    bias_base_type="full",
    lm=True,
    has_bos=False,
    has_eos=False,
)

v = torch.rand(batch_size, seq_len, num_heads, embed_dim)


@torch.no_grad()
def test_lm_naive_2d():
    cumsum = 2 * torch.cumsum(v, dim=-3)
    config["pos_bias_type"] = "naive_2d"
    pos_bias = PositionalBias(**config)
    pos_bias.eval()
    pos_bias.bias.w.data = repeat(torch.ones(2 * int(seq_len ** 0.5) - 1), 's -> h s', h=num_heads).unsqueeze(0)

    ppb = pos_bias(v)
    assert torch.allclose(ppb, cumsum, atol=1e-3), "Cumsum and new v are not equal"


@torch.no_grad()
def test_lm_naive():
    cumsum = torch.cumsum(v, dim=-3)
    config["pos_bias_type"] = "naive"
    pos_bias = PositionalBias(**config)
    pos_bias.eval()
    pos_bias.bias.w.data = repeat(torch.ones(2 * seq_len - 1), 's -> h s', h=num_heads).unsqueeze(0)

    ppb = pos_bias(v)
    assert torch.allclose(ppb, cumsum, atol=1e-3), "Cumsum and new v are not equal"
