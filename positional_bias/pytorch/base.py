import torch
import torch.nn as nn


class BiasBase(nn.Module):
    def __init__(
            self,
            bias_base_type: str,
            pos_bias_type: str,
            num_attention_heads: int,
            max_seq_len: int,
            has_bos: bool,
            has_eos: bool,
            lm: bool,
    ) -> None:
        super(BiasBase, self).__init__()
        self.bias_base_type = bias_base_type
        self.pos_bias_type = pos_bias_type
        self.lm = lm
        self.has_bos = has_bos
        self.has_eos = has_eos
        self.n_heads = num_attention_heads
        self.max_seq_len = max_seq_len
        self.shape_ = 0

        if self.has_bos:
            self.max_seq_len = self.max_seq_len - 1
        if self.has_eos:
            self.max_seq_len = self.max_seq_len - 1

        if self.lm:
            ones = torch.ones(self.max_seq_len, self.max_seq_len)
            self.mask = nn.Parameter(
                torch.tril(ones).unsqueeze(0),
                requires_grad=False
            )

    def _init_bias(self):
        w_ = torch.arange(self.shape_).unsqueeze(0)
        w_ = w_ * 0.00001 * torch.rand(self.n_heads, 1) + 0.000001 * torch.randn(self.n_heads, 1)

        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape_ - 1
            w_ = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1).unsqueeze(0)
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape_
            w_ = w_.unsqueeze(0)
        else:
            raise ValueError("Unknown bias base type")

        self.w = torch.nn.Parameter(w_, requires_grad=True)
