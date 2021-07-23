import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BiasBase


class FFTBiasBase(BiasBase):
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
        super(FFTBiasBase, self).__init__(
            bias_base_type=bias_base_type,
            pos_bias_type=pos_bias_type,
            num_attention_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            has_bos=has_bos,
            has_eos=has_eos,
            lm=lm,
        )

    def _process(self, x: torch.Tensor):
        if self.has_bos or self.has_eos:
            pad = [0, 0]
            if self.has_bos:
                pad[0] += 1
            if self.has_eos:
                pad[1] += 1
            x = F.pad(input=x, pad=pad, mode='constant', value=0)
        return x

    def _compute_z_fft(self, seq_len: int):
        # [num_heads, seq_len]
        if self.bias_base_type == "full":
            z = self.w[..., self.shape_ - seq_len: self.shape_ + seq_len - 1]
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
            z = torch.cat([
                w_[..., -1].unsqueeze(-1),  # w_{N-1}
                torch.flip(w_[..., 1:], dims=[-1]),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], dim=-1)
        else:
            raise ValueError("Unknown bias base type")

        if self.lm:
            mask = torch.ones_like(z)
            *_, shape = z.shape
            mask_len = (shape + 1) // 2
            mask[mask_len:] = 0
            z = z * mask
        # z_fft has shape [num_heads, seq_len * 2 - 1, 2], the last two dims belongs to real and img parts
        return torch.fft.rfft(z)


class FFTBias(FFTBiasBase):
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
        super(FFTBias, self).__init__(
            bias_base_type=bias_base_type,
            pos_bias_type=pos_bias_type,
            num_attention_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            has_bos=has_bos,
            has_eos=has_eos,
            lm=lm,
        )

        self.shape_ = int(self.max_seq_len)
        self._init_bias()

    def forward(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], n_heads, emb_dim]
        v_ = v
        if self.has_bos:
            v_ = v_[:, 1:, :, :]
        if self.has_eos:
            v_ = v_[:, :-1, :, :]
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * seq_len - 1
        z_fft = self._compute_z_fft(seq_len)

        v_ = v_.permute(0, 3, 2, 1)

        pad_size = seq_len - 1

        v_ = nn.functional.pad(v_, [pad_size, 0])
        v_fft = torch.fft.rfft(v_)

        pbv = torch.fft.irfft(v_fft * z_fft.unsqueeze(1), n=n)
        pbv = pbv[..., :seq_len]
        pbv = self._process(pbv)
        pbv = pbv.permute(0, 3, 2, 1)

        return pbv


class FFTBias2d(FFTBiasBase):
    def __init__(
            self,
            bias_base_type: str,
            pos_bias_type: str,
            num_attention_heads: int,
            max_seq_len: int,
            has_bos: bool,
            has_eos: bool,
            n_channels: int = 1,
            lm: bool = False,
    ) -> None:
        super(FFTBias2d, self).__init__(
            bias_base_type=bias_base_type,
            pos_bias_type=pos_bias_type,
            num_attention_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            has_bos=has_bos,
            has_eos=has_eos,
            lm=lm,
        )
        self.shape_ = int((max_seq_len / n_channels) ** 0.5)
        self.n_channels = n_channels
        self._init_bias()

    def forward(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], n_heads, emb_dim]
        # where seq_len is H x W x C for images
        v_ = v
        if self.has_bos:
            v_ = v_[:, 1:, :, :]
        if self.has_eos:
            v_ = v_[:, :-1, :, :]

        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * self.shape_ - 1
        z_fft = self._compute_z_fft(self.shape_)

        v_ = v_.transpose(-3, -1)
        v_ = v_.reshape(batch_size, emb_dim, n_heads, self.shape_, self.shape_, self.n_channels)
        v_ = v_.permute(0, 1, 5, 3, 2, 4)

        v_m = nn.functional.pad(v_.sum(-3), [self.shape_ - 1, 0])
        v_m_fft = torch.fft.rfft(v_m)

        u_m = nn.functional.pad(v_.transpose(-3, -1).sum(-3), [self.shape_ - 1, 0])
        u_m_fft = torch.fft.rfft(u_m)

        RxV_m = torch.fft.irfft(v_m_fft * z_fft.unsqueeze(1), n=n)
        RxV_m = RxV_m[..., :self.shape_]
        RxU_m = torch.fft.irfft(u_m_fft * z_fft.unsqueeze(1), n=n)
        RxU_m = RxU_m[..., :self.shape_]

        pbv = RxV_m.unsqueeze(-2) + RxU_m.unsqueeze(-1)
        pbv = pbv.reshape(batch_size, emb_dim, self.n_channels, n_heads, -1)
        pbv = pbv.permute(0, 1, 3, 4, 2)
        pbv = pbv.reshape(batch_size, emb_dim, n_heads, seq_len)
        pbv = self._process(pbv)
        pbv = pbv.permute(0, 3, 2, 1)

        return pbv
