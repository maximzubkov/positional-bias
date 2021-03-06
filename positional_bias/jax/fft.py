import flax.nn as nn
import jax.numpy as jnp

from .base import init_bias, compute_w_shape


def _process(
        x: jnp.array,
        has_bos: bool,
        has_eos: bool,
):
    if has_bos or has_eos:
        last = ([0, 0],)
        if has_bos:
            last[0][0] += 1
        if has_eos:
            last[0][1] += 1
        padding = ([0, 0], ) * (len(x.shape) - 1) + last
        x = jnp.pad(x, pad_width=padding, mode='constant', constant_values=0)
    return x


def _compute_z_fft(w: jnp.array, shape_: int, bias_base_type: str, seq_len: int):
    # [num_heads, seq_len]
    if bias_base_type == "full":
        z = w[..., shape_ - seq_len: shape_ + seq_len - 1]
    elif bias_base_type == "symmetric":
        w_ = w[..., :seq_len]
        z = jnp.concatenate([
            jnp.expand_dims(w_[..., -1], axis=-1),  # w_{N-1}
            jnp.flip(w_[..., 1:], axis=-1),  # w_{N-1}, w_{N-2}, ..., w_{1}
            w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
        ], axis=-1)
    else:
        raise ValueError("Unknown bias base type")

    # z_fft has shape [num_heads, seq_len * 2 - 1, 2], the last two dims belongs to real and img parts
    return jnp.fft.rfft(z)


class FFTBias(nn.Module):
    def apply(
            self,
            v: jnp.array,
            bias_base_type: str,
            pos_bias_type: str,
            num_attention_heads: int,
            max_seq_len: int,
            has_bos: bool,
            has_eos: bool,
            lm: bool = False,
    ):
        shape_ = int(max_seq_len)

        # [batch_size, [bos] + [...] x seq_len + [eos], n_heads, emb_dim]
        v_ = v
        if has_bos:
            shape_ -= 1
            v_ = v_[:, 1:, :, :]
        if has_eos:
            shape_ -= 1
            v_ = v_[:, :-1, :, :]

        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * seq_len - 1

        w_shape = compute_w_shape(shape_=shape_, bias_base_type=bias_base_type)
        w = self.param(
            "w",
            (1, n_heads, w_shape),
            lambda key, shape: init_bias(
                shape_=shape_,
                n_heads=n_heads,
                bias_base_type=bias_base_type
            )
        )

        z_fft = _compute_z_fft(w, shape_=shape_, bias_base_type=bias_base_type, seq_len=seq_len)

        v_ = jnp.transpose(v_, axes=[0, 3, 2, 1])

        v_ = jnp.pad(v_, pad_width=([0, 0], [0, 0], [0, 0], [shape_ - 1, 0]), mode='constant', constant_values=0)
        v_fft = jnp.fft.rfft(v_)

        pbv = jnp.fft.irfft(v_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        pbv = pbv[..., :seq_len]
        pbv = _process(pbv, has_bos=has_bos, has_eos=has_eos)
        pbv = jnp.transpose(pbv, axes=[0, 3, 2, 1])

        return pbv


class FFTBias2d(nn.Module):

    def apply(
            self,
            v: jnp.array,
            bias_base_type: str,
            pos_bias_type: str,
            num_attention_heads: int,
            max_seq_len: int,
            has_bos: bool,
            has_eos: bool,
            n_channels: int = 1,
            lm: bool = False,
    ):
        # [batch_size, [bos] + [...] x seq_len + [eos], seq_len]
        v_ = v
        if has_bos:
            max_seq_len -= 1
            v_ = v_[:, 1:, :, :]
        if has_eos:
            max_seq_len -= 1
            v_ = v_[:, :-1, :, :]

        shape_ = int((max_seq_len / n_channels) ** 0.5)

        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * shape_ - 1

        w_shape = compute_w_shape(shape_=shape_, bias_base_type=bias_base_type)
        w = self.param(
            "w",
            (1, n_heads, w_shape),
            lambda key, shape: init_bias(
                shape_=shape_,
                n_heads=n_heads,
                bias_base_type=bias_base_type
            )
        )

        z_fft = _compute_z_fft(w, shape_=shape_, bias_base_type=bias_base_type, seq_len=shape_)

        v_ = jnp.transpose(v_, axes=[0, 3, 2, 1])
        v_ = jnp.reshape(v_, newshape=[batch_size, emb_dim, n_heads, shape_, shape_, n_channels])
        v_ = jnp.transpose(v_, axes=[0, 1, 5, 3, 2, 4])

        pad_width = ([0, 0], [0, 0], [0, 0], [0, 0], [shape_ - 1, 0])

        v_s = v_.sum(-3)
        v_m = jnp.pad(v_s, pad_width=pad_width, mode='constant', constant_values=0)
        v_m_fft = jnp.fft.rfft(v_m)

        u_s = jnp.transpose(v_, axes=[0, 1, 2, 5, 4, 3]).sum(-3)
        u_m = jnp.pad(u_s, pad_width=pad_width, mode='constant', constant_values=0)
        u_m_fft = jnp.fft.rfft(u_m)

        RxV_m = jnp.fft.irfft(v_m_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        RxV_m = RxV_m[..., :shape_]
        RxU_m = jnp.fft.irfft(u_m_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        RxU_m = RxU_m[..., :shape_]

        pbv = jnp.expand_dims(RxV_m, axis=-2) + jnp.expand_dims(RxU_m, axis=-1)
        pbv = jnp.reshape(pbv, newshape=[batch_size, emb_dim, n_channels, n_heads, shape_ * shape_])
        pbv = jnp.transpose(pbv, axes=[0, 1, 3, 4, 2])
        pbv = jnp.reshape(pbv, newshape=[batch_size, emb_dim, n_heads, seq_len])
        pbv = _process(pbv, has_bos=has_bos, has_eos=has_eos)
        pbv = jnp.transpose(pbv, axes=[0, 3, 2, 1])

        return pbv
