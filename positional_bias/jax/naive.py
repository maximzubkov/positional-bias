import flax.nn as nn
import jax.numpy as jnp

from .base import init_bias, compute_w_shape


def _process(
        w_: jnp.array,
        batch_size: int,
        has_bos: bool,
        has_eos: bool,
):
    if has_bos or has_eos:
        last = [0, 0]
        if has_bos:
            last[0] += 1
        if has_eos:
            last[1] += 1
        last = tuple(last)
        w_ = jnp.pad(w_, pad_width=[(0, 0), (0, 0), last, last], mode='constant', constant_values=0)

    if (len(w_.shape) == 4) and (w_.shape[0] == 1):
        w_ = jnp.repeat(w_, repeats=batch_size, axis=0)
    return w_


def _construct_bias(w_: jnp.array, seq_len: int, bias_base_type: str):
    if bias_base_type == "full":
        w_ = jnp.concatenate([
            w_[..., 1:],  # w_{2N-1}, w_{2N-2}, ..., w_{1}
            jnp.expand_dims(w_[..., 0], axis=-1),  # w_{0}
        ], axis=-1)
    elif bias_base_type == "symmetric":
        w_ = jnp.concatenate([
            jnp.flip(w_[..., 1:], axis=-1),
            w_
        ], axis=-1)
    else:
        raise ValueError("Unknown bias base type")
    bias = jnp.concatenate([
        jnp.expand_dims(w_[..., seq_len - i - 1: 2 * seq_len - i - 1], axis=-1)
        for i in range(seq_len)
    ], axis=-1)

    return bias


class NaiveBias(nn.Module):

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

        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape

        if has_bos:
            shape_ -= 1
            seq_len -= 1
        if has_eos:
            shape_ -= 1
            seq_len -= 1

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

        if bias_base_type == "full":
            w_ = w[..., shape_ - seq_len: shape_ + seq_len - 1]
        elif bias_base_type == "symmetric":
            w_ = w[..., :seq_len]
        else:
            raise ValueError("Unknown bias base type")

        bias = _construct_bias(w_, seq_len=seq_len, bias_base_type=bias_base_type)
        bias = _process(bias, batch_size=batch_size, has_bos=has_bos, has_eos=has_eos)
        pbv = jnp.einsum("nlhd,nhlj->njhd", v, jnp.transpose(bias, axes=[0, 1, 3, 2]))
        return pbv


class NaiveBias2d(nn.Module):

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
        if has_bos:
            max_seq_len -= 1
        if has_eos:
            max_seq_len -= 1

        shape_ = int(max_seq_len ** 0.5)
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape

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

        bias = _construct_bias(w, seq_len=shape_, bias_base_type=bias_base_type)
        x_ = jnp.expand_dims(bias, axis=-3)
        x_ = jnp.expand_dims(x_, axis=-2)
        y_ = jnp.expand_dims(bias, axis=-2)
        y_ = jnp.expand_dims(y_, axis=-1)
        w_ = x_ + y_
        w_batch_shape, *_ = w_.shape
        w_ = jnp.reshape(w_, newshape=[w_batch_shape, n_heads, shape_, shape_, -1])
        w_ = jnp.reshape(w_, newshape=[w_batch_shape, n_heads, -1, shape_ ** 2])
        w_ = _process(w_, batch_size=batch_size, has_bos=has_bos, has_eos=has_eos)
        pbv = jnp.einsum("nlhd,nhlj->njhd", v, jnp.transpose(w_, axes=[0, 1, 3, 2]))
        return pbv
