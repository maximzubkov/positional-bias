import jax.numpy as jnp

from .base import BiasBase


class NaiveBiasBase(BiasBase):
    def _process(self, w_: jnp.array, batch_size: int):
        if self.has_specials:
            w_ = jnp.pad(w_, pad_width=[1, 1, 1, 1], mode='constant', constant_values=0)

        if (len(w_.shape) == 4) and (w_.shape[0] == 1):
            w_ = w_.squeeze()
            w_ = jnp.repeat(w_, repeats=batch_size, axis=0)
        return w_

    def _construct_bias(self, w_: jnp.array, seq_len: int):
        if self.bias_base_type == "full":
            w_ = jnp.concatenate([
                w_[..., 1:],  # w_{2N-1}, w_{2N-2}, ..., w_{1}
                jnp.expand_dims(w_[..., 0], axis=-1),  # w_{0}
            ], axis=-1)
        elif self.bias_base_type == "symmetric":
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


class NaiveBias(NaiveBiasBase):
    def setup(self):
        self.shape_ = self.full_seq_len
        self._init_bias()

    def apply(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        if self.has_specials:
            seq_len -= 2

        if self.bias_base_type == "full":
            w_ = self.w[..., self.shape_ - seq_len: self.shape_ + seq_len - 1]
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
        else:
            raise ValueError("Unknown bias base type")

        bias = self._construct_bias(w_, seq_len)
        bias = self._process(bias, batch_size)
        z_pb = jnp.transpose(bias.sum(-1), axes=[0, 1, 3, 2])

        pbv = jnp.einsum("nlhd,nhlj->njhd", v, jnp.transpose(bias, axes=[0, 1, 3, 2]))
        return pbv, z_pb


class NaiveBias2d(NaiveBiasBase):
    def setup(self):
        self.shape_ = int(self.full_seq_len ** 0.5)
        self._init_bias()

    def apply(self, v):
        # [batch_size, seq_len, seq_len]
        batch_size, seq_len, n_heads, emb_dim = v.shape
        bias = self._construct_bias(self.w, self.shape_)
        x_ = jnp.expand_dims(bias, axis=-3)
        x_ = jnp.expand_dims(x_, axis=-2)
        y_ = jnp.expand_dims(bias, axis=-2)
        y_ = jnp.expand_dims(y_, axis=-1)
        w_ = x_ + y_
        w_batch_shape, *_ = w_.shape
        w_ = jnp.reshape(w_, newshape=[w_batch_shape, n_heads, self.shape_, self.shape_, -1])
        w_ = jnp.reshape(w_, newshape=[w_batch_shape, n_heads, -1, self.shape_ ** 2])
        w_ = self._process(w_, batch_size)
        z_pb = jnp.transpose(w_.sum(-1), axes=[0, 1, 3, 2])
        pbv = jnp.einsum("nlhd,nhlj->njhd", v, jnp.transpose(w_, axes=[0, 1, 3, 2]))
        return pbv, z_pb
