import jax.numpy as jnp

from .base import BiasBase


class FFTBiasBase(BiasBase):
    def _process(self, x: jnp.array):
        if self.has_specials:
            x = jnp.pad(x, pad_width=[1, 1], mode='constant', constant_values=0)
        return x

    def _compute_z_fft(self, seq_len: int):
        # [num_heads, seq_len]
        if self.bias_base_type == "full":
            z = self.w[..., self.shape_ - seq_len: self.shape_ + seq_len - 1]
        elif self.bias_base_type == "symmetric":
            w_ = self.w[..., :seq_len]
            z = jnp.concatenate([
                jnp.expand_dims(w_[..., -1], axis=-1),  # w_{N-1}
                jnp.flip(w_[..., 1:], axis=-1),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], axis=-1)
        else:
            raise ValueError("Unknown bias base type")

        # z_fft has shape [num_heads, seq_len * 2 - 1, 2], the last two dims belongs to real and img parts
        return jnp.fft.rfft(z)


class FFTBias(FFTBiasBase):
    o_: jnp.array

    def setup(self):
        self.shape_ = int(self.full_seq_len)
        self._init_bias()

        self.o_ = jnp.ones(self.shape_)

    def apply(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], n_heads, emb_dim]
        v_ = v[:, 1:-1, :, :] if self.has_specials else v
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * seq_len - 1
        z_fft = self._compute_z_fft(seq_len)

        v_ = jnp.transpose(v_, axes=[0, 3, 2, 1])

        pad_size = seq_len - 1

        v_ = jnp.pad(v_, pad_width=[pad_size, 0])
        v_fft = jnp.fft.rfft(v_)

        pbv = jnp.fft.irfft(v_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        pbv = pbv[..., :seq_len]
        pbv = self._process(pbv)
        pbv = jnp.transpose(pbv, axes=[0, 3, 2, 1])

        o_ = jnp.pad(self.o_[:seq_len], pad_width=[pad_size, 0])
        o_fft = jnp.fft.rfft(o_)

        z_pb = jnp.fft.irfft(z_fft * o_fft, n=n)
        z_pb = z_pb[..., :seq_len]
        z_pb = self._process(z_pb)
        z_pb = jnp.transpose(z_pb, axes=[0, 1, 3, 2])
        return pbv, z_pb


class FFTBias2d(FFTBiasBase):
    o_: jnp.array

    def setup(self):
        self.shape_ = int(self.full_seq_len ** 0.5)
        self._init_bias()

        self.o_ = jnp.ones(self.shape_)
        self.o_ = jnp.pad(self.o_, pad_width=[self.shape_ - 1, 0])

    def apply(self, v):
        # [batch_size, [bos] + [...] x seq_len + [eos], seq_len]
        v_ = v[:, 1:-1, :, :] if self.has_specials else v
        batch_size, seq_len, n_heads, emb_dim = v_.shape
        n = 2 * self.shape_ - 1
        z_fft = self._compute_z_fft(self.shape_)

        v_ = jnp.transpose(v_, axes=[0, 3, 2, 1])
        v_ = jnp.reshape(v_, newshape=[batch_size, emb_dim, n_heads, self.shape_, self.shape_])
        v_ = jnp.transpose(v_, axes=[0, 2, 1, 3])

        v_s = jnp.sum(v_, axis=-3)
        v_m = jnp.pad(v_s, pad_width=[self.shape_ - 1, 0])
        v_m_fft = jnp.fft.rfft(v_m)

        u_s = jnp.sum(jnp.transpose(v_, axes=[0, 3, 2, 1]), axis=-3)
        u_m = jnp.pad(u_s, pad_width=[self.shape_ - 1, 0])
        u_m_fft = jnp.fft.rfft(u_m)

        RxV_m = jnp.fft.irfft(v_m_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        RxV_m = RxV_m[..., :self.shape_]
        RxU_m = jnp.fft.irfft(u_m_fft * jnp.expand_dims(z_fft, axis=1), n=n)
        RxU_m = RxU_m[..., :self.shape_]

        pbv = jnp.expand_dims(RxV_m, axis=-2) + jnp.expand_dims(RxU_m, axis=-1)
        pbv = jnp.reshape(pbv, newshape=[batch_size, emb_dim, n_heads, seq_len])
        pbv = self._process(pbv)
        pbv = jnp.transpose(pbv, axes=[0, 3, 2, 1])

        o_fft = jnp.fft.rfft(self.o_)
        z_pb = jnp.fft.irfft(o_fft * z_fft, n=n)
        z_pb = z_pb[..., :self.shape_] * self.shape_

        z_pb = jnp.expand_dims(z_pb[..., -1], axis=-2) + jnp.expand_dims(z_pb[..., -1], axis=-1)
        z_pb = jnp.reshape(z_pb, newshape=[-1, n_heads, self.shape_ * self.shape_])
        z_pb = self._process(z_pb)
        z_pb = jnp.transpose(z_pb, axes=[0, 1, 3, 2])

        return pbv, z_pb
