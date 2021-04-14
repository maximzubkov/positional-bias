import jax
import jax.numpy as jnp
from flax import nn
from jax.random import normal, uniform


class BiasBase(nn.Module):
    bias_base_type: str
    pos_bias_type: str
    n_heads: int
    full_seq_len: int
    has_specials: bool
    shape_: int = 0
    alpha: float = 0.00001
    beta: float = 0.000001
    key = jax.random.PRNGKey(9)

    def setup(self):
        if self.has_specials:
            self.full_seq_len = self.full_seq_len - 2

    def _init_bias(self):
        w_ = jnp.arange(self.shape_)
        w_ = jnp.expand_dims(w_, axis=0)
        w_ = w_ * self.alpha * normal(self.key, shape=(self.n_heads, 1)) \
            + self.beta * uniform(self.key, shape=(self.n_heads, 1))

        if self.bias_base_type == "full":
            self.w_shape = 2 * self.shape_ - 1
            w_ = jnp.concatenate([
                jnp.expand_dims(w_[..., -1], axis=-1),  # w_{N-1}
                jnp.flip(w_[..., 1:], axis=-1),  # w_{N-1}, w_{N-2}, ..., w_{1}
                w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
            ], axis=-1)
            w_ = jnp.expand_dims(w_, axis=0)
        elif self.bias_base_type == "symmetric":
            self.w_shape = self.shape_
            w_ = jnp.expand_dims(w_, axis=0)
        else:
            raise ValueError("Unknown bias base type")

        self.w = w_
