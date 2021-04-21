import jax
import jax.numpy as jnp
from jax.random import normal, uniform


def compute_w_shape(shape_: int, bias_base_type: str):
    if bias_base_type == "full":
        w_shape = 2 * shape_ - 1
    elif bias_base_type == "symmetric":
        w_shape = shape_
    else:
        raise ValueError("Unknown bias base type")
    return w_shape


def init_bias(shape_: int, n_heads: int, bias_base_type: str):
    key = jax.random.PRNGKey(9)
    w_ = jnp.arange(shape_)
    w_ = jnp.expand_dims(w_, axis=0)
    w_ = w_ * 0.00001 * normal(key, shape=(n_heads, 1)) \
        + 0.000001 * uniform(key, shape=(n_heads, 1))

    if bias_base_type == "full":
        w_ = jnp.concatenate([
            jnp.expand_dims(w_[..., -1], axis=-1),  # w_{N-1}
            jnp.flip(w_[..., 1:], axis=-1),  # w_{N-1}, w_{N-2}, ..., w_{1}
            w_[..., :-1]  # w_{0}, w_{1}, ..., w_{N-2}
        ], axis=-1)
        w_ = jnp.expand_dims(w_, axis=0)
    elif bias_base_type == "symmetric":
        w_ = jnp.expand_dims(w_, axis=0)
    else:
        raise ValueError("Unknown bias base type")
    return w_
