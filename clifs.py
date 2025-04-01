from typing import Callable, NamedTuple

import jax
import optax
import seaborn as sns
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jaxtyping import Array, Key, PyTree, Scalar
from optax import GradientTransformation, OptState


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
dim = 2


def base_logdensity_fn(x: Array) -> Scalar:
    return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), jnp.eye(dim))


def base_sample_fn(rng_key: Key, num_samples: int = 1) -> Array:
    return jr.multivariate_normal(
        rng_key, jnp.zeros(dim), jnp.eye(dim), (num_samples,)
    )


def target_logdensity_fn(x: Array) -> Scalar:
    assert x.shape[-1] == 2 
    m1 = jsp.stats.multivariate_normal.logpdf(x, jnp.array([2.0, 2.0]), jnp.eye(2))
    m2 = jsp.stats.multivariate_normal.logpdf(x, jnp.array([-2.0, -2.0]), jnp.eye(2))
    m3 = jsp.stats.multivariate_normal.logpdf(x, jnp.array([2.0, -2.0]), jnp.eye(2))
    m4 = jsp.stats.multivariate_normal.logpdf(x, jnp.array([-2.0, 2.0]), jnp.eye(2))
    log_mixtures = jnp.array([m1, m2, m3, m4])
    log_weights = jnp.log(jnp.array([0.25, 0.25, 0.25, 0.25]))
    assert log_mixtures.ndim == 1
    return jsp.special.logsumexp(log_mixtures + log_weights)


def probability_path_logdensity_fn(x: Array, time: Scalar) -> Scalar:
    """Logdensity of time-dependent probability path that linearly interpolates 
    btwn a base distribution and an unnormalized target distribution."""
    return (1 - time) * base_logdensity_fn(x) + time * target_logdensity_fn(x)


def divergence(fn: Callable[[Array], Array]) -> Callable[[Array], Scalar]:
    """Compute the divergence of a vector field for calleable fn: R^d -> R^d.

    This implementation inefficiently computes the entire Jacobian in O(n*(n+1)/2) even
    though only diagonal in O(n) is needed.
    
    Code from: https://github.com/jax-ml/jax/issues/3022#issuecomment-2100553108.
    """
    return lambda x: jnp.trace(jax.jacobian(fn)(x))


class Velocity(eqx.Module):
    """Wrapper class for eqx.nn.MLP to allow input of two arguments instead of one. 
    
    This velocity MLP operates on both location x and time t. Instead of concatenating
    them as a single input, we pass them as separate arguments to the MLP. This is 
    helpful for (1) better semantics and (2) easier to differentiate with respect to 
    location x and time t separately.
    """
    mlp: eqx.nn.MLP
    
    def __init__(
        self, 
        in_size: int, 
        out_size: int, 
        width_size: int = 256, # previously 256
        depth: int = 8, # previously 8
        *,
        key: Key
    ):
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            key=key,
        )
        
    def __call__(self, x: Array, time: Scalar) -> Array:
        # TODO: confirm expand_dims along axis=-1 (instead of axis=0)
        expanded_time = jnp.expand_dims(time, axis=-1) # ensures vmap-compatible concatenation
        inputs = jnp.concatenate([x, expanded_time], axis=-1)
        return self.mlp(inputs)


class LFISState(NamedTuple):
    params: PyTree
    opt_state: OptState


class LFISInfo(NamedTuple):
    loss: Array
    x_t: Array


def init(params: PyTree, optimizer: GradientTransformation) -> LFISState:
    opt_state = optimizer.init(params)
    return LFISState(params=params, opt_state=opt_state)


def continuity_error(
    params: PyTree,
    static: PyTree,
    x_t: Array,
    time: Scalar,
) -> Array:
    """Compute error in continuity equation at time t for multiple points x_t ~ p_t."""
    velocity = eqx.combine(params, static)

    def vmap_me_plz(x_t: Array, time: Scalar) -> tuple:
        """Clean way to batch multiple computations over `num_samples`."""
        vel = velocity(x_t, time) # shape (dim,)
        div = divergence(lambda x: velocity(x, time))(x_t)  # shape ()
        score = jax.grad(probability_path_logdensity_fn, argnums=0)(x_t, time)  # shape (dim,)
        time_partial = jax.grad(probability_path_logdensity_fn, argnums=1)(x_t, time)  # shape ()
        return vel, div, score, time_partial

    vel, div, score, time_partial = jax.vmap(vmap_me_plz, in_axes=(0, None))(x_t, time)
    partial_t_Z = jnp.mean(time_partial, axis=0) # shape (num_samples,)
    return div + jnp.vecdot(vel, score) + time_partial - partial_t_Z # shape (num_samples,)


@eqx.filter_jit
def step(
    rng_key: Key,
    state: LFISState,
    time: Scalar,
    optimizer: GradientTransformation,
    static: PyTree,
    num_samples: int = 1,
) -> tuple[LFISState, LFISInfo]:
    x_t = sample(
        rng_key, time, state.params, static, base_sample_fn, num_samples
    )  # shape (num_samples, dim), shape (num_samples,)
    
    def continuity_error_loss_fn(params: PyTree) -> Scalar:
        """Mean squared error in continuity equation at time t."""
        eps = continuity_error(params, static, x_t, time)
        return jnp.mean(eps ** 2, axis=0)

    loss, grads = jax.value_and_grad(continuity_error_loss_fn)(state.params)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return LFISState(params=new_params, opt_state=new_opt_state), LFISInfo(loss=loss, x_t=x_t)


def sample(
    rng_key: Key,
    time: Scalar,
    params: PyTree,
    static: PyTree,
    base_sample_fn: Callable,
    num_samples: int = 1,
    delta_t: Scalar = 0.005,
) -> Array:
    x_0 = base_sample_fn(rng_key, num_samples) # shape (num_samples, dim)
    velocity = eqx.combine(params, static)
    euler_step = lambda x_t, time: x_t + delta_t * velocity(x_t, time)
    vmap_euler_step = jax.vmap(euler_step, in_axes=(0, None))
    
    def body_fn(time, carry):
        x_t = carry
        x_t = vmap_euler_step(x_t, time)
        return x_t
    
    x_t = jax.lax.fori_loop(
        lower=1,
        upper=1 + jnp.array(time / delta_t, dtype=int),
        body_fun=lambda time_idx, carry: body_fn(time_idx / num_time_steps, carry),
        init_val=x_0,
    )
    return x_t


def main(seed: int, num_time_steps: int, num_train_steps: int, num_samples: int):
    rng_key = jr.key(seed)
    rng_key, nn_key = jr.split(rng_key, 2)
    
    velocity = Velocity(in_size=dim + 1, out_size=dim, key=nn_key)
    params, static = eqx.partition(velocity, eqx.is_inexact_array)
    optimizer = optax.adam(1e-3)
    state = init(params, optimizer)

    for i in range(num_time_steps):
        time_key = jr.fold_in(rng_key, i)
        time = jr.uniform(time_key) # shape ()
        # time = jnp.array((i + 1) / num_time_steps) # shape ()
        print(f"\nIter {i} at time: {time}")

        for j in range(num_train_steps):
            step_key = jr.fold_in(time_key, j)
            state, info = step(
                step_key,
                state,
                time,
                optimizer,
                static,
                num_samples,
            )
            if j % 25 == 0:
                print(f"Step {j} Loss: {info.loss}")

    # generate approximate samples and plot them
    rng_key, sub_key = jr.split(rng_key)
    velocity = eqx.combine(state.params, static)
    approx_samples = sample(sub_key, 1, state.params, static, base_sample_fn, num_samples)
    sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], alpha=0.5)
    plt.savefig('hist.png', dpi=300)


if __name__ == "__main__":
    seed = 12345
    num_time_steps = 200
    num_train_steps = 500
    num_samples = 1000
    
    main(seed, num_time_steps, num_train_steps, num_samples)