from typing import Callable, NamedTuple

import jax
import optax
import seaborn as sns
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jaxtyping import Array, Key, PyTree, Float
from optax import GradientTransformation, OptState

seed = 12345
rng_key = jr.PRNGKey(seed)
dim = 2


def base_logdensity_fn(x: Array) -> Float:
    return jsp.stats.norm.logpdf(x, 0, 1).sum(axis=-1)


def base_sample_fn(rng_key: Key, num_samples: int = 1) -> Array:
    return jr.normal(rng_key, (num_samples, dim))


def target_logdensity_fn(x: Array) -> Float:
    # mixture of four Gaussian in 2D
    assert x.shape[-1] == dim
    g1 = jsp.stats.norm.logpdf(x, jnp.array([2, 2]), 0.5)
    g2 = jsp.stats.norm.logpdf(x, jnp.array([-2, -2]), 0.5)
    g3 = jsp.stats.norm.logpdf(x, jnp.array([2, -2]), 0.5)
    g4 = jsp.stats.norm.logpdf(x, jnp.array([-2, 2]), 0.5)
    # TODO: confirm correct weighting of logpdfs
    return jnp.log(0.25 * (jnp.exp(g1) + jnp.exp(g2) + jnp.exp(g3) + jnp.exp(g4))).sum(
        axis=-1
    )


def probability_path_logdensity_fn(x: Array, time: Float) -> Float:
    """Logdensity of time-dependent probability path that linearly interpolates 
    btwn a base distribution and an unnormalized target distribution."""
    return (1 - time) * base_logdensity_fn(x) + time * target_logdensity_fn(x)


def divergence(fn: Callable[[Array], Array]) -> Callable[[Array], Float]:
    """Compute the divergence of a vector field for calleable fn: R^d -> R^d.

    Code from: https://github.com/jax-ml/jax/issues/3022#issuecomment-2100553108.
    """
    return lambda x: jnp.trace(jax.jacobian(fn)(x))


class LFISState(NamedTuple):
    params: PyTree
    opt_state: OptState


class LFISInfo(NamedTuple):
    loss: Float


def init(params: PyTree, optimizer: GradientTransformation):
    opt_state = optimizer.init(params)
    return LFISState(params=params, opt_state=opt_state)


def step(
    rng_key: Key,
    state: LFISState,
    time: Float,
    probability_path_logdensity_fn: Callable,
    optimizer: GradientTransformation,
    static: PyTree,
    num_samples: int = 1,
):

    def continuity_error(params):
        """Compute mean of squared continuity error."""
        # TODO: compute partial_t_Z once per time step instead of every train step
        # requires computing partial_t_Z in first loop of main() and passing it as an arg
        # TODO: put sample code (and not-yet-implemented importance reweighting) outside of 
        # continuity error function; change signature to `continuity_error(params, x_t, time)`
        velocity = eqx.combine(params, static)
        x_t = sample(rng_key, time, velocity, base_sample_fn, num_samples)  # (num_samples, dim)

        def vmap_me_plz(x_t, time):
          """Clean way to add batch dimension (of size `num_samples`) for multiple computations."""
          vel = velocity(x_t) # (dim)
          div = divergence(velocity)(x_t)  # (1,)
          score = jax.grad(probability_path_logdensity_fn, argnums=0)(x_t, time)  # (dim)
          time_partial = jax.grad(probability_path_logdensity_fn, argnums=1)(x_t, time)  # (1,)
          return vel, div, score, time_partial

        vel, div, score, time_partial = jax.vmap(vmap_me_plz)(x_t, time)
        partial_t_Z = jnp.mean(time_partial, axis=0) # (1,)
        eps = div + jnp.vecdot(vel, score) + time_partial - partial_t_Z
        return jnp.mean(eps**2, axis=0)

    loss, grads = jax.value_and_grad(continuity_error)(state.params)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return LFISState(params=new_params, opt_state=new_opt_state), LFISInfo(loss=loss)


def sample(
    rng_key: Key,
    time: Float,
    velocity: Callable,
    base_sample_fn: Callable,
    num_samples: int = 1,
):

    x_0 = base_sample_fn(rng_key, num_samples)
    time = jnp.expand_dims(time, axis=-1)
    x_t = x_0 + time * jax.vmap(velocity)(x_0)
    return x_t


def main():
    num_time_steps = 1
    num_train_steps = 500
    num_samples = 2000

    rng_key, nn_key = jr.split(rng_key, 2)

    velocity = eqx.nn.MLP(
        in_size=dim,
        out_size=dim,
        width_size=64,
        depth=2,
        key=nn_key,
    )

    params, static = eqx.partition(velocity, eqx.is_inexact_array)
    optimizer = optax.adam(5e-4)
    state = init(params, optimizer)

    for i in range(num_time_steps):
        time_key = jr.fold_in(rng_key, i)
        time = jr.uniform(time_key, (num_samples))

        for j in range(num_train_steps):
            step_key = jr.fold_in(time_key, j)
            state, info = step(
                step_key,
                state,
                time,
                probability_path_logdensity_fn,
                optimizer,
                static,
                num_samples,
            )
            if j % 25 == 0:
                print(f"Step {j} Loss: {info.loss}")

    # generate approximate samples and plot them
    rng_key, sub_key = jr.split(rng_key)
    velocity = eqx.combine(state.params, static)
    approx_samples = sample(sub_key, 1, velocity, base_sample_fn, num_samples)

    sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], alpha=0.5)


if __name__ == "__main__":
    main()
