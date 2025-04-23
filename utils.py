from typing import Callable

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Key, PyTree, Scalar


def divergence(f: Callable) -> Callable:
    """Compute the divergence of a vector field for callable function f: R^d -> R^d.

    This implementation inefficiently computes the entire Jacobian in O(n*(n+1)/2) even
    though only diagonal in O(n) is needed.
    
    Code from: https://github.com/jax-ml/jax/issues/3022#issuecomment-2100553108.
    """
    return lambda x: jnp.trace(jax.jacobian(f)(x))


def compute_1D_wasserstein_distance(X, Y):
    """Efficiently compute 1D Wasserstein distance between two point clouds."""
    return jnp.mean(jnp.abs(jnp.sort(X, axis=0) - jnp.sort(Y, axis=0)))


def euler_step(f, x, time, delta_t):
    """Perform a single Euler step for the ODE dx/dt = f(x, t)."""
    return x + delta_t * f(x, time)


def runge_kutta_step(f, x, time, delta_t):
    """Perform a single Runge-Kutta step for the ODE dx/dt = f(x, t)."""
    k1 = f(x, time)
    k2 = f(x + 0.5 * delta_t * k1, time + 0.5 * delta_t)
    k3 = f(x + 0.5 * delta_t * k2, time + 0.5 * delta_t)
    k4 = f(x + delta_t * k3, time + delta_t)
    return x + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def continuity_error(
    params: PyTree,
    static: PyTree,
    x_t: Array,
    time: Scalar,
    probability_path_logdensity_fn: Callable
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

