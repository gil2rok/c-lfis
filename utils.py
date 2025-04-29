from typing import Callable

import jax
import optax
import ott
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


@jax.jit
def marginal_wasserstein(X: Array, Y: Array) -> Array:
    """Compute marginal Wasserstein distance between X and Y in O(nlogn).
    
    See https://en.wikipedia.org/wiki/Wasserstein_metric#One_dimension.
    """
    return jnp.mean(jnp.abs(jnp.sort(X, axis=0) - jnp.sort(Y, axis=0)))


@jax.jit
def hungarian_wasserstein(X: Array, Y: Array) -> Array:
    """Compute exact Wasserstein distance between X and Y in O(n^3) time."""
    displacement = jnp.expand_dims(X, axis=1) - jnp.expand_dims(Y, axis=0)
    cost_matrix = jnp.linalg.norm(displacement, ord=2, axis=-1)
    i, j = optax.assignment.hungarian_algorithm(cost_matrix)
    return cost_matrix[i, j].sum()


@jax.jit
def sinkhorn_wasserstein(X: Array, Y: Array) -> Array:
    """Compute approximate Wasserstein distance between X and Y in O(n^2)."""
    geom = ott.geometry.pointcloud.PointCloud(X, Y, epsilon=1e-2)
    ot = ott.solvers.linear.solve(geom)
    return ot.primal_cost


def euler_step(f: Callable, x: Array, time: Array, delta_t: float) -> Array:
    """Perform a single Euler step for the ODE dx/dt = f(x, t)."""
    return x + delta_t * f(x, time)


def runge_kutta_step(f: Callable, x: Array, time: Array, delta_t: float) -> Array:
    """Perform a single Runge-Kutta step for the ODE dx/dt = f(x, t)."""
    k1 = f(x, time)
    k2 = f(x + 0.5 * delta_t * k1, time + 0.5 * delta_t)
    k3 = f(x + 0.5 * delta_t * k2, time + 0.5 * delta_t)
    k4 = f(x + delta_t * k3, time + delta_t)
    return x + (delta_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def continuity_error(
    params: PyTree,
    static: PyTree,
    x_t: Array, # shape (num_samples, dim)
    time: Array, # shape (num_samples,)
    probability_path_logdensity_fn: Callable
) -> Array:
    """Compute error in continuity equation at time t for multiple points x_t ~ p_t."""
    velocity = eqx.combine(params, static)

    def vmap_me_plz(x_t: Array, time: Array) -> tuple:
        """Clean way to batch multiple computations over `num_samples`."""
        vel = velocity(x_t, time) # shape (dim,)
        div = divergence(lambda x: velocity(x, time))(x_t)  # shape ()
        score = jax.grad(probability_path_logdensity_fn, argnums=0)(x_t, time)  # shape (dim,)
        log_time_partial = jax.grad(probability_path_logdensity_fn, argnums=1)(x_t, time)  # shape ()
        return vel, div, score, log_time_partial

    vel, div, score, log_time_partial = jax.vmap(vmap_me_plz, in_axes=(0, 0))(x_t, time)
    log_partial_t_Z = jnp.mean(log_time_partial, axis=0) # shape (num_samples,) # TODO: incorrect!
    return div + jnp.vecdot(vel, score) + log_time_partial - log_partial_t_Z # shape (num_samples,)


class Velocity(eqx.Module):
    """Neural velocity field with optional Fourier time encoding.
    
    This velocity MLP operates on both location x and time t. Instead of concatenating
    them as a single input, we pass them as separate arguments to the MLP. This is 
    helpful for (1) better semantics and (2) easier to differentiate with respect to 
    location x and time t separately.
    """
    network: eqx.nn.MLP
    embedding_dim: int
    max_freq: float
    encode_time: bool
    
    def __init__(
        self, 
        logdensity_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 8,
        embedding_dim: int = 10,
        max_freq: float = 10.0,
        encode_time: bool = True,
        *,
        key: Key
    ):
        time_dim = 2 * embedding_dim if encode_time else 1
        self.network = eqx.nn.MLP(
            in_size=logdensity_dim + time_dim,
            out_size=logdensity_dim,
            width_size=hidden_dim,
            depth=num_layers,
            key=key,
        )
        self.embedding_dim = embedding_dim
        self.max_freq = max_freq
        self.encode_time = encode_time
        
    def __call__(self, x: Array, time: Scalar) -> Array:
        """Compute velocity at position x and time t."""
        time_features = (
            self._encode_time(time) if self.encode_time else jnp.atleast_1d(time)
        )
        inputs = jnp.concatenate([x, time_features], axis=-1)
        return self.network(inputs)


    def _encode_time(self, time: float) -> Array:
        """Transform time to Fourier features for better expressivity."""
        freqs = jnp.geomspace(1.0, self.max_freq, self.embedding_dim)
        phases = jnp.outer(jnp.atleast_1d(time), freqs).squeeze() # shape (batch_size, num_freqs)
        return jnp.concatenate([jnp.sin(phases), jnp.cos(phases)], axis=-1) # shape (batch_size, num_freqs * 2,)