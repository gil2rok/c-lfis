from functools import partial
from typing import Callable, NamedTuple

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree, Scalar
from optax import GradientTransformation, OptState

from utils import continuity_error, euler_step


class LFISAlgorithm(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable
    

class LFISState(NamedTuple):
    params: PyTree
    opt_state: OptState


class LFISInfo(NamedTuple):
    loss: Array
    time: Array
    x_t: Array


def init(params: PyTree, optimizer: GradientTransformation) -> LFISState:
    opt_state = optimizer.init(params)
    return LFISState(params=params, opt_state=opt_state)


@eqx.filter_jit
def step(
    rng_key: Key,
    state: LFISState,
    optimizer: GradientTransformation,
    static: PyTree,
    base_sample_fn: Callable,
    probability_path_logdensity_fn: Callable,
    num_samples: int = 1,
    num_unique_time: int = 100,
) -> tuple[LFISState, LFISInfo]:
    time_key, sample_key = jr.split(rng_key)
    unique_time = jr.beta(time_key, 4, 4, shape=(num_unique_time,)) # shape (num_unique_time,)
    time = jnp.repeat(unique_time, num_samples // num_unique_time) # shape (num_samples,)
    x_t = jax.vmap(sample, in_axes=(0, 0, None, None, None))(
        jr.split(sample_key, num_samples),
        time,
        state.params,
        static,
        base_sample_fn,
    )  # shape (num_samples, dim)
        
    def continuity_error_loss_fn(params: PyTree) -> Scalar:
        """Mean squared error in continuity equation at time t."""
        eps = continuity_error(
            params, static, x_t, time, probability_path_logdensity_fn
        )
        return jnp.mean(eps ** 2, axis=0)

    loss, grads = jax.value_and_grad(continuity_error_loss_fn)(state.params)
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_params = optax.incremental_update(
        new_tensors = optax.apply_updates(state.params, updates),
        old_tensors = state.params,
        step_size = 0.999
    ) # exponential moving average btwn prev params and new params from optimizer update
    new_state = LFISState(params=new_params, opt_state=new_opt_state)
    new_info = LFISInfo(loss=loss, time=time, x_t=x_t)
    return new_state, new_info


def sample(
    rng_key: Key,
    time: Array,
    params: PyTree,
    static: PyTree,
    base_sample_fn: Callable,
    delta_t: Scalar = 0.005,
) -> Array:
    """Evolve a single sample from x_0 ~ p_0 to x_t ~ p_t using neural velocity field."""
    x_0 = base_sample_fn(rng_key) # shape (dim,)
    velocity = eqx.combine(params, static)
  
    def body_fn(time: Scalar, carry: Array, delta_t: float) -> Array:
        """Apply numerical integration step from `time` to `time + delta_t`."""
        x_t = carry
        x_t = euler_step(velocity, x_t, time, delta_t)
        return x_t
    
    def body_fn_wrapper(time_idx: int, carry: Array) -> Array:
        """Convert time index to time and handle adaptive final integration step.
        
        Each step for numerical integration is of size delta_t, except for the last 
        one which is adaptively computed as `min(cur_time + delta_t, time)`.
        """
        cur_time = (time_idx - 1) * delta_t
        next_time = jnp.minimum(time_idx * delta_t, time)
        adaptive_delta_t = next_time - cur_time
        return body_fn(cur_time, carry, adaptive_delta_t)
        
    num_integration_steps = jnp.ceil(time / delta_t).astype(int)
    x_t = jax.lax.fori_loop(
        lower=1, 
        upper=1 + num_integration_steps,
        body_fun=body_fn_wrapper,
        init_val=x_0,
    )
    return x_t


def as_top_level_api(
    optimizer: GradientTransformation,
    static: PyTree,
    base_sample_fn: Callable,
    probability_path_logdensity_fn: Callable,
    num_samples: int = 1,
    num_unique_time: int = 100,
    delta_t: float = 0.005,
):
    def init_fn(params: PyTree) -> LFISState:
        return init(params, optimizer)
    
    def step_fn(rng_key: Key, state: LFISState) -> tuple[LFISState, LFISInfo]:
        return step(
            rng_key=rng_key, 
            state=state,
            optimizer=optimizer, 
            static=static, 
            base_sample_fn=base_sample_fn, 
            probability_path_logdensity_fn=probability_path_logdensity_fn, 
            num_samples=num_samples,
            num_unique_time=num_unique_time,
        )
    
    def sample_fn(rng_key: Key, time: Scalar, params: PyTree) -> Array:
        return sample(
            rng_key=rng_key,
            time=time,
            params=params,
            static=static,
            base_sample_fn=base_sample_fn,
            delta_t=delta_t,
        )
    
    return LFISAlgorithm(init_fn, step_fn, sample_fn)
