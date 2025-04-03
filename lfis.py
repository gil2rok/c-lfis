from typing import Callable, NamedTuple

import jax
import optax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree, Scalar
from optax import GradientTransformation, OptState

from utils import continuity_error


class LFISAlgorithm(NamedTuple):
    init: Callable
    step: Callable
    sample: Callable
    

class LFISState(NamedTuple):
    params: PyTree
    opt_state: OptState


class LFISInfo(NamedTuple):
    loss: Array
    x_t: Array


def init(params: PyTree, optimizer: GradientTransformation) -> LFISState:
    opt_state = optimizer.init(params)
    return LFISState(params=params, opt_state=opt_state)


@eqx.filter_jit
def step(
    rng_key: Key,
    state: LFISState,
    time: Scalar,
    optimizer: GradientTransformation,
    static: PyTree,
    base_sample_fn: Callable,
    probability_path_logdensity_fn: Callable,
    num_samples: int = 1,
    num_time_steps: int = 200,
    delta_t: Scalar = 0.005,
) -> tuple[LFISState, LFISInfo]:
    x_t = sample(
        rng_key=rng_key,
        time=time,
        params=state.params, 
        static=static, 
        base_sample_fn=base_sample_fn, 
        num_samples=num_samples, 
        delta_t=delta_t
    )  # shape (num_samples, dim)
    
    def continuity_error_loss_fn(params: PyTree) -> Scalar:
        """Mean squared error in continuity equation at time t."""
        eps = continuity_error(
            params, static, x_t, time, probability_path_logdensity_fn
        )
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
    euler_step = lambda x_t, time, delta_t: x_t + delta_t * velocity(x_t, time)
    vmap_euler_step = jax.vmap(euler_step, in_axes=(0, None, None))
    
    def body_fn(time, carry, delta_t):
        x_t = carry
        x_t = vmap_euler_step(x_t, time, delta_t)
        return x_t
    
    def body_fn_wrapper(time_idx, carry):
        """Convert time index to time and handle adaptive final integration step."""
        cur_time = jnp.minimum(time_idx * delta_t, time)
        prev_time = (time_idx - 1) * delta_t
        actual_delta_t = cur_time - prev_time
        return body_fn(cur_time, carry, actual_delta_t)
        
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
    num_time_steps: int = 200,
    delta_t: Scalar = 0.005,
):
    def init_fn(params: PyTree) -> LFISState:
        return init(params, optimizer)
    
    def step_fn(
        rng_key: Key, state: LFISState, time: Scalar
    ) -> tuple[LFISState, LFISInfo]:
        return step(
            rng_key=rng_key, 
            state=state, 
            time=time, 
            optimizer=optimizer, 
            static=static, 
            base_sample_fn=base_sample_fn, 
            probability_path_logdensity_fn=probability_path_logdensity_fn, 
            num_samples=num_samples, 
            num_time_steps=num_time_steps,
            delta_t=delta_t,
        )
    
    def sample_fn(rng_key: Key, time: Scalar, params: PyTree) -> Array:
        return sample(
            rng_key=rng_key,
            time=time,
            params=params,
            static=static,
            base_sample_fn=base_sample_fn,
            num_samples=num_samples,
            delta_t=delta_t,
        )
    
    return LFISAlgorithm(init_fn, step_fn, sample_fn)
