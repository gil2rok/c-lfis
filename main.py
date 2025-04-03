from typing import Callable, NamedTuple

import jax
import optax
import wandb
import seaborn as sns
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jaxtyping import Array, Key, PyTree, Scalar

from lfis import as_top_level_api
from utils import (
    continuity_error, compute_1D_wasserstein_distance, Velocity
)


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


def target_sample_fn(rng_key, num_samples=1):
    m1_key, m2_key, m3_key, m4_key, cat_key = jr.split(rng_key, 5)
    m1 = jr.multivariate_normal(m1_key, jnp.array([2.0, 2.0]), jnp.eye(2), shape=(num_samples,))
    m2 = jr.multivariate_normal(m2_key, jnp.array([-2.0, -2.0]), jnp.eye(2), shape=(num_samples,))
    m3 = jr.multivariate_normal(m3_key, jnp.array([2.0, -2.0]), jnp.eye(2), shape=(num_samples,))
    m4 = jr.multivariate_normal(m4_key, jnp.array([-2.0, 2.0]), jnp.eye(2), shape=(num_samples,))
    mixtures = jnp.stack([m1, m2, m3, m4], axis=1)
    weights = jnp.array([0.25, 0.25, 0.25, 0.25])
    categories = jr.categorical(cat_key, weights, shape=(num_samples,))
    one_hot_categories = jax.nn.one_hot(categories, num_classes=4)  # shape (num_samples, 4)
    one_hot_expanded = jnp.expand_dims(one_hot_categories, axis=-1)  # shape (num_samples, 4, 1)
    return jnp.sum(mixtures * one_hot_expanded, axis=1)  # shape (num_samples, 2)


def probability_path_logdensity_fn(x: Array, time: Scalar) -> Scalar:
    """Logdensity of time-dependent probability path that linearly interpolates 
    btwn a base distribution and an unnormalized target distribution."""
    return (1 - time) * base_logdensity_fn(x) + time * target_logdensity_fn(x)


class LFISMetrics(NamedTuple):
    true_wass: Array
    approx_wass: Array
    true_continuity_error: Array
    approx_continuity_error: Array
    

def compute_metrics(rng_key, lfis, static, LFISState, LFISInfo) -> LFISMetrics:
    target_key, target_key2, approx_key = jr.split(rng_key, 3)
    num_samples = LFISInfo.x_t.shape[0]
    
    target_samples = target_sample_fn(target_key, num_samples)
    target_samples2 = target_sample_fn(target_key2, num_samples)
    approx_samples = lfis.sample(approx_key, 1.0, LFISState.params)
    
    true_wass = compute_1D_wasserstein_distance(target_samples, target_samples2)
    approx_wass = compute_1D_wasserstein_distance(target_samples, approx_samples)
    
    true_continuity_error = jnp.mean(
        continuity_error(
            LFISState.params, static, target_samples, 1.0, probability_path_logdensity_fn
        ) ** 2
    )
    approx_continuity_error = jnp.mean(
        continuity_error(
            LFISState.params, static, approx_samples, 1.0, probability_path_logdensity_fn
        ) ** 2
    )
    
    figure = sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], alpha=0.5)
    
    wandb.log(
        {
            "true_wass": true_wass,
            "approx_wass": approx_wass,
            "true_continuity_error": true_continuity_error,
            "approx_continuity_error": approx_continuity_error,
            "approx_samples": wandb.Image(figure.get_figure()),
        }
    )
    return LFISMetrics(
        true_wass=true_wass,
        approx_wass=approx_wass,
        true_continuity_error=jnp.mean(true_continuity_error ** 2),
        approx_continuity_error=jnp.mean(approx_continuity_error ** 2),
    )


def main(
    seed: int, 
    num_time_steps: int, 
    num_train_steps: int, 
    num_samples: int, 
    delta_t: Scalar
):
    rng_key = jr.key(seed)
    rng_key, nn_key = jr.split(rng_key, 2)
    
    velocity = Velocity(in_size=dim + 1, out_size=dim, key=nn_key)
    params, static = eqx.partition(velocity, eqx.is_inexact_array)
    optimizer = optax.adam(1e-3)
    
    lfis = as_top_level_api(
        optimizer=optimizer,
        static=static,
        base_sample_fn=base_sample_fn,
        probability_path_logdensity_fn=probability_path_logdensity_fn,
        num_samples=num_samples,
        num_time_steps=num_time_steps,
        delta_t=delta_t,
    )
    state = lfis.init(params)

    for i in range(num_time_steps):
        time_key = jr.fold_in(rng_key, i)
        time = jr.uniform(time_key) # shape ()
        # time = jnp.array((i + 1) / num_time_steps) # shape ()
        # time = jnp.linspace(0.4, 0.6, num_time_steps)
        print(f"\nIter {i} at time: {time}")

        for j in range(num_train_steps):
            step_key = jr.fold_in(time_key, j)
            state, info = lfis.step(
                rng_key=step_key,
                state=state,
                time=time,
            )
            if j % 50 == 0:
                print(f"Step {j} Loss: {info.loss}")
        
        metrics_key = jr.fold_in(time_key, 1)
        metrics = compute_metrics(metrics_key, lfis, static, state, info)
        print(f"\tWass Dist: {metrics.approx_wass:.4f} ({metrics.true_wass:.4f})")
        print(f"\tContinuity Error: {metrics.approx_continuity_error:.4f} ({metrics.true_continuity_error:.4f})")

    # generate approximate samples and plot them
    rng_key, sub_key = jr.split(rng_key)
    velocity = eqx.combine(state.params, static)
    approx_samples = lfis.sample(sub_key, 1, state.params)
    sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], alpha=0.5)
    plt.savefig('hist.png', dpi=300)


if __name__ == "__main__":
    seed = 12345
    num_time_steps = 200
    num_train_steps = 500
    num_samples = 1000
    delta_t = 0.005
    
    wandb.init(
        project="lfis",
        config={
            "seed": seed,
            "num_time_steps": num_time_steps,
            "num_train_steps": num_train_steps,
            "num_samples": num_samples,
            "delta_t": delta_t,
        },
    )
    
    main(seed, num_time_steps, num_train_steps, num_samples, delta_t)