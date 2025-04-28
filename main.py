import jax
import optax
import wandb
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import seaborn as sns
from jaxtyping import Array, Key, Scalar

from lfis import as_top_level_api
from utils import marginal_wasserstein, sinkhorn_wasserstein, Velocity


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
dim = 2


def base_logdensity_fn(x: Array) -> Scalar:
    return jsp.stats.multivariate_normal.logpdf(x, jnp.zeros(dim), jnp.eye(dim))


def base_sample_fn(rng_key: Key, num_samples: int = 1) -> Array:
    return jr.multivariate_normal(
        rng_key, jnp.zeros(dim), jnp.eye(dim), (num_samples,)
    )


def target_logdensity_fn(x: Array) -> Scalar:
    scale_param, latent_params = jnp.split(x, [1], axis=-1)
    scale_logpdf = jsp.stats.norm.logpdf(scale_param, loc=0, scale=3) # shape (num_samples, 1)
    latent_logpdf = jsp.stats.norm.logpdf(
        latent_params, loc=0, scale=jnp.exp(scale_param / 2)
    ) # shape (num_samples, dim - 1)
    return scale_logpdf.squeeze() + jnp.sum(latent_logpdf, axis=-1) # shape ()


def target_sample_fn(rng_key: Key, num_samples: int = 1) -> Array:
    scale_key, latent_key = jr.split(rng_key)
    scale_param = jr.normal(scale_key, shape=(num_samples, 1)) * 3 # shape (num_samples, 1)
    latent_params = jr.normal(
        latent_key, shape=(num_samples, dim - 1)
    ) * jnp.exp(scale_param / 2) # shape (num_samples, dim - 1)
    return jnp.concatenate([scale_param, latent_params], axis=-1) # shape (num_samples, dim)


"""
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
    m1 = jr.multivariate_normal(m1_key, jnp.array([2.0, 2.0]), jnp.eye(2), shape=(num_samples,)) # shape (num_samples, dim)
    m2 = jr.multivariate_normal(m2_key, jnp.array([-2.0, -2.0]), jnp.eye(2), shape=(num_samples,))
    m3 = jr.multivariate_normal(m3_key, jnp.array([2.0, -2.0]), jnp.eye(2), shape=(num_samples,))
    m4 = jr.multivariate_normal(m4_key, jnp.array([-2.0, 2.0]), jnp.eye(2), shape=(num_samples,))
    mixtures = jnp.stack([m1, m2, m3, m4], axis=1) # shape (num_samples, num_mixtures, dim)
    weights = jnp.array([0.25, 0.25, 0.25, 0.25])
    categories = jr.categorical(cat_key, weights, shape=(num_samples,))
    one_hot_categories = jax.nn.one_hot(categories, num_classes=4)  # shape (num_samples, num_mixtures)
    one_hot_expanded = jnp.expand_dims(one_hot_categories, axis=-1)  # shape (num_samples, num_mixtures, 1)
    return jnp.sum(mixtures * one_hot_expanded, axis=1)  # shape (num_samples, dim)
"""


def probability_path_logdensity_fn(x: Array, time: Scalar) -> Scalar:
    """Logdensity of time-dependent probability path that linearly interpolates 
    btwn a base distribution and an unnormalized target distribution."""
    return (1 - time) * base_logdensity_fn(x) + time * target_logdensity_fn(x)


def compute_metrics(rng_key, idx, lfis, state, info):
    num_samples = info.x_t.shape[0]
    target_key, target_key2, approx_key = jr.split(rng_key, 3)
    target_samples = target_sample_fn(target_key, num_samples)
    target_samples2 = target_sample_fn(target_key2, num_samples)
    approx_samples = lfis.sample(approx_key, 1.0, state.params)
    true_wass = marginal_wasserstein(target_samples, target_samples2)
    approx_wass = marginal_wasserstein(target_samples, approx_samples)
    plt.clf()
    
    true_wass2 = sinkhorn_wasserstein(target_samples, target_samples2)
    approx_wass2 = sinkhorn_wasserstein(target_samples, approx_samples)
    
    figure = sns.scatterplot(x=approx_samples[:, 0], y=approx_samples[:, 1], alpha=0.5)
    wandb.log(
        step=idx,
        data={
            "Marginal_Wass_True": true_wass,
            "Marginal_Wass_Approx": approx_wass,
            "Sinkhorn_Wass_True": true_wass2,
            "Sinkhorn_Wass_Approx": approx_wass2,
            "approx_samples": wandb.Image(figure.get_figure()),
            "time": info.time,
            "loss": info.loss,
        }
    )


def main(
    seed: int, 
    num_train_steps: int, 
    num_samples: int, 
    learning_rate: float, 
    delta_t: float,
    encode_time: bool,
):
    rng_key = jr.key(seed)
    rng_key, nn_key = jr.split(rng_key, 2)
    velocity = Velocity(logdensity_dim=dim, key=nn_key, encode_time=encode_time)
    params, static = eqx.partition(velocity, eqx.is_inexact_array)
    optimizer = optax.adam(learning_rate)
    lfis = as_top_level_api(
        optimizer=optimizer,
        static=static,
        base_sample_fn=base_sample_fn,
        probability_path_logdensity_fn=probability_path_logdensity_fn,
        num_samples=num_samples,
        delta_t=delta_t,
    )
    state = lfis.init(params)

    for i in range(num_train_steps):
        time_key, metrics_key = jr.split(jr.fold_in(rng_key, i))
        state, info = lfis.step(time_key, state)
        if i % 10 == 0:
            compute_metrics(metrics_key, i, lfis, state, info)


if __name__ == "__main__":
    seed = 12345
    num_train_steps = 500
    num_samples = 1_000
    learning_rate = 1e-3
    delta_t = 0.005
    encode_time = False
    
    wandb.init(
        project="lfis",
        config={
            "seed": seed,
            "num_train_steps": num_train_steps,
            "num_samples": num_samples,
            "learning_rate": learning_rate,
            "delta_t": delta_t,
            "encode_time": encode_time,
        },
    )
    main(seed, num_train_steps, num_samples, learning_rate, delta_t, encode_time)