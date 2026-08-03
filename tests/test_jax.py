import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp
import jax.random as jr

from affine.jax import sample, sample_batch


def gaussian_logp(theta):
    return -0.5 * jnp.sum(theta ** 2, axis=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0):
    k1, k2 = jr.split(jr.PRNGKey(seed))
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return 0.5 * jr.normal(k1, shape), 0.5 * jr.normal(k2, shape)


def test_sample_shapes():
    state = make_state(n_walkers=16, n_params=3)
    chain = sample(jr.PRNGKey(0), gaussian_logp, 10, state, progressbar=False)
    assert chain.shape == (10, 32, 3)


def test_recovers_gaussian_moments():
    state = make_state(n_walkers=64)
    chain = sample(jr.PRNGKey(1), gaussian_logp, 600, state, progressbar=False)
    samples = np.asarray(chain[300:]).reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    def half_gaussian_logp(theta):
        logp = -0.5 * jnp.sum(theta ** 2, axis=-1)
        return jnp.where(theta[..., 0] < 0.0, jnp.nan, logp)

    w = 0.5 + jr.uniform(jr.PRNGKey(3), (32, 2))
    chain = sample(jr.PRNGKey(2), half_gaussian_logp, 200, (w, w + 0.1), progressbar=False)
    samples = np.asarray(chain)
    assert np.all(np.isfinite(samples))
    assert np.all(samples[..., 0] >= 0.0)


def test_reproducible_with_key():
    state = make_state()
    chain1 = sample(jr.PRNGKey(42), gaussian_logp, 50, state, progressbar=False)
    chain2 = sample(jr.PRNGKey(42), gaussian_logp, 50, state, progressbar=False)
    assert np.array_equal(np.asarray(chain1), np.asarray(chain2))


def test_sample_batch_recovers_batch_means():
    means = jnp.array([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * jnp.sum((theta - means[None, :, None]) ** 2, axis=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = sample_batch(jr.PRNGKey(4), batch_logp, 600, state, progressbar=False)
    assert chain.shape == (600, 128, 3, 2)
    est_means = np.asarray(chain[300:]).mean(axis=(0, 1, 3))
    assert np.all(np.abs(est_means - np.asarray(means)) < 0.15)


def test_args_passthrough():
    def shifted_logp(theta, mu):
        return -0.5 * jnp.sum((theta - mu) ** 2, axis=-1)

    state = make_state(n_walkers=64)
    chain = sample(jr.PRNGKey(6), shifted_logp, 400, state, progressbar=False, args=(3.0,))
    samples = np.asarray(chain[200:]).reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0) - 3.0) < 0.15)


def test_legacy_top_level_import():
    # the jax-branch call style dispatches on the (non-callable) first argument
    from affine import sample as legacy_sample

    state = list(make_state(n_walkers=16, n_params=3))
    chain = legacy_sample(jr.PRNGKey(0), gaussian_logp, 10, state, progressbar=False)
    assert chain.shape == (10, 32, 3)
