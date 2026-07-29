import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp
import jax.random as jr

from affine.backends import jax as backend


def gaussian_logp(theta):
    return -0.5 * jnp.sum(theta ** 2, axis=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0):
    k1, k2 = jr.split(jr.PRNGKey(seed))
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return 0.5 * jr.normal(k1, shape), 0.5 * jr.normal(k2, shape)


def test_sample_shapes():
    state = make_state(n_walkers=16, n_params=3)
    chain = backend.sample(gaussian_logp, 10, state, rng=jr.PRNGKey(0), progressbar=False)
    assert chain.shape == (10, 32, 3)

    chain, logp = backend.sample(gaussian_logp, 10, state, rng=jr.PRNGKey(0),
                                 progressbar=False, return_logp=True)
    assert chain.shape == (10, 32, 3)
    assert logp.shape == (10, 32)


def test_rng_is_required():
    state = make_state(n_walkers=8)
    with pytest.raises(ValueError, match="PRNG key"):
        backend.sample(gaussian_logp, 10, state, progressbar=False)


def test_burnin_and_thin_shapes():
    state = make_state(n_walkers=8)
    # stored steps: 3, 5, 7, 9 -> 4 entries
    chain = backend.sample(gaussian_logp, 10, state, rng=jr.PRNGKey(0),
                           n_burnin=3, thin=2, progressbar=False)
    assert chain.shape[0] == 4


def test_recovers_gaussian_moments():
    state = make_state(n_walkers=64)
    chain = backend.sample(gaussian_logp, 600, state, rng=jr.PRNGKey(1), progressbar=False)
    samples = np.asarray(chain[300:]).reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    def half_gaussian_logp(theta):
        logp = -0.5 * jnp.sum(theta ** 2, axis=-1)
        return jnp.where(theta[..., 0] < 0.0, jnp.nan, logp)

    w = 0.5 + jr.uniform(jr.PRNGKey(3), (32, 2))
    state = (w, w + 0.1)
    chain = backend.sample(half_gaussian_logp, 200, state, rng=jr.PRNGKey(2),
                           progressbar=False)
    samples = np.asarray(chain)
    assert np.all(np.isfinite(samples))
    assert np.all(samples[..., 0] >= 0.0)


def test_reproducible_with_key():
    state = make_state()
    chain1 = backend.sample(gaussian_logp, 50, state, rng=jr.PRNGKey(42), progressbar=False)
    chain2 = backend.sample(gaussian_logp, 50, state, rng=jr.PRNGKey(42), progressbar=False)
    assert np.array_equal(np.asarray(chain1), np.asarray(chain2))


def test_sample_batch_recovers_batch_means():
    means = jnp.array([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * jnp.sum((theta - means[None, :, None]) ** 2, axis=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = backend.sample_batch(batch_logp, 600, state, rng=jr.PRNGKey(4),
                                 progressbar=False)
    assert chain.shape == (600, 128, 3, 2)
    est_means = np.asarray(chain[300:]).mean(axis=(0, 1, 3))
    assert np.all(np.abs(est_means - np.asarray(means)) < 0.15)


def test_legacy_api():
    from affine import sample

    state = list(make_state(n_walkers=16, n_params=3))
    # jax-branch signature: sample(rng, log_prob, n_steps, current_state)
    chain = sample(jr.PRNGKey(0), gaussian_logp, 10, state, progressbar=False)
    assert chain.shape == (10, 32, 3)

    # keyword style should dispatch to jax too
    chain = sample(rng=jr.PRNGKey(0), log_prob=gaussian_logp, n_steps=10,
                   current_state=state, progressbar=False)
    assert chain.shape == (10, 32, 3)
