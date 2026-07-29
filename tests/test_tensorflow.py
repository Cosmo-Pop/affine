import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from affine.backends import tensorflow as backend


def gaussian_logp(theta):
    return -0.5 * tf.reduce_sum(theta ** 2, axis=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return (tf.constant(rng.normal(0.0, 0.5, shape).astype(dtype)),
            tf.constant(rng.normal(0.0, 0.5, shape).astype(dtype)))


def test_sample_shapes():
    state = make_state(n_walkers=16, n_params=3)
    chain = backend.sample(gaussian_logp, 10, state, rng=0, progressbar=False)
    assert chain.shape == (10, 32, 3)

    chain, logp = backend.sample(gaussian_logp, 10, state, rng=0, progressbar=False,
                                 return_logp=True)
    assert chain.shape == (10, 32, 3)
    assert logp.shape == (10, 32)


def test_burnin_and_thin_shapes():
    state = make_state(n_walkers=8)
    # stored steps: 3, 5, 7, 9 -> 4 entries
    chain = backend.sample(gaussian_logp, 10, state, rng=0, n_burnin=3, thin=2,
                           progressbar=False)
    assert chain.shape[0] == 4


def test_recovers_gaussian_moments():
    state = make_state(n_walkers=64)
    chain = backend.sample(gaussian_logp, 600, state, rng=1, progressbar=False)
    samples = chain[300:].numpy().reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    # NaN outside the domain must behave like -inf: no sample ever lands there
    def half_gaussian_logp(theta):
        logp = -0.5 * tf.reduce_sum(theta ** 2, axis=-1)
        return tf.where(theta[..., 0] < 0.0, np.nan * tf.ones_like(logp), logp)

    rng = np.random.default_rng(3)
    w = rng.uniform(0.5, 1.5, (32, 2)).astype(np.float32)
    state = (tf.constant(w), tf.constant(w + 0.1))
    chain = backend.sample(half_gaussian_logp, 200, state, rng=2, progressbar=False)
    samples = chain.numpy()
    assert np.all(np.isfinite(samples))
    assert np.all(samples[..., 0] >= 0.0)


def test_reproducible_with_seed():
    state = make_state()
    chain1 = backend.sample(gaussian_logp, 50, state, rng=42, progressbar=False)
    chain2 = backend.sample(gaussian_logp, 50, state, rng=42, progressbar=False)
    assert np.array_equal(chain1.numpy(), chain2.numpy())


def test_float64_supported():
    state = make_state(dtype=np.float64)

    def logp64(theta):
        return -0.5 * tf.reduce_sum(theta ** 2, axis=-1)

    chain = backend.sample(logp64, 20, state, rng=0, progressbar=False)
    assert chain.dtype == tf.float64


def test_sample_batch_recovers_batch_means():
    means = tf.constant([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * tf.reduce_sum((theta - means[None, :, None]) ** 2, axis=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = backend.sample_batch(batch_logp, 600, state, rng=4, progressbar=False)
    assert chain.shape == (600, 128, 3, 2)
    samples = chain[300:].numpy()
    est_means = samples.mean(axis=(0, 1, 3))
    assert np.all(np.abs(est_means - means.numpy()) < 0.15)


def test_args_passthrough():
    def shifted_logp(theta, mu):
        return -0.5 * tf.reduce_sum((theta - mu) ** 2, axis=-1)

    state = make_state(n_walkers=64)
    chain = backend.sample(shifted_logp, 400, state, args=(3.0,), rng=6,
                           progressbar=False)
    samples = chain[200:].numpy().reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0) - 3.0) < 0.15)


def test_legacy_api():
    from affine import affine_sample, affine_sample_batch

    state = list(make_state(n_walkers=16))
    chain = affine_sample(gaussian_logp, 10, state, args=[], progressbar=False)
    assert chain.shape == (10, 32, 2)

    def batch_logp(theta):
        return -0.5 * tf.reduce_sum(theta ** 2, axis=-1)

    state = list(make_state(n_walkers=16, n_batch=4))
    chain = affine_sample_batch(batch_logp, 10, state, args=[], progressbar=False)
    assert chain.shape == (10, 32, 4, 2)


def test_legacy_module_path():
    from affine.affine import affine_sample  # pre-1.0 layout

    state = make_state(n_walkers=8)
    chain = affine_sample(gaussian_logp, 5, state, progressbar=False)
    assert chain.shape == (5, 16, 2)
