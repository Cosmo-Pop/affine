import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from affine.tensorflow import affine_sample, affine_sample_batch


def gaussian_logp(theta):
    return -0.5 * tf.reduce_sum(theta ** 2, axis=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return (tf.constant(rng.normal(0.0, 0.5, shape).astype(dtype)),
            tf.constant(rng.normal(0.0, 0.5, shape).astype(dtype)))


def test_sample_shapes():
    state = make_state(n_walkers=16, n_params=3)
    chain = affine_sample(gaussian_logp, 10, state, progressbar=False)
    assert chain.shape == (10, 32, 3)


def test_chain_starts_at_initial_state():
    state = make_state(n_walkers=8)
    chain = affine_sample(gaussian_logp, 5, state, progressbar=False)
    assert np.array_equal(chain[0].numpy(), tf.concat(state, axis=0).numpy())


def test_recovers_gaussian_moments():
    tf.random.set_seed(1)
    state = make_state(n_walkers=64)
    chain = affine_sample(gaussian_logp, 600, state, progressbar=False)
    samples = chain[300:].numpy().reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    # NaN outside the domain must behave like -inf: no sample ever lands there
    def half_gaussian_logp(theta):
        logp = -0.5 * tf.reduce_sum(theta ** 2, axis=-1)
        return tf.where(theta[..., 0] < 0.0, np.nan * tf.ones_like(logp), logp)

    tf.random.set_seed(2)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.5, 1.5, (32, 2)).astype(np.float32)
    state = (tf.constant(w), tf.constant(w + 0.1))
    chain = affine_sample(half_gaussian_logp, 200, state, progressbar=False)
    samples = chain.numpy()
    assert np.all(np.isfinite(samples))
    assert np.all(samples[..., 0] >= 0.0)


def test_reproducible_with_global_seed():
    state = make_state()
    tf.random.set_seed(42)
    chain1 = affine_sample(gaussian_logp, 50, state, progressbar=False)
    tf.random.set_seed(42)
    chain2 = affine_sample(gaussian_logp, 50, state, progressbar=False)
    assert np.array_equal(chain1.numpy(), chain2.numpy())


def test_float64_supported():
    state = make_state(dtype=np.float64)
    chain = affine_sample(gaussian_logp, 20, state, progressbar=False)
    assert chain.dtype == tf.float64


def test_sample_batch_recovers_batch_means():
    tf.random.set_seed(4)
    means = tf.constant([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * tf.reduce_sum((theta - means[None, :, None]) ** 2, axis=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = affine_sample_batch(batch_logp, 600, state, progressbar=False)
    assert chain.shape == (600, 128, 3, 2)
    est_means = chain[300:].numpy().mean(axis=(0, 1, 3))
    assert np.all(np.abs(est_means - means.numpy()) < 0.15)


def test_args_passthrough():
    def shifted_logp(theta, mu):
        return -0.5 * tf.reduce_sum((theta - mu) ** 2, axis=-1)

    tf.random.set_seed(6)
    state = make_state(n_walkers=64)
    chain = affine_sample(shifted_logp, 400, state, args=[3.0], progressbar=False)
    samples = chain[200:].numpy().reshape(-1, 2)
    assert np.all(np.abs(samples.mean(axis=0) - 3.0) < 0.15)


def test_burnin_thin_and_save_lp():
    state = make_state(n_walkers=16)
    # stored steps: 3, 5, 7, 9 -> 4 entries
    chain = affine_sample(gaussian_logp, 10, state, progressbar=False, n_burnin=3, thin=2)
    assert chain.shape == (4, 32, 2)

    chain, lp = affine_sample(gaussian_logp, 10, state, progressbar=False, save_lp=True)
    assert lp.shape == (10, 32)
    # saved log probs match log_prob evaluated at the stored samples
    assert np.allclose(lp.numpy(), gaussian_logp(chain).numpy(), atol=1e-5)

    with pytest.raises(ValueError):
        affine_sample(gaussian_logp, 10, state, progressbar=False, n_burnin=10)


def test_save_extras():
    # extras returned by log_prob are stored per sample; they must equal the
    # same quantities recomputed from the stored chain (accept/reject tracking)
    def logp_with_extras(theta):
        logp = -0.5 * tf.reduce_sum(theta ** 2, axis=-1)
        extras = tf.stack([tf.reduce_sum(theta, axis=-1),
                           tf.reduce_sum(theta ** 2, axis=-1)], axis=-1)
        return logp, extras

    state = make_state(n_walkers=16)
    chain, extras = affine_sample(logp_with_extras, 20, state, progressbar=False,
                                  save_extras=True)
    assert extras.shape == (20, 32, 2)
    expected = tf.stack([tf.reduce_sum(chain, axis=-1),
                         tf.reduce_sum(chain ** 2, axis=-1)], axis=-1)
    assert np.allclose(extras.numpy(), expected.numpy(), atol=1e-5)

    # combined with save_lp the return order is (chain, lp, extras)
    chain, lp, extras = affine_sample(logp_with_extras, 10, state, progressbar=False,
                                      save_lp=True, save_extras=True)
    assert lp.shape == (10, 32)
    assert extras.shape == (10, 32, 2)

    # batched variant
    state = make_state(n_walkers=16, n_batch=3)
    chain, extras = affine_sample_batch(logp_with_extras, 20, state, progressbar=False,
                                        save_extras=True)
    assert extras.shape == (20, 32, 3, 2)
    expected = tf.stack([tf.reduce_sum(chain, axis=-1),
                         tf.reduce_sum(chain ** 2, axis=-1)], axis=-1)
    assert np.allclose(extras.numpy(), expected.numpy(), atol=1e-5)


def test_legacy_top_level_import():
    from affine import affine_sample as legacy_sample
    from affine import affine_sample_batch as legacy_batch

    state = list(make_state(n_walkers=16))
    chain = legacy_sample(gaussian_logp, 10, state, args=[], progressbar=False)
    assert chain.shape == (10, 32, 2)

    state = list(make_state(n_walkers=16, n_batch=4))
    chain = legacy_batch(gaussian_logp, 10, state, args=[], progressbar=False)
    assert chain.shape == (10, 32, 4, 2)
