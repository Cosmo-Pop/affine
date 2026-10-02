import numpy as np
import pytest

torch = pytest.importorskip("torch")

from affine.torch import sample, sample_batch


def gaussian_logp(theta):
    return -0.5 * torch.sum(theta ** 2, dim=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return (0.5 * torch.randn(shape, generator=gen, dtype=dtype),
            0.5 * torch.randn(shape, generator=gen, dtype=dtype))


def test_sample_shapes():
    w1, w2 = make_state(n_walkers=16, n_params=3)
    chain = sample(gaussian_logp, 3, 16, 10, w1, w2, progress=False)
    assert chain.shape == (10, 32, 3)

    chain, lp = sample(gaussian_logp, 3, 16, 10, w1, w2, progress=False, save_lp=True)
    assert chain.shape == (10, 32, 3)
    assert lp.shape == (10, 32)


def test_wrong_walker_shape_raises():
    w1, w2 = make_state(n_walkers=16, n_params=3)
    with pytest.raises(ValueError):
        sample(gaussian_logp, 2, 16, 10, w1, w2, progress=False)


def test_recovers_gaussian_moments():
    torch.manual_seed(1)
    w1, w2 = make_state(n_walkers=64)
    chain = sample(gaussian_logp, 2, 64, 600, w1, w2, progress=False)
    samples = chain[300:].reshape(-1, 2).numpy()
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    # regression test: the old torch branch mapped NaN log-probs to +inf,
    # accepting every proposal in the invalid region
    def half_gaussian_logp(theta):
        logp = -0.5 * torch.sum(theta ** 2, dim=-1)
        return torch.where(theta[..., 0] < 0.0, torch.full_like(logp, float("nan")), logp)

    torch.manual_seed(2)
    gen = torch.Generator().manual_seed(3)
    w = 0.5 + torch.rand((32, 2), generator=gen)
    chain = sample(half_gaussian_logp, 2, 32, 200, w, w + 0.1, progress=False)
    assert torch.all(torch.isfinite(chain))
    assert torch.all(chain[..., 0] >= 0.0)


def test_no_trailing_zeros_in_batch_chain():
    # regression test: the old torch branch left the final row of the
    # preallocated chain uninitialized (all zeros) with default settings
    torch.manual_seed(0)
    state = make_state(n_walkers=16, n_batch=3, seed=7)

    def batch_logp(theta):
        return -0.5 * torch.sum(theta ** 2, dim=-1)

    chain = sample_batch(batch_logp, 50, state, progress=False)
    assert chain.shape[0] == 50
    assert not torch.all(chain[-1] == 0.0)
    assert not torch.all(chain[0] == 0.0)


def test_burnin_and_thin():
    state = make_state(n_walkers=8, n_batch=2)
    # stored steps: 3, 5, 7, 9 -> 4 entries
    chain = sample_batch(gaussian_logp, 10, state, n_burnin=3, thin=2, progress=False)
    assert chain.shape == (4, 16, 2, 2)

    with pytest.raises(ValueError):
        sample_batch(gaussian_logp, 10, state, n_burnin=10, progress=False)
    with pytest.raises(ValueError):
        sample_batch(gaussian_logp, 10, state, thin=0, progress=False)


def test_reproducible_with_manual_seed():
    w1, w2 = make_state()
    torch.manual_seed(42)
    chain1 = sample(gaussian_logp, 2, 64, 50, w1, w2, progress=False)
    torch.manual_seed(42)
    chain2 = sample(gaussian_logp, 2, 64, 50, w1, w2, progress=False)
    assert torch.equal(chain1, chain2)


def test_float64_supported():
    w1, w2 = make_state(dtype=torch.float64)
    chain = sample(gaussian_logp, 2, 64, 20, w1, w2, progress=False)
    assert chain.dtype == torch.float64


def test_sample_batch_recovers_batch_means():
    torch.manual_seed(4)
    means = torch.tensor([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * torch.sum((theta - means[None, :, None]) ** 2, dim=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = sample_batch(batch_logp, 600, state, progress=False)
    assert chain.shape == (600, 128, 3, 2)
    est_means = chain[300:].mean(dim=(0, 1, 3))
    assert torch.all(torch.abs(est_means - means) < 0.15)

    chain, lp = sample_batch(batch_logp, 10, state, progress=False, save_lp=True)
    assert lp.shape == (10, 128, 3)


def test_args_passthrough():
    def shifted_logp(theta, mu):
        return -0.5 * torch.sum((theta - mu) ** 2, dim=-1)

    torch.manual_seed(6)
    w1, w2 = make_state(n_walkers=64)
    chain = sample(shifted_logp, 2, 64, 400, w1, w2, progress=False, args=(3.0,))
    samples = chain[200:].reshape(-1, 2).numpy()
    assert np.all(np.abs(samples.mean(axis=0) - 3.0) < 0.15)


def test_save_extras():
    # extras returned by log_prob are stored per sample; they must equal the
    # same quantities recomputed from the stored chain (accept/reject tracking)
    def logp_with_extras(theta):
        logp = -0.5 * torch.sum(theta ** 2, dim=-1)
        extras = torch.stack([theta.sum(dim=-1), (theta ** 2).sum(dim=-1)], dim=-1)
        return logp, extras

    w1, w2 = make_state(n_walkers=16)
    chain, extras = sample(logp_with_extras, 2, 16, 20, w1, w2, progress=False,
                           save_extras=True)
    assert extras.shape == (20, 32, 2)
    expected = torch.stack([chain.sum(dim=-1), (chain ** 2).sum(dim=-1)], dim=-1)
    assert torch.allclose(extras, expected, atol=1e-5)

    # combined with save_lp the return order is (chain, lp, extras)
    chain, lp, extras = sample(logp_with_extras, 2, 16, 10, w1, w2, progress=False,
                               save_lp=True, save_extras=True)
    assert lp.shape == (10, 32)
    assert extras.shape == (10, 32, 2)

    # batched variant
    state = make_state(n_walkers=16, n_batch=3)
    chain, extras = sample_batch(logp_with_extras, 20, state, progress=False,
                                 save_extras=True)
    assert extras.shape == (20, 32, 3, 2)
    expected = torch.stack([chain.sum(dim=-1), (chain ** 2).sum(dim=-1)], dim=-1)
    assert torch.allclose(extras, expected, atol=1e-5)


def test_legacy_top_level_import():
    from affine import sample as legacy_sample
    from affine import sample_batch as legacy_batch

    w1, w2 = make_state(n_walkers=16, n_params=3)
    chain = legacy_sample(gaussian_logp, 3, 16, 10, w1, w2, progress=False)
    assert chain.shape == (10, 32, 3)

    state = list(make_state(n_walkers=16, n_batch=4))
    chain = legacy_batch(gaussian_logp, 10, state, args=[], progress=False)
    assert chain.shape == (10, 32, 4, 2)
