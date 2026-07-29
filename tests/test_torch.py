import numpy as np
import pytest

torch = pytest.importorskip("torch")

from affine.backends import torch as backend


def gaussian_logp(theta):
    return -0.5 * torch.sum(theta ** 2, dim=-1)


def make_state(n_walkers=64, n_params=2, n_batch=None, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_walkers, n_params) if n_batch is None else (n_walkers, n_batch, n_params)
    return (0.5 * torch.randn(shape, generator=gen, dtype=dtype),
            0.5 * torch.randn(shape, generator=gen, dtype=dtype))


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
    samples = chain[300:].reshape(-1, 2).numpy()
    assert np.all(np.abs(samples.mean(axis=0)) < 0.1)
    assert np.all(np.abs(samples.std(axis=0) - 1.0) < 0.15)


def test_nan_logp_is_rejected():
    # regression test for the old torch branch, which mapped NaN log-probs to
    # +inf and therefore accepted every proposal in the invalid region
    def half_gaussian_logp(theta):
        logp = -0.5 * torch.sum(theta ** 2, dim=-1)
        return torch.where(theta[..., 0] < 0.0, torch.full_like(logp, float("nan")), logp)

    gen = torch.Generator().manual_seed(3)
    w = 0.5 + torch.rand((32, 2), generator=gen)
    state = (w, w + 0.1)
    chain = backend.sample(half_gaussian_logp, 200, state, rng=2, progressbar=False)
    assert torch.all(torch.isfinite(chain))
    assert torch.all(chain[..., 0] >= 0.0)


def test_no_trailing_zeros_in_batch_chain():
    # regression test for the old torch branch, which with the default
    # n_burnin/thin left the final row of the preallocated chain all zeros
    state = make_state(n_walkers=16, n_batch=3, seed=7)

    def batch_logp(theta):
        return -0.5 * torch.sum(theta ** 2, dim=-1)

    chain = backend.sample_batch(batch_logp, 50, state, rng=0, progressbar=False)
    assert chain.shape[0] == 50
    assert not torch.all(chain[-1] == 0.0)
    assert not torch.all(chain[0] == 0.0)


def test_reproducible_with_seed():
    state = make_state()
    chain1 = backend.sample(gaussian_logp, 50, state, rng=42, progressbar=False)
    chain2 = backend.sample(gaussian_logp, 50, state, rng=42, progressbar=False)
    assert torch.equal(chain1, chain2)


def test_float64_supported():
    state = make_state(dtype=torch.float64)
    chain = backend.sample(gaussian_logp, 20, state, rng=0, progressbar=False)
    assert chain.dtype == torch.float64


def test_sample_batch_recovers_batch_means():
    means = torch.tensor([0.0, 2.0, -1.0])

    def batch_logp(theta):
        return -0.5 * torch.sum((theta - means[None, :, None]) ** 2, dim=-1)

    state = make_state(n_walkers=64, n_params=2, n_batch=3, seed=5)
    chain = backend.sample_batch(batch_logp, 600, state, rng=4, progressbar=False)
    assert chain.shape == (600, 128, 3, 2)
    est_means = chain[300:].mean(dim=(0, 1, 3))
    assert torch.all(torch.abs(est_means - means) < 0.15)


def test_legacy_api():
    from affine import sample, sample_batch

    w1, w2 = make_state(n_walkers=16, n_params=3)
    chain = sample(gaussian_logp, 3, 16, 10, w1, w2, progress=False)
    assert chain.shape == (10, 32, 3)

    chain, lp = sample(gaussian_logp, 3, 16, 10, w1, w2, progress=False, save_lp=True)
    assert chain.shape == (10, 32, 3)
    assert lp.shape == (10, 32)

    def batch_logp(theta):
        return -0.5 * torch.sum(theta ** 2, dim=-1)

    state = list(make_state(n_walkers=16, n_batch=4))
    chain = sample_batch(batch_logp, 10, state, args=[], progress=False)
    assert chain.shape == (10, 32, 4, 2)
    assert not torch.all(chain[-1] == 0.0)


def test_legacy_api_rejects_wrong_shapes():
    from affine import sample

    w1, w2 = make_state(n_walkers=16, n_params=3)
    with pytest.raises(ValueError):
        sample(gaussian_logp, 2, 16, 10, w1, w2, progress=False)
