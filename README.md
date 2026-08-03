# affine

Parallelized affine-invariant ensemble MCMC sampling ([Goodman & Weare 2010](https://doi.org/10.2140/camcos.2010.5.65)),
implemented in **TensorFlow**, **PyTorch**, and **JAX**.

The sampler uses the parallel stretch move of
[Foreman-Mackey et al. (2013)](https://doi.org/10.1086/670067) (as in `emcee`):
the ensemble is split into two halves and each half is updated using partners
from the other half, so an entire half is evolved with a single vectorized
(GPU-friendly) call to the target log-probability. A batched mode samples many
independent posteriors simultaneously — useful for hierarchical models and
population analyses.

Each framework has its own self-contained module — `affine.tensorflow`,
`affine.torch`, `affine.jax` — and only the one you import needs to be
installed.

## Installation

```bash
pip install "affine[tensorflow] @ git+https://github.com/Cosmo-Pop/affine.git"
pip install "affine[torch] @ git+https://github.com/Cosmo-Pop/affine.git"
pip install "affine[jax] @ git+https://github.com/Cosmo-Pop/affine.git"
```

or plain `pip install git+https://github.com/Cosmo-Pop/affine.git` if your
framework is already installed.

## Usage

In all three modules you initialize *two* ensembles of walkers (the two
halves of the parallel stretch move), each of shape `(n_walkers, n_params)`
— ideally in a tight ball around a reasonable parameter guess, inside the
support of the posterior. `log_prob` must be vectorized over walkers:
`(n_walkers, n_params) -> (n_walkers,)`. NaN log-probabilities are treated
as `-inf` (the proposal is rejected). The returned chain has shape
`(n_steps, 2 * n_walkers, n_params)` and includes the initial state as its
first entry.

### TensorFlow

```python
import tensorflow as tf
from affine import affine_sample   # or: from affine.tensorflow import affine_sample

def log_prob(theta):
    return -0.5 * tf.reduce_sum(theta**2, axis=-1)

walkers1 = tf.random.normal([100, 2])
walkers2 = tf.random.normal([100, 2])
chain = affine_sample(log_prob, 1000, [walkers1, walkers2])   # (1000, 200, 2)
```

Reproducibility: call `tf.random.set_seed(seed)` before sampling.

### PyTorch

```python
import torch
from affine.torch import sample

def log_prob(theta):
    return -0.5 * torch.sum(theta**2, dim=-1)

walkers1 = torch.randn(100, 2)
walkers2 = torch.randn(100, 2)
chain = sample(log_prob, 2, 100, 1000, walkers1, walkers2)    # (1000, 200, 2)
```

The sampler runs on whatever device and dtype the walkers are on. Pass
`save_lp=True` to also get the log-probability of every sample.
Reproducibility: call `torch.manual_seed(seed)` before sampling.

### JAX

```python
import jax
import jax.numpy as jnp
from affine.jax import sample

def log_prob(theta):
    return -0.5 * jnp.sum(theta**2, axis=-1)

key, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
walkers1 = jax.random.normal(k1, (100, 2))
walkers2 = jax.random.normal(k2, (100, 2))
chain = sample(key, log_prob, 1000, (walkers1, walkers2))     # (1000, 200, 2)
```

JAX random numbers are functional, so the PRNG key is the first argument;
chains are exactly reproducible for a given key.

### Batched sampling

Every module also provides a batched variant that samples `n_batch`
independent posteriors at once: walker states get an extra middle dimension
`(n_walkers, n_batch, n_params)` and `log_prob` must return
`(n_walkers, n_batch)`. In TensorFlow it is `affine_sample_batch`, in
PyTorch and JAX `sample_batch`.

All samplers (single and batched, in all three modules) accept the keyword
options `args` (extra arguments passed to `log_prob`), `n_burnin` and `thin`
(to bound memory on long runs), and `save_lp` (also return the
log-probability of every stored sample).

Worked examples, including Gibbs sampling of a hierarchical model with the
batched sampler, are in [`examples/`](examples/).

<!-- ## Backwards compatibility

The historical entry points of all three original branches of this
repository still work at the top level:

- `from affine import affine_sample, affine_sample_batch` — old `main`
  branch (TensorFlow).
- `from affine import sample, sample_batch` — old `torch` and `jax`
  branches; the two `sample` call styles are distinguished automatically by
  their first argument.

See [CHANGELOG.md](CHANGELOG.md) for bug fixes relative to the old branches. -->

<!-- ## Tips

- Initialize walkers **inside the support of the posterior** — a walker that
  starts at zero probability can never move.
- Use `n_walkers` much larger than `n_params` per ensemble so the stretch
  move spans the parameter space. -->

## Citing

If you use this sampler, please cite
[Goodman & Weare (2010)](https://doi.org/10.2140/camcos.2010.5.65) for the
algorithm and [Foreman-Mackey et al. (2013)](https://doi.org/10.1086/670067)
for the parallel stretch move. Please also cite [Alsing et al. (2024)](https://doi.org/10.3847/1538-4365/ad5c69) , [Thorp et al. (2024)](https://doi.org/10.3847/1538-4357/ad7736) and [Thorp et al. (2025)](https://doi.org/10.3847/1538-4357/ae0936) where `affine` was first introduced.

## License

MIT — see [LICENSE](LICENSE).
