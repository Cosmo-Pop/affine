# affine

Parallelized affine-invariant ensemble MCMC sampling ([Goodman & Weare 2010](https://doi.org/10.2140/camcos.2010.5.65)),
implemented in **TensorFlow**, **PyTorch**, and **JAX** with a common API.

The sampler uses the parallel stretch move of
[Foreman-Mackey et al. (2013)](https://doi.org/10.1086/670067) (as in `emcee`):
the ensemble is split into two halves and each half is updated using partners
from the other half, so an entire half is evolved with a single vectorized
(GPU-friendly) call to the target log-probability. A batched mode samples many
independent posteriors simultaneously — useful for hierarchical models and
population analyses.

## Installation

Install with the backend(s) you want to use:

```bash
pip install "affine[tensorflow] @ git+https://github.com/Cosmo-Pop/affine.git"
pip install "affine[torch] @ git+https://github.com/Cosmo-Pop/affine.git"
pip install "affine[jax] @ git+https://github.com/Cosmo-Pop/affine.git"
```

or plain `pip install git+https://github.com/Cosmo-Pop/affine.git` if the
framework is already installed. Only the backend you import is required at
runtime.

## Usage

Every backend provides two functions with the same signature:

- `sample(log_prob, n_steps, current_state, ...)` — sample a single posterior.
- `sample_batch(log_prob, n_steps, current_state, ...)` — sample a batch of
  independent posteriors in parallel.

`current_state` is a tuple of **two** walker ensembles (the two halves of the
parallel stretch move), each of shape `(n_walkers, n_params)` for `sample` or
`(n_walkers, n_batch, n_params)` for `sample_batch`. `log_prob` must map an
ensemble of parameter vectors to their log-probabilities, vectorized over the
leading dimension(s): `(n_walkers, n_params) -> (n_walkers,)`, or
`(n_walkers, n_batch, n_params) -> (n_walkers, n_batch)`. NaN log-probabilities
are treated as `-inf` (the proposal is rejected).

Common options: `args` (extra arguments for `log_prob`), `rng` (seed /
generator / PRNG key), `n_burnin`, `thin`, `progressbar`, and `return_logp`
(also return the log-probability of every stored sample). The returned chain
has shape `(n_stored, 2*n_walkers, [n_batch,] n_params)` and includes the
initial state as its first entry.

### TensorFlow

```python
import tensorflow as tf
from affine.backends import tensorflow as affine_tf

def log_prob(theta):
    return -0.5 * tf.reduce_sum(theta**2, axis=-1)

walkers = (tf.random.normal([100, 2]), tf.random.normal([100, 2]))
chain = affine_tf.sample(log_prob, 1000, walkers, rng=0)  # (1000, 200, 2)
```

### PyTorch

```python
import torch
from affine.backends import torch as affine_torch

def log_prob(theta):
    return -0.5 * torch.sum(theta**2, dim=-1)

walkers = (torch.randn(100, 2), torch.randn(100, 2))
chain = affine_torch.sample(log_prob, 1000, walkers, rng=0, device="cpu")
```

### JAX

```python
import jax
import jax.numpy as jnp
from affine.backends import jax as affine_jax

def log_prob(theta):
    return -0.5 * jnp.sum(theta**2, axis=-1)

key, k1, k2 = jax.random.split(jax.random.PRNGKey(0), 3)
walkers = (jax.random.normal(k1, (100, 2)), jax.random.normal(k2, (100, 2)))
chain = affine_jax.sample(log_prob, 1000, walkers, rng=key)
```

Worked examples, including Gibbs sampling of a hierarchical model with the
batched sampler, are in [`examples/`](examples/).

## Backwards compatibility

Code written against the historical branches of this repository keeps working
unchanged:

- `from affine import affine_sample, affine_sample_batch` — the TensorFlow
  functions from the old `main` branch.
- `from affine import sample, sample_batch` — the old `torch`-branch API
  (`sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2)`), and
  the old `jax`-branch API (`sample(rng, log_prob, n_steps, current_state)`),
  distinguished automatically by their first argument.

See [CHANGELOG.md](CHANGELOG.md) for the bug fixes and behavioural
improvements relative to the old branches.

## Tips

- Initialize the two walker ensembles in a tight ball around a reasonable
  parameter guess, **inside the support of the posterior** — walkers that
  start at zero probability can never move.
- You need `n_walkers > n_params` (ideally many times larger) per ensemble
  for the stretch move to span the parameter space.

## Citing

If you use this sampler, please cite
[Goodman & Weare (2010)](https://doi.org/10.2140/camcos.2010.5.65) for the
algorithm and [Foreman-Mackey et al. (2013)](https://doi.org/10.1086/670067)
for the parallel stretch move.

## License

MIT — see [LICENSE](LICENSE).
