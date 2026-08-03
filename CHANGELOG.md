# Changelog

## 1.0.0

Merges the historical `main` (TensorFlow), `torch`, and `jax` branches into
a single package: one self-contained module per framework
(`affine.tensorflow`, `affine.torch`, `affine.jax`), each keeping the
function signatures its branch had. Frameworks are imported lazily, so only
the one you use needs to be installed.

### New

- Batched sampling (`sample_batch`) added to the JAX module (previously
  TensorFlow and PyTorch only).
- PyTorch `sample_batch` keeps its `n_burnin` and `thin` options, now with
  input validation.
- Proper packaging via `pyproject.toml` with optional extras
  (`pip install affine[tensorflow|torch|jax]`), a test suite, and CI.

### Fixed (relative to the old branches)

- **torch:** NaN log-probabilities were mapped to `+inf` instead of `-inf`
  in `sample()`, so proposals in invalid regions of parameter space were
  *accepted* with probability 1 (and the walker then froze there). They are
  now correctly treated as `-inf` and rejected.
- **torch:** with the default `n_burnin=0, thin=1`, `sample_batch()`
  returned a chain whose final row was uninitialized (all zeros). The chain
  now stores the initial state and every subsequent step, with no
  uninitialized entries.
- **torch:** `sample()` crashed with a device mismatch when the walkers
  lived on a GPU; random draws now follow the walkers' device.
- All modules: acceptance is computed in log space
  (`log u < (d-1) log z + Δlog p`) instead of `u < z^(d-1) exp(Δlog p)`,
  which overflowed to NaN in float32 for d ≳ 130 parameters and silently
  rejected valid proposals.
- All modules: dtype follows the walkers (float64 works; float32 was
  previously hard-coded), and tensors returned by `log_prob` are no longer
  mutated in place.
- TensorFlow: partner selection now uses TensorFlow's RNG instead of NumPy's,
  so `tf.random.set_seed` makes chains reproducible (PyTorch:
  `torch.manual_seed`; JAX: the explicit PRNG key).
- Gibbs example notebook: latent chains were split with `n_hyper_walkers`
  instead of `n_latent_walkers`, and `tf.concat` of scalars fails on modern
  TensorFlow.

### Backwards compatibility

The top-level entry points of all three branches are preserved:
`affine_sample`/`affine_sample_batch` (TensorFlow) and `sample`/`sample_batch`
(torch/jax call styles, distinguished by their first argument). The
`affine.affine` module path from the pre-1.0 layout was removed — import from
the top level or from `affine.tensorflow` instead. Chains are not bitwise
identical to the old implementations (the acceptance step is mathematically
equivalent but numerically different), and the torch-branch `sample_batch`
chain now includes the initial state.
