# Changelog

## 1.0.0

Unified multi-backend release, merging the historical `main` (TensorFlow),
`torch`, and `jax` branches into a single package.

### New

- Three interchangeable backends with a common API: `affine.backends.tensorflow`,
  `affine.backends.torch`, `affine.backends.jax`, each providing `sample` and
  `sample_batch`. Backends are imported lazily, so only the framework you use
  needs to be installed.
- Batched sampling (`sample_batch`) is now available in all three backends
  (previously TensorFlow and PyTorch only).
- All samplers accept `rng` (integer seed, framework generator, or JAX PRNG
  key) for reproducible chains, plus `n_burnin`, `thin`, and `return_logp`.
- Proper packaging via `pyproject.toml` with optional extras
  (`pip install affine[tensorflow|torch|jax]`), a test suite, and CI.

### Fixed (relative to the old branches)

- **torch branch:** NaN log-probabilities were mapped to `+inf` instead of
  `-inf` in `sample()`, so proposals in invalid regions of parameter space
  were *accepted* with probability 1 (and the walker then froze there).
  They are now correctly treated as `-inf` and rejected.
- **torch branch:** with the default `n_burnin=0, thin=1`, `sample_batch()`
  returned a chain whose final row was uninitialized (all zeros). The chain
  now stores the initial state and every subsequent step, with no
  uninitialized entries.
- **torch branch:** `sample()` crashed with a device mismatch when the walkers
  lived on a GPU; random draws now follow the walkers' device.
- Acceptance is computed in log space (`log u < (d-1) log z + Δlog p`) instead
  of `u < z^(d-1) exp(Δlog p)`, which overflowed to NaN in float32 for
  d ≳ 130 parameters and silently rejected valid proposals.
- The samplers are now dtype-agnostic (float64 walkers work; float32 was
  previously hard-coded) and no longer mutate the tensors returned by
  `log_prob` in place.
- The TensorFlow sampler no longer mixes NumPy and TensorFlow random number
  generation.

### Backwards compatibility

The historical entry points are preserved at the top level with their old
signatures: `affine_sample`/`affine_sample_batch` (TensorFlow) and
`sample`/`sample_batch` (torch/jax branches). `from affine.affine import
affine_sample` (the pre-1.0 module layout) also still works. Chains are not
bitwise-identical to the old implementations (the acceptance step is
mathematically equivalent but numerically different), and the torch-branch
`sample_batch` chain now includes the initial state.
