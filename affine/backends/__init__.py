"""Backend implementations of the affine-invariant ensemble sampler.

Each backend lives in its own module and is only imported when used, so
installing a single framework is enough:

- :mod:`affine.backends.tensorflow`
- :mod:`affine.backends.torch`
- :mod:`affine.backends.jax`

All backends expose the same two functions, ``sample`` and ``sample_batch``.
"""
