"""Parallelized affine-invariant ensemble MCMC (Goodman & Weare 2010).

The sampler is implemented in three interchangeable backends, imported lazily
so only the framework you actually use needs to be installed:

    from affine.backends import tensorflow  # affine.backends.tensorflow.sample(...)
    from affine.backends import torch       # affine.backends.torch.sample(...)
    from affine.backends import jax         # affine.backends.jax.sample(...)

Every backend exposes ``sample`` (single posterior) and ``sample_batch``
(batch of independent posteriors) with a common signature.

For backwards compatibility the historical entry points are still available
at the top level: ``affine_sample``/``affine_sample_batch`` (TensorFlow,
former ``main`` branch) and ``sample``/``sample_batch`` (former ``torch`` and
``jax`` branches).
"""

from importlib.util import find_spec

__version__ = "1.0.0"

_LEGACY_TF = ("affine_sample", "affine_sample_batch")
_LEGACY_TORCH_JAX = ("sample", "sample_batch")

# advertise only the legacy names whose backend is installed, so
# `from affine import *` keeps working in single-framework environments
__all__ = ["__version__"]
if find_spec("tensorflow") is not None:
    __all__ += list(_LEGACY_TF)
if find_spec("torch") is not None or find_spec("jax") is not None:
    __all__ += list(_LEGACY_TORCH_JAX)


def __getattr__(name):
    if name in _LEGACY_TF + _LEGACY_TORCH_JAX:
        from . import _compat
        return getattr(_compat, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LEGACY_TF) | set(_LEGACY_TORCH_JAX))
