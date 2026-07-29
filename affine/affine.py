"""Backwards-compatibility shim for the pre-1.0 module layout.

The TensorFlow implementation now lives in :mod:`affine.backends.tensorflow`;
``from affine.affine import affine_sample`` keeps working via this module.
"""

from ._compat import affine_sample, affine_sample_batch  # noqa: F401
