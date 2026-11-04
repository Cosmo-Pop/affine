"""Parallelized affine-invariant ensemble MCMC (Goodman & Weare 2010).

Pick the module for your framework:

    from affine.tensorflow import sample, sample_batch
    from affine.torch import sample, sample_batch
    from affine.jax import sample, sample_batch

The wrappers below keep the historical top-level imports working; they defer
importing the framework until the first call, so ``import affine`` needs none
of them installed.
"""

__version__ = "1.0.0"


def affine_sample(*args, **kwargs):
    """TensorFlow sampler (old main branch); see affine.tensorflow.sample."""
    from .tensorflow import sample
    return sample(*args, **kwargs)


def affine_sample_batch(*args, **kwargs):
    """TensorFlow batch sampler (old main branch); see affine.tensorflow.sample_batch."""
    from .tensorflow import sample_batch
    return sample_batch(*args, **kwargs)


def sample(*args, **kwargs):
    """Old torch- and jax-branch sampler, told apart by the first argument.

    torch style: sample(log_prob, n_steps, current_state, ...)
    jax style:   sample(rng, log_prob, n_steps, current_state, ...)
    """
    if args and callable(args[0]):
        from .torch import sample
    else:
        from .jax import sample
    return sample(*args, **kwargs)


def sample_batch(*args, **kwargs):
    """Old torch-branch batch sampler; see affine.torch.sample_batch."""
    from .torch import sample_batch
    return sample_batch(*args, **kwargs)
