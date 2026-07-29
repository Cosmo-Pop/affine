"""Legacy top-level API.

These wrappers preserve the call signatures of the three historical branches
of this repository so that existing code keeps working unchanged:

- ``main`` (TensorFlow): ``affine_sample``, ``affine_sample_batch``
- ``torch`` branch:      ``sample``, ``sample_batch``
- ``jax`` branch:        ``sample``

New code should prefer the uniform API in :mod:`affine.backends`.
"""


def affine_sample(log_prob, n_steps, current_state, args=(), progressbar=True):
    """TensorFlow sampler with the historical ``main``-branch signature."""
    from .backends import tensorflow as backend
    return backend.sample(log_prob, n_steps, tuple(current_state),
                          args=tuple(args), progressbar=progressbar)


def affine_sample_batch(log_prob, n_steps, current_state, args=(), progressbar=True):
    """Batched TensorFlow sampler with the historical ``main``-branch signature."""
    from .backends import tensorflow as backend
    return backend.sample_batch(log_prob, n_steps, tuple(current_state),
                                args=tuple(args), progressbar=progressbar)


def _torch_sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2,
                  progress=True, save_lp=False):
    """PyTorch sampler with the historical ``torch``-branch signature."""
    from .backends import torch as backend
    if tuple(walkers1.shape) != (n_walkers, n_params):
        raise ValueError(
            f"walkers1 has shape {tuple(walkers1.shape)}, expected (n_walkers, n_params) = "
            f"({n_walkers}, {n_params})"
        )
    return backend.sample(log_prob, n_steps, (walkers1, walkers2),
                          progressbar=progress, return_logp=save_lp)


def _jax_sample(rng, log_prob, n_steps, current_state, progressbar=True):
    """JAX sampler with the historical ``jax``-branch signature."""
    from .backends import jax as backend
    return backend.sample(log_prob, n_steps, tuple(current_state),
                          rng=rng, progressbar=progressbar)


def sample(*args, **kwargs):
    """Legacy ``sample`` supporting both the torch- and jax-branch signatures.

    - torch branch: ``sample(log_prob, n_params, n_walkers, n_steps, walkers1,
      walkers2, progress=True, save_lp=False)``
    - jax branch: ``sample(rng, log_prob, n_steps, current_state,
      progressbar=True)``

    Dispatch: if the first argument is a PRNG key (or ``rng`` is passed as a
    keyword) the jax version is used, otherwise the torch version.
    """
    jax_style = "rng" in kwargs or (len(args) > 0 and not callable(args[0]))
    if jax_style:
        return _jax_sample(*args, **kwargs)
    return _torch_sample(*args, **kwargs)


def sample_batch(log_prob, n_steps, current_state, n_burnin=0, thin=1, args=(),
                 progress=True, save_lp=False, device="cpu"):
    """Batched PyTorch sampler with the historical ``torch``-branch signature.

    Unlike the historical version, the chain now includes the initial state
    and never contains uninitialized (zero) rows, and NaN log-probabilities
    are correctly treated as -inf. ``device="cpu"`` (the default) leaves the
    walkers on whatever device they are already on, matching the old
    behaviour of only moving them when a device was explicitly requested.
    """
    from .backends import torch as backend
    return backend.sample_batch(log_prob, n_steps, tuple(current_state),
                                args=tuple(args), n_burnin=n_burnin, thin=thin,
                                progressbar=progress, return_logp=save_lp,
                                device=None if device == "cpu" else device)
