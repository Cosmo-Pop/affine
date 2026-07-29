"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in TensorFlow.

Implements the parallel stretch move of Foreman-Mackey et al. (2013): the
ensemble is split into two halves, and each half is updated using partners
drawn from the other half, so that a whole half can be evolved with a single
vectorized call to the target log-probability.
"""

import tensorflow as tf
from tqdm import trange

from ._utils import keep_step, validate_steps

__all__ = ["sample", "sample_batch"]


def _sanitize(logp):
    """Replace NaN log-probabilities with -inf so those proposals are rejected."""
    logp = tf.convert_to_tensor(logp)
    neg_inf = tf.fill(tf.shape(logp), tf.constant(float("-inf"), logp.dtype))
    return tf.where(tf.math.is_nan(logp), neg_inf, logp)


def _make_rng(rng):
    """Return (uniform, randint) draw functions, seeded if ``rng`` is given.

    ``rng`` may be None (TensorFlow's global RNG, seedable with
    ``tf.random.set_seed``), an integer seed, or a ``tf.random.Generator``.
    """
    if rng is None:
        def uniform(shape, dtype):
            return tf.random.uniform(shape, dtype=dtype)

        def randint(shape, maxval):
            return tf.random.uniform(shape, minval=0, maxval=maxval, dtype=tf.int32)
    else:
        gen = rng if isinstance(rng, tf.random.Generator) else tf.random.Generator.from_seed(int(rng))

        def uniform(shape, dtype):
            return gen.uniform(shape, dtype=dtype)

        def randint(shape, maxval):
            return gen.uniform(shape, minval=0, maxval=maxval, dtype=tf.int32)

    return uniform, randint


def _step_half(log_prob, args, moving, fixed, logp_moving, uniform, randint):
    """Stretch-move update of one half of the ensemble against the other."""
    n_walkers, n_batch, n_params = (int(s) for s in moving.shape)

    partners = tf.gather(fixed, randint([n_walkers], n_walkers))
    z = 0.5 * (uniform([n_walkers, n_batch], moving.dtype) + 1.0) ** 2
    proposed = partners + tf.expand_dims(z, -1) * (moving - partners)

    logp_proposed = _sanitize(log_prob(proposed, *args))

    # accept/reject in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
    d = tf.constant(n_params - 1, logp_proposed.dtype)
    log_accept = d * tf.cast(tf.math.log(z), logp_proposed.dtype) + logp_proposed - logp_moving
    u = uniform([n_walkers, n_batch], logp_proposed.dtype)
    accept = tf.math.log(u) < log_accept

    moving = tf.where(tf.expand_dims(accept, -1), proposed, moving)
    logp_moving = tf.where(accept, logp_proposed, logp_moving)
    return moving, logp_moving


def sample_batch(log_prob, n_steps, current_state, *, args=(), rng=None,
                 n_burnin=0, thin=1, progressbar=True, return_logp=False):
    """Sample a batch of independent posteriors with the affine sampler.

    Parameters
    ----------
    log_prob : callable
        Target log-probability. For input of shape ``(n_walkers, n_batch,
        n_params)`` it must return shape ``(n_walkers, n_batch)``. NaN return
        values are treated as log-probability -inf (proposal rejected).
    n_steps : int
        Number of steps, counting the initial state as step 0. With the
        default ``n_burnin``/``thin`` the chain has ``n_steps`` entries.
    current_state : tuple of tensors
        The two walker ensembles ``(walkers1, walkers2)``, each of shape
        ``(n_walkers, n_batch, n_params)``.
    args : tuple, optional
        Extra positional arguments passed to ``log_prob``.
    rng : int or tf.random.Generator, optional
        Seed or generator for reproducible sampling. Default uses
        TensorFlow's global RNG.
    n_burnin : int, optional
        Number of leading steps to discard from the returned chain.
    thin : int, optional
        Store only every ``thin``-th step after burn-in.
    progressbar : bool, optional
        Show a tqdm progress bar.
    return_logp : bool, optional
        Also return the log-probability of every stored sample.

    Returns
    -------
    chain : tf.Tensor
        Shape ``(n_stored, 2*n_walkers, n_batch, n_params)`` where
        ``n_stored = ceil((n_steps - n_burnin) / thin)``.
    logp : tf.Tensor
        Shape ``(n_stored, 2*n_walkers, n_batch)``. Only if ``return_logp``.
    """
    validate_steps(n_steps, n_burnin, thin)
    state1, state2 = current_state
    state1 = tf.convert_to_tensor(state1)
    state2 = tf.convert_to_tensor(state2)
    if len(state1.shape) != 3:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_batch, n_params), got {tuple(state1.shape)}"
        )
    uniform, randint = _make_rng(rng)

    logp1 = _sanitize(log_prob(state1, *args))
    logp2 = _sanitize(log_prob(state2, *args))

    chain, logp_chain = [], []
    if keep_step(0, n_burnin, thin):
        chain.append(tf.concat([state1, state2], axis=0))
        logp_chain.append(tf.concat([logp1, logp2], axis=0))

    steps = trange(1, n_steps) if progressbar else range(1, n_steps)
    for step in steps:
        state1, logp1 = _step_half(log_prob, args, state1, state2, logp1, uniform, randint)
        state2, logp2 = _step_half(log_prob, args, state2, state1, logp2, uniform, randint)
        if keep_step(step, n_burnin, thin):
            chain.append(tf.concat([state1, state2], axis=0))
            logp_chain.append(tf.concat([logp1, logp2], axis=0))

    chain = tf.stack(chain, axis=0)
    if return_logp:
        return chain, tf.stack(logp_chain, axis=0)
    return chain


def sample(log_prob, n_steps, current_state, *, args=(), rng=None,
           n_burnin=0, thin=1, progressbar=True, return_logp=False):
    """Sample a single posterior with the affine sampler.

    Identical to :func:`sample_batch` but without the batch dimension:
    ``current_state`` holds two ensembles of shape ``(n_walkers, n_params)``,
    ``log_prob`` maps ``(n_walkers, n_params)`` to ``(n_walkers,)``, and the
    returned chain has shape ``(n_stored, 2*n_walkers, n_params)``.
    """
    state1, state2 = current_state
    state1 = tf.convert_to_tensor(state1)
    state2 = tf.convert_to_tensor(state2)
    if len(state1.shape) != 2:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_params), got {tuple(state1.shape)}"
        )

    def batched_log_prob(theta, *a):
        return tf.expand_dims(log_prob(theta[:, 0, :], *a), -1)

    result = sample_batch(
        batched_log_prob, n_steps,
        (tf.expand_dims(state1, 1), tf.expand_dims(state2, 1)),
        args=args, rng=rng, n_burnin=n_burnin, thin=thin,
        progressbar=progressbar, return_logp=return_logp,
    )
    if return_logp:
        chain, logp = result
        return chain[:, :, 0, :], logp[:, :, 0]
    return result[:, :, 0, :]
