"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in TensorFlow.

Uses the parallel stretch move of Foreman-Mackey et al. (2013): the ensemble
is split into two halves and each half is updated with partners drawn from
the other half, so that a whole half is evolved with a single vectorized call
to the target log-probability.

For reproducible chains call ``tf.random.set_seed(<seed>)`` before sampling.
"""

import tensorflow as tf
from tqdm import trange

__all__ = ["affine_sample", "affine_sample_batch"]


def _nan_to_neginf(logp):
    # treat NaN log probabilities as -inf so those proposals are rejected
    logp = tf.convert_to_tensor(logp)
    return tf.where(tf.math.is_nan(logp), tf.constant(float("-inf"), logp.dtype), logp)


def affine_sample(log_prob, n_steps, current_state, args=[], progressbar=True):
    """Sample a single posterior.

    Parameters
    ----------
    log_prob : callable
        Vectorized target log-probability: maps parameters of shape
        ``(n_walkers, n_params)`` to log-probabilities of shape
        ``(n_walkers,)``. NaN values are treated as -inf (proposal rejected).
    n_steps : int
        Number of steps; the initial state counts as step 0, so the returned
        chain has ``n_steps`` entries.
    current_state : pair of tensors
        The two walker ensembles, each of shape ``(n_walkers, n_params)``.
    args : list, optional
        Extra arguments passed to ``log_prob``.
    progressbar : bool, optional
        Show a tqdm progress bar.

    Returns
    -------
    tf.Tensor of shape ``(n_steps, 2 * n_walkers, n_params)``
    """
    # split the current state
    current_state1 = tf.convert_to_tensor(current_state[0])
    current_state2 = tf.convert_to_tensor(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers and parameters
    n_walkers, n_params = map(int, current_state1.shape)

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1, *args))
    logp_current2 = _nan_to_neginf(log_prob(current_state2, *args))

    # holder for the whole chain, starting with the initial state
    chain = [tf.concat([current_state1, current_state2], axis=0)]

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = tf.gather(current_state2, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z1 = 0.5 * (tf.random.uniform([n_walkers], dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1, *args))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * tf.cast(tf.math.log(z1), logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = tf.math.log(tf.random.uniform([n_walkers], dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = tf.where(accept1[:, None], proposed_state1, current_state1)
        logp_current1 = tf.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = tf.gather(current_state1, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z2 = 0.5 * (tf.random.uniform([n_walkers], dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2, *args))

        # acceptance in log space
        log_accept2 = (n_params - 1) * tf.cast(tf.math.log(z2), logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = tf.math.log(tf.random.uniform([n_walkers], dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = tf.where(accept2[:, None], proposed_state2, current_state2)
        logp_current2 = tf.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        chain.append(tf.concat([current_state1, current_state2], axis=0))

    # stack up the chain and return
    return tf.stack(chain, axis=0)


def affine_sample_batch(log_prob, n_steps, current_state, args=[], progressbar=True):
    """Sample a batch of independent posteriors simultaneously.

    Same as :func:`affine_sample`, but with an extra batch dimension: walker
    states have shape ``(n_walkers, n_batch, n_params)`` and ``log_prob``
    must return shape ``(n_walkers, n_batch)``.

    Returns
    -------
    tf.Tensor of shape ``(n_steps, 2 * n_walkers, n_batch, n_params)``
    """
    # split the current state
    current_state1 = tf.convert_to_tensor(current_state[0])
    current_state2 = tf.convert_to_tensor(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers, batch size, and parameters
    n_walkers, n_batch, n_params = map(int, current_state1.shape)

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1, *args))
    logp_current2 = _nan_to_neginf(log_prob(current_state2, *args))

    # holder for the whole chain, starting with the initial state
    chain = [tf.concat([current_state1, current_state2], axis=0)]

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = tf.gather(current_state2, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z1 = 0.5 * (tf.random.uniform([n_walkers, n_batch], dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, :, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1, *args))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * tf.cast(tf.math.log(z1), logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = tf.math.log(tf.random.uniform([n_walkers, n_batch], dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = tf.where(accept1[:, :, None], proposed_state1, current_state1)
        logp_current1 = tf.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = tf.gather(current_state1, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z2 = 0.5 * (tf.random.uniform([n_walkers, n_batch], dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, :, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2, *args))

        # acceptance in log space
        log_accept2 = (n_params - 1) * tf.cast(tf.math.log(z2), logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = tf.math.log(tf.random.uniform([n_walkers, n_batch], dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = tf.where(accept2[:, :, None], proposed_state2, current_state2)
        logp_current2 = tf.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        chain.append(tf.concat([current_state1, current_state2], axis=0))

    # stack up the chain and return
    return tf.stack(chain, axis=0)
