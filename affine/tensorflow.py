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


def _check_steps(n_steps, n_burnin, thin):
    # validate burn-in and thinning settings
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if not 0 <= n_burnin < n_steps:
        raise ValueError(f"need 0 <= n_burnin < n_steps, got n_burnin={n_burnin}, n_steps={n_steps}")


def _call_log_prob(log_prob, theta, args, save_extras):
    # evaluate the target; with save_extras, log_prob returns (logp, extras)
    if save_extras:
        logp, extras = log_prob(theta, *args)
        return _nan_to_neginf(logp), tf.convert_to_tensor(extras)
    return _nan_to_neginf(log_prob(theta, *args)), None


def affine_sample(log_prob, n_steps, current_state, args=[], progressbar=True, *,
                  n_burnin=0, thin=1, save_lp=False, save_extras=False,
                  progress=None):
    """Sample a single posterior.

    Parameters
    ----------
    log_prob : callable
        Vectorized target log-probability: maps parameters of shape
        ``(n_walkers, n_params)`` to log-probabilities of shape
        ``(n_walkers,)``. NaN values are treated as -inf (proposal rejected).
    n_steps : int
        Number of steps; the initial state counts as step 0.
    current_state : pair of tensors
        The two walker ensembles, each of shape ``(n_walkers, n_params)``.
    args : list, optional
        Extra arguments passed to ``log_prob``.
    progressbar : bool, optional
        Show a tqdm progress bar (``progress`` is accepted as an alias).
    n_burnin : int, optional
        Discard the first ``n_burnin`` steps from the returned chain.
    thin : int, optional
        Store only every ``thin``-th step after burn-in.
    save_lp : bool, optional
        Also return the log-probability of every stored sample.
    save_extras : bool, optional
        If True, ``log_prob`` must instead return a pair ``(logp, extras)``,
        where ``extras`` holds additional per-walker quantities of shape
        ``(n_walkers, n_extras)`` computed alongside the log-probability
        (e.g. the log prior and log likelihood separately). The extras of
        every stored sample are returned, tracking accept/reject.

    Returns
    -------
    chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers, n_params)``
        where ``n_stored = ceil((n_steps - n_burnin) / thin)``; with the
        defaults this is ``n_steps``, the initial state included as row 0.
    lp_chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers)``
        Only returned if ``save_lp=True``.
    extras_chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers, n_extras)``
        Only returned if ``save_extras=True`` (always the last return value).
    """
    if progress is not None:
        progressbar = progress
    _check_steps(n_steps, n_burnin, thin)

    # split the current state
    current_state1 = tf.convert_to_tensor(current_state[0])
    current_state2 = tf.convert_to_tensor(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers and parameters
    n_walkers, n_params = map(int, current_state1.shape)

    # initial target log prob (and extras) for the walkers
    logp_current1, extras_current1 = _call_log_prob(log_prob, current_state1, args, save_extras)
    logp_current2, extras_current2 = _call_log_prob(log_prob, current_state2, args, save_extras)

    # the state at steps n_burnin, n_burnin + thin, ... is stored
    def keep(step):
        return step >= n_burnin and (step - n_burnin) % thin == 0

    chain, lp_chain, extras_chain = [], [], []
    if keep(0):
        chain.append(tf.concat([current_state1, current_state2], axis=0))
        if save_lp:
            lp_chain.append(tf.concat([logp_current1, logp_current2], axis=0))
        if save_extras:
            extras_chain.append(tf.concat([extras_current1, extras_current2], axis=0))

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for step in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = tf.gather(current_state2, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z1 = 0.5 * (tf.random.uniform([n_walkers], dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1, extras_proposed1 = _call_log_prob(log_prob, proposed_state1, args, save_extras)

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * tf.cast(tf.math.log(z1), logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = tf.math.log(tf.random.uniform([n_walkers], dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = tf.where(accept1[:, None], proposed_state1, current_state1)
        logp_current1 = tf.where(accept1, logp_proposed1, logp_current1)
        if save_extras:
            extras_current1 = tf.where(accept1[:, None], extras_proposed1, extras_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = tf.gather(current_state1, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z2 = 0.5 * (tf.random.uniform([n_walkers], dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2, extras_proposed2 = _call_log_prob(log_prob, proposed_state2, args, save_extras)

        # acceptance in log space
        log_accept2 = (n_params - 1) * tf.cast(tf.math.log(z2), logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = tf.math.log(tf.random.uniform([n_walkers], dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = tf.where(accept2[:, None], proposed_state2, current_state2)
        logp_current2 = tf.where(accept2, logp_proposed2, logp_current2)
        if save_extras:
            extras_current2 = tf.where(accept2[:, None], extras_proposed2, extras_current2)

        # append to chain
        if keep(step):
            chain.append(tf.concat([current_state1, current_state2], axis=0))
            if save_lp:
                lp_chain.append(tf.concat([logp_current1, logp_current2], axis=0))
            if save_extras:
                extras_chain.append(tf.concat([extras_current1, extras_current2], axis=0))

    # stack up the chain and return
    chain = tf.stack(chain, axis=0)
    returns = (chain,)
    if save_lp:
        returns += (tf.stack(lp_chain, axis=0),)
    if save_extras:
        returns += (tf.stack(extras_chain, axis=0),)
    return returns if len(returns) > 1 else chain


def affine_sample_batch(log_prob, n_steps, current_state, args=[], progressbar=True, *,
                        n_burnin=0, thin=1, save_lp=False, save_extras=False,
                        progress=None):
    """Sample a batch of independent posteriors simultaneously.

    Same as :func:`affine_sample`, but with an extra batch dimension: walker
    states have shape ``(n_walkers, n_batch, n_params)``, ``log_prob`` must
    return shape ``(n_walkers, n_batch)``, and with ``save_extras=True`` the
    extras must have shape ``(n_walkers, n_batch, n_extras)``.

    Returns
    -------
    chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers, n_batch, n_params)``
        where ``n_stored = ceil((n_steps - n_burnin) / thin)``.
    lp_chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers, n_batch)``
        Only returned if ``save_lp=True``.
    extras_chain : tf.Tensor of shape ``(n_stored, 2 * n_walkers, n_batch, n_extras)``
        Only returned if ``save_extras=True`` (always the last return value).
    """
    if progress is not None:
        progressbar = progress
    _check_steps(n_steps, n_burnin, thin)

    # split the current state
    current_state1 = tf.convert_to_tensor(current_state[0])
    current_state2 = tf.convert_to_tensor(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers, batch size, and parameters
    n_walkers, n_batch, n_params = map(int, current_state1.shape)

    # initial target log prob (and extras) for the walkers
    logp_current1, extras_current1 = _call_log_prob(log_prob, current_state1, args, save_extras)
    logp_current2, extras_current2 = _call_log_prob(log_prob, current_state2, args, save_extras)

    # the state at steps n_burnin, n_burnin + thin, ... is stored
    def keep(step):
        return step >= n_burnin and (step - n_burnin) % thin == 0

    chain, lp_chain, extras_chain = [], [], []
    if keep(0):
        chain.append(tf.concat([current_state1, current_state2], axis=0))
        if save_lp:
            lp_chain.append(tf.concat([logp_current1, logp_current2], axis=0))
        if save_extras:
            extras_chain.append(tf.concat([extras_current1, extras_current2], axis=0))

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for step in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = tf.gather(current_state2, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z1 = 0.5 * (tf.random.uniform([n_walkers, n_batch], dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, :, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1, extras_proposed1 = _call_log_prob(log_prob, proposed_state1, args, save_extras)

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * tf.cast(tf.math.log(z1), logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = tf.math.log(tf.random.uniform([n_walkers, n_batch], dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = tf.where(accept1[:, :, None], proposed_state1, current_state1)
        logp_current1 = tf.where(accept1, logp_proposed1, logp_current1)
        if save_extras:
            extras_current1 = tf.where(accept1[:, :, None], extras_proposed1, extras_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = tf.gather(current_state1, tf.random.uniform([n_walkers], maxval=n_walkers, dtype=tf.int32))
        z2 = 0.5 * (tf.random.uniform([n_walkers, n_batch], dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, :, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2, extras_proposed2 = _call_log_prob(log_prob, proposed_state2, args, save_extras)

        # acceptance in log space
        log_accept2 = (n_params - 1) * tf.cast(tf.math.log(z2), logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = tf.math.log(tf.random.uniform([n_walkers, n_batch], dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = tf.where(accept2[:, :, None], proposed_state2, current_state2)
        logp_current2 = tf.where(accept2, logp_proposed2, logp_current2)
        if save_extras:
            extras_current2 = tf.where(accept2[:, :, None], extras_proposed2, extras_current2)

        # append to chain
        if keep(step):
            chain.append(tf.concat([current_state1, current_state2], axis=0))
            if save_lp:
                lp_chain.append(tf.concat([logp_current1, logp_current2], axis=0))
            if save_extras:
                extras_chain.append(tf.concat([extras_current1, extras_current2], axis=0))

    # stack up the chain and return
    chain = tf.stack(chain, axis=0)
    returns = (chain,)
    if save_lp:
        returns += (tf.stack(lp_chain, axis=0),)
    if save_extras:
        returns += (tf.stack(extras_chain, axis=0),)
    return returns if len(returns) > 1 else chain
