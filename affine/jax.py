"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in JAX.

Uses the parallel stretch move of Foreman-Mackey et al. (2013): the ensemble
is split into two halves and each half is updated with partners drawn from
the other half, so that a whole half is evolved with a single vectorized call
to the target log-probability.

JAX random numbers are functional, so the samplers take an explicit PRNG key
as their first argument, e.g. ``sample(jax.random.PRNGKey(0), ...)``.
"""

import jax.numpy as jnp
import jax.random as jr
from tqdm import trange

__all__ = ["sample", "sample_batch"]


def _nan_to_neginf(logp):
    # treat NaN log probabilities as -inf so those proposals are rejected
    logp = jnp.asarray(logp)
    return jnp.where(jnp.isnan(logp), -jnp.inf, logp)


def sample(rng, log_prob, n_steps, current_state, progressbar=True, args=()):
    """Sample a single posterior.

    Parameters
    ----------
    rng : jax PRNG key
        E.g. ``jax.random.PRNGKey(0)``. Chains are exactly reproducible for
        a given key.
    log_prob : callable
        Vectorized target log-probability: maps parameters of shape
        ``(n_walkers, n_params)`` to log-probabilities of shape
        ``(n_walkers,)``. NaN values are treated as -inf (proposal rejected).
    n_steps : int
        Number of steps; the initial state counts as step 0, so the returned
        chain has ``n_steps`` entries.
    current_state : pair of arrays
        The two walker ensembles, each of shape ``(n_walkers, n_params)``.
    progressbar : bool, optional
        Show a tqdm progress bar.
    args : tuple, optional
        Extra arguments passed to ``log_prob``.

    Returns
    -------
    jnp.ndarray of shape ``(n_steps, 2 * n_walkers, n_params)``
    """
    # split the current state
    current_state1 = jnp.asarray(current_state[0])
    current_state2 = jnp.asarray(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers and parameters
    n_walkers, n_params = current_state1.shape

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1, *args))
    logp_current2 = _nan_to_neginf(log_prob(current_state2, *args))

    # holder for the whole chain, starting with the initial state
    chain = [jnp.concatenate([current_state1, current_state2], axis=0)]

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # fresh keys for every random draw this step
        rng, k_partners1, k_z1, k_u1, k_partners2, k_z2, k_u2 = jr.split(rng, 7)

        # first set of walkers:

        # proposals
        partners1 = current_state2[jr.randint(k_partners1, (n_walkers,), 0, n_walkers)]
        z1 = 0.5 * (jr.uniform(k_z1, (n_walkers,), dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1, *args))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * jnp.log(z1).astype(logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = jnp.log(jr.uniform(k_u1, (n_walkers,), dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = jnp.where(accept1[:, None], proposed_state1, current_state1)
        logp_current1 = jnp.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = current_state1[jr.randint(k_partners2, (n_walkers,), 0, n_walkers)]
        z2 = 0.5 * (jr.uniform(k_z2, (n_walkers,), dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2, *args))

        # acceptance in log space
        log_accept2 = (n_params - 1) * jnp.log(z2).astype(logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = jnp.log(jr.uniform(k_u2, (n_walkers,), dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = jnp.where(accept2[:, None], proposed_state2, current_state2)
        logp_current2 = jnp.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        chain.append(jnp.concatenate([current_state1, current_state2], axis=0))

    # stack up the chain and return
    return jnp.stack(chain, axis=0)


def sample_batch(rng, log_prob, n_steps, current_state, progressbar=True, args=()):
    """Sample a batch of independent posteriors simultaneously.

    Same as :func:`sample`, but with an extra batch dimension: walker states
    have shape ``(n_walkers, n_batch, n_params)`` and ``log_prob`` must
    return shape ``(n_walkers, n_batch)``.

    Returns
    -------
    jnp.ndarray of shape ``(n_steps, 2 * n_walkers, n_batch, n_params)``
    """
    # split the current state
    current_state1 = jnp.asarray(current_state[0])
    current_state2 = jnp.asarray(current_state[1])
    dtype = current_state1.dtype

    # pull out the number of walkers, batch size, and parameters
    n_walkers, n_batch, n_params = current_state1.shape

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1, *args))
    logp_current2 = _nan_to_neginf(log_prob(current_state2, *args))

    # holder for the whole chain, starting with the initial state
    chain = [jnp.concatenate([current_state1, current_state2], axis=0)]

    # progress bar?
    loop = trange if progressbar else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # fresh keys for every random draw this step
        rng, k_partners1, k_z1, k_u1, k_partners2, k_z2, k_u2 = jr.split(rng, 7)

        # first set of walkers:

        # proposals
        partners1 = current_state2[jr.randint(k_partners1, (n_walkers,), 0, n_walkers)]
        z1 = 0.5 * (jr.uniform(k_z1, (n_walkers, n_batch), dtype=dtype) + 1) ** 2
        proposed_state1 = partners1 + z1[:, :, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1, *args))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * jnp.log(z1).astype(logp_proposed1.dtype) + logp_proposed1 - logp_current1
        accept1 = jnp.log(jr.uniform(k_u1, (n_walkers, n_batch), dtype=logp_proposed1.dtype)) < log_accept1

        # update the state
        current_state1 = jnp.where(accept1[:, :, None], proposed_state1, current_state1)
        logp_current1 = jnp.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = current_state1[jr.randint(k_partners2, (n_walkers,), 0, n_walkers)]
        z2 = 0.5 * (jr.uniform(k_z2, (n_walkers, n_batch), dtype=dtype) + 1) ** 2
        proposed_state2 = partners2 + z2[:, :, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2, *args))

        # acceptance in log space
        log_accept2 = (n_params - 1) * jnp.log(z2).astype(logp_proposed2.dtype) + logp_proposed2 - logp_current2
        accept2 = jnp.log(jr.uniform(k_u2, (n_walkers, n_batch), dtype=logp_proposed2.dtype)) < log_accept2

        # update the state
        current_state2 = jnp.where(accept2[:, :, None], proposed_state2, current_state2)
        logp_current2 = jnp.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        chain.append(jnp.concatenate([current_state1, current_state2], axis=0))

    # stack up the chain and return
    return jnp.stack(chain, axis=0)
