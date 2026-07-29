"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in JAX.

Implements the parallel stretch move of Foreman-Mackey et al. (2013): the
ensemble is split into two halves, and each half is updated using partners
drawn from the other half, so that a whole half can be evolved with a single
vectorized call to the target log-probability.

JAX's random numbers are functional, so both samplers require an explicit
PRNG key via the ``rng`` argument, e.g. ``rng=jax.random.PRNGKey(0)``.
"""

import jax.numpy as jnp
import jax.random as jr
from tqdm import trange

from ._utils import keep_step, validate_steps

__all__ = ["sample", "sample_batch"]


def _sanitize(logp):
    """Replace NaN log-probabilities with -inf so those proposals are rejected."""
    logp = jnp.asarray(logp)
    return jnp.where(jnp.isnan(logp), -jnp.inf, logp)


def _step_half(rng, log_prob, args, moving, fixed, logp_moving):
    """Stretch-move update of one half of the ensemble against the other."""
    n_walkers, n_batch, n_params = moving.shape
    k_idx, k_z, k_u = jr.split(rng, 3)

    partners = fixed[jr.randint(k_idx, (n_walkers,), 0, n_walkers)]
    z = 0.5 * (jr.uniform(k_z, (n_walkers, n_batch), dtype=moving.dtype) + 1.0) ** 2
    proposed = partners + z[..., None] * (moving - partners)

    logp_proposed = _sanitize(log_prob(proposed, *args))

    # accept/reject in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
    log_accept = (n_params - 1) * jnp.log(z).astype(logp_proposed.dtype) + logp_proposed - logp_moving
    u = jr.uniform(k_u, (n_walkers, n_batch), dtype=logp_proposed.dtype)
    accept = jnp.log(u) < log_accept

    moving = jnp.where(accept[..., None], proposed, moving)
    logp_moving = jnp.where(accept, logp_proposed, logp_moving)
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
    current_state : tuple of arrays
        The two walker ensembles ``(walkers1, walkers2)``, each of shape
        ``(n_walkers, n_batch, n_params)``.
    args : tuple, optional
        Extra positional arguments passed to ``log_prob``.
    rng : jax PRNG key
        Required, e.g. ``jax.random.PRNGKey(0)``.
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
    chain : jnp.ndarray
        Shape ``(n_stored, 2*n_walkers, n_batch, n_params)`` where
        ``n_stored = ceil((n_steps - n_burnin) / thin)``.
    logp : jnp.ndarray
        Shape ``(n_stored, 2*n_walkers, n_batch)``. Only if ``return_logp``.
    """
    if rng is None:
        raise ValueError(
            "the jax backend needs an explicit PRNG key: pass rng=jax.random.PRNGKey(<seed>)"
        )
    validate_steps(n_steps, n_burnin, thin)
    state1, state2 = current_state
    state1 = jnp.asarray(state1)
    state2 = jnp.asarray(state2)
    if state1.ndim != 3:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_batch, n_params), got {state1.shape}"
        )

    logp1 = _sanitize(log_prob(state1, *args))
    logp2 = _sanitize(log_prob(state2, *args))

    chain, logp_chain = [], []
    if keep_step(0, n_burnin, thin):
        chain.append(jnp.concatenate([state1, state2], axis=0))
        logp_chain.append(jnp.concatenate([logp1, logp2], axis=0))

    steps = trange(1, n_steps) if progressbar else range(1, n_steps)
    for step in steps:
        rng, k1, k2 = jr.split(rng, 3)
        state1, logp1 = _step_half(k1, log_prob, args, state1, state2, logp1)
        state2, logp2 = _step_half(k2, log_prob, args, state2, state1, logp2)
        if keep_step(step, n_burnin, thin):
            chain.append(jnp.concatenate([state1, state2], axis=0))
            logp_chain.append(jnp.concatenate([logp1, logp2], axis=0))

    chain = jnp.stack(chain, axis=0)
    if return_logp:
        return chain, jnp.stack(logp_chain, axis=0)
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
    state1 = jnp.asarray(state1)
    state2 = jnp.asarray(state2)
    if state1.ndim != 2:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_params), got {state1.shape}"
        )

    def batched_log_prob(theta, *a):
        return jnp.asarray(log_prob(theta[:, 0, :], *a))[..., None]

    result = sample_batch(
        batched_log_prob, n_steps,
        (state1[:, None, :], state2[:, None, :]),
        args=args, rng=rng, n_burnin=n_burnin, thin=thin,
        progressbar=progressbar, return_logp=return_logp,
    )
    if return_logp:
        chain, logp = result
        return chain[:, :, 0, :], logp[:, :, 0]
    return result[:, :, 0, :]
