"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in PyTorch.

Implements the parallel stretch move of Foreman-Mackey et al. (2013): the
ensemble is split into two halves, and each half is updated using partners
drawn from the other half, so that a whole half can be evolved with a single
vectorized call to the target log-probability.

Note: the sampler never needs gradients; if your ``log_prob`` builds an
autograd graph, wrap the call in ``torch.no_grad()`` to save memory.
"""

import torch
from tqdm import trange

from ._utils import keep_step, validate_steps

__all__ = ["sample", "sample_batch"]


def _sanitize(logp):
    """Replace NaN log-probabilities with -inf so those proposals are rejected."""
    if not torch.is_tensor(logp):
        logp = torch.as_tensor(logp)
    return torch.where(torch.isnan(logp), torch.full_like(logp, float("-inf")), logp)


def _make_generator(rng, device):
    """Build a torch.Generator on ``device`` from a seed, or pass one through."""
    if rng is None:
        return None
    if isinstance(rng, torch.Generator):
        return rng
    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng))
    return gen


def _step_half(log_prob, args, moving, fixed, logp_moving, gen, device):
    """Stretch-move update of one half of the ensemble against the other."""
    n_walkers, n_batch, n_params = moving.shape

    idx = torch.randint(0, n_walkers, (n_walkers,), generator=gen, device=device)
    partners = fixed[idx]
    z = 0.5 * (torch.rand((n_walkers, n_batch), generator=gen, dtype=moving.dtype, device=device) + 1.0) ** 2
    proposed = partners + z.unsqueeze(-1) * (moving - partners)

    logp_proposed = _sanitize(log_prob(proposed, *args))

    # accept/reject in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
    log_accept = (n_params - 1) * torch.log(z).to(logp_proposed.dtype) + logp_proposed - logp_moving
    u = torch.rand((n_walkers, n_batch), generator=gen, dtype=logp_proposed.dtype, device=device)
    accept = torch.log(u) < log_accept

    moving = torch.where(accept.unsqueeze(-1), proposed, moving)
    logp_moving = torch.where(accept, logp_proposed, logp_moving)
    return moving, logp_moving


def sample_batch(log_prob, n_steps, current_state, *, args=(), rng=None,
                 n_burnin=0, thin=1, progressbar=True, return_logp=False,
                 device=None):
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
    current_state : tuple of torch.Tensor
        The two walker ensembles ``(walkers1, walkers2)``, each of shape
        ``(n_walkers, n_batch, n_params)``.
    args : tuple, optional
        Extra positional arguments passed to ``log_prob``.
    rng : int or torch.Generator, optional
        Seed or generator for reproducible sampling. Default uses PyTorch's
        global RNG.
    n_burnin : int, optional
        Number of leading steps to discard from the returned chain.
    thin : int, optional
        Store only every ``thin``-th step after burn-in.
    progressbar : bool, optional
        Show a tqdm progress bar.
    return_logp : bool, optional
        Also return the log-probability of every stored sample.
    device : str or torch.device, optional
        Move the walkers (and run the sampler) on this device. Default: the
        device the walkers are already on.

    Returns
    -------
    chain : torch.Tensor
        Shape ``(n_stored, 2*n_walkers, n_batch, n_params)`` where
        ``n_stored = ceil((n_steps - n_burnin) / thin)``.
    logp : torch.Tensor
        Shape ``(n_stored, 2*n_walkers, n_batch)``. Only if ``return_logp``.
    """
    n_stored = validate_steps(n_steps, n_burnin, thin)
    state1, state2 = current_state
    state1 = torch.as_tensor(state1)
    state2 = torch.as_tensor(state2)
    if device is not None:
        state1 = state1.to(device)
        state2 = state2.to(device)
    device = state1.device
    if state1.dim() != 3:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_batch, n_params), got {tuple(state1.shape)}"
        )
    n_walkers, n_batch, n_params = state1.shape
    gen = _make_generator(rng, device)

    logp1 = _sanitize(log_prob(state1, *args))
    logp2 = _sanitize(log_prob(state2, *args))

    # preallocate the (possibly thinned) chain
    chain = torch.empty((n_stored, 2 * n_walkers, n_batch, n_params), dtype=state1.dtype, device=device)
    if return_logp:
        logp_chain = torch.empty((n_stored, 2 * n_walkers, n_batch), dtype=logp1.dtype, device=device)

    stored = 0
    if keep_step(0, n_burnin, thin):
        chain[stored] = torch.cat([state1, state2], dim=0)
        if return_logp:
            logp_chain[stored] = torch.cat([logp1, logp2], dim=0)
        stored += 1

    steps = trange(1, n_steps) if progressbar else range(1, n_steps)
    for step in steps:
        state1, logp1 = _step_half(log_prob, args, state1, state2, logp1, gen, device)
        state2, logp2 = _step_half(log_prob, args, state2, state1, logp2, gen, device)
        if keep_step(step, n_burnin, thin):
            chain[stored] = torch.cat([state1, state2], dim=0)
            if return_logp:
                logp_chain[stored] = torch.cat([logp1, logp2], dim=0)
            stored += 1

    if return_logp:
        return chain, logp_chain
    return chain


def sample(log_prob, n_steps, current_state, *, args=(), rng=None,
           n_burnin=0, thin=1, progressbar=True, return_logp=False,
           device=None):
    """Sample a single posterior with the affine sampler.

    Identical to :func:`sample_batch` but without the batch dimension:
    ``current_state`` holds two ensembles of shape ``(n_walkers, n_params)``,
    ``log_prob`` maps ``(n_walkers, n_params)`` to ``(n_walkers,)``, and the
    returned chain has shape ``(n_stored, 2*n_walkers, n_params)``.
    """
    state1, state2 = current_state
    state1 = torch.as_tensor(state1)
    state2 = torch.as_tensor(state2)
    if state1.dim() != 2:
        raise ValueError(
            f"expected walker states of shape (n_walkers, n_params), got {tuple(state1.shape)}"
        )

    def batched_log_prob(theta, *a):
        logp = log_prob(theta[:, 0, :], *a)
        if not torch.is_tensor(logp):
            logp = torch.as_tensor(logp)
        return logp.unsqueeze(-1)

    result = sample_batch(
        batched_log_prob, n_steps,
        (state1.unsqueeze(1), state2.unsqueeze(1)),
        args=args, rng=rng, n_burnin=n_burnin, thin=thin,
        progressbar=progressbar, return_logp=return_logp, device=device,
    )
    if return_logp:
        chain, logp = result
        return chain[:, :, 0, :], logp[:, :, 0]
    return result[:, :, 0, :]
