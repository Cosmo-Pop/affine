"""Affine-invariant ensemble MCMC (Goodman & Weare 2010) in PyTorch.

Uses the parallel stretch move of Foreman-Mackey et al. (2013): the ensemble
is split into two halves and each half is updated with partners drawn from
the other half, so that a whole half is evolved with a single vectorized call
to the target log-probability.

For reproducible chains call ``torch.manual_seed(<seed>)`` before sampling.
The sampler never needs gradients; if your ``log_prob`` builds an autograd
graph, call the sampler under ``torch.no_grad()`` to save memory.
"""

import torch
from tqdm import trange

__all__ = ["sample", "sample_batch"]


def _nan_to_neginf(logp):
    # treat NaN log probabilities as -inf so those proposals are rejected
    if not torch.is_tensor(logp):
        logp = torch.as_tensor(logp)
    return torch.where(torch.isnan(logp), torch.full_like(logp, float("-inf")), logp)


def sample(log_prob, n_params, n_walkers, n_steps, walkers1, walkers2,
           progress=True, save_lp=False):
    """Sample a single posterior.

    Parameters
    ----------
    log_prob : callable
        Vectorized target log-probability: maps parameters of shape
        ``(n_walkers, n_params)`` to log-probabilities of shape
        ``(n_walkers,)``. NaN values are treated as -inf (proposal rejected).
    n_params, n_walkers : int
        Number of parameters, and of walkers per ensemble (the shapes of
        ``walkers1``/``walkers2`` are checked against these).
    n_steps : int
        Number of steps; the initial state counts as step 0, so the returned
        chain has ``n_steps`` entries.
    walkers1, walkers2 : torch.Tensor
        Initial positions of the two ensembles, each of shape
        ``(n_walkers, n_params)``. The sampler runs on whatever device and
        dtype the walkers are on.
    progress : bool, optional
        Show a tqdm progress bar.
    save_lp : bool, optional
        Also return the log-probability of every stored sample.

    Returns
    -------
    chain : torch.Tensor of shape ``(n_steps, 2 * n_walkers, n_params)``
    lp_chain : torch.Tensor of shape ``(n_steps, 2 * n_walkers)``
        Only returned if ``save_lp=True``.
    """
    current_state1 = torch.as_tensor(walkers1)
    current_state2 = torch.as_tensor(walkers2)
    for w in (current_state1, current_state2):
        if tuple(w.shape) != (n_walkers, n_params):
            raise ValueError(
                f"walkers must have shape (n_walkers, n_params) = ({n_walkers}, {n_params}), "
                f"got {tuple(w.shape)}")
    device, dtype = current_state1.device, current_state1.dtype

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1))
    logp_current2 = _nan_to_neginf(log_prob(current_state2))

    # holder for the whole chain, starting with the initial state
    chain = [torch.cat([current_state1, current_state2], dim=0)]
    if save_lp:
        lp_chain = [torch.cat([logp_current1, logp_current2], dim=0)]

    # progress bar?
    loop = trange if progress else range

    # MCMC loop
    for _ in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = current_state2[torch.randint(0, n_walkers, (n_walkers,), device=device)]
        z1 = 0.5 * (torch.rand(n_walkers, dtype=dtype, device=device) + 1) ** 2
        proposed_state1 = partners1 + z1[:, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * torch.log(z1).to(logp_proposed1.dtype) + logp_proposed1 - logp_current1
        u1 = torch.rand(n_walkers, dtype=logp_proposed1.dtype, device=device)
        accept1 = torch.log(u1) < log_accept1

        # update the state
        current_state1 = torch.where(accept1[:, None], proposed_state1, current_state1)
        logp_current1 = torch.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = current_state1[torch.randint(0, n_walkers, (n_walkers,), device=device)]
        z2 = 0.5 * (torch.rand(n_walkers, dtype=dtype, device=device) + 1) ** 2
        proposed_state2 = partners2 + z2[:, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2))

        # acceptance in log space
        log_accept2 = (n_params - 1) * torch.log(z2).to(logp_proposed2.dtype) + logp_proposed2 - logp_current2
        u2 = torch.rand(n_walkers, dtype=logp_proposed2.dtype, device=device)
        accept2 = torch.log(u2) < log_accept2

        # update the state
        current_state2 = torch.where(accept2[:, None], proposed_state2, current_state2)
        logp_current2 = torch.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        chain.append(torch.cat([current_state1, current_state2], dim=0))
        if save_lp:
            lp_chain.append(torch.cat([logp_current1, logp_current2], dim=0))

    # stack up the chain and return
    chain = torch.stack(chain, dim=0)
    if save_lp:
        return chain, torch.stack(lp_chain, dim=0)
    return chain


def sample_batch(log_prob, n_steps, current_state, n_burnin=0, thin=1, args=[],
                 progress=True, save_lp=False, device=None):
    """Sample a batch of independent posteriors simultaneously.

    Same as :func:`sample`, but with a batch dimension: walker states have
    shape ``(n_walkers, n_batch, n_params)`` and ``log_prob`` must return
    shape ``(n_walkers, n_batch)``.

    Parameters
    ----------
    log_prob : callable
        Vectorized target log-probability (NaN treated as -inf).
    n_steps : int
        Number of steps; the initial state counts as step 0.
    current_state : pair of torch.Tensor
        The two walker ensembles, each ``(n_walkers, n_batch, n_params)``.
    n_burnin : int, optional
        Discard the first ``n_burnin`` steps from the returned chain.
    thin : int, optional
        Store only every ``thin``-th step after burn-in.
    args : list, optional
        Extra arguments passed to ``log_prob``.
    progress : bool, optional
        Show a tqdm progress bar.
    save_lp : bool, optional
        Also return the log-probability of every stored sample.
    device : str or torch.device, optional
        Move the walkers to this device first. Default: run on whatever
        device the walkers are already on.

    Returns
    -------
    chain : torch.Tensor of shape ``(n_stored, 2 * n_walkers, n_batch, n_params)``
        where ``n_stored = ceil((n_steps - n_burnin) / thin)``.
    lp_chain : torch.Tensor of shape ``(n_stored, 2 * n_walkers, n_batch)``
        Only returned if ``save_lp=True``.
    """
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if not 0 <= n_burnin < n_steps:
        raise ValueError(f"need 0 <= n_burnin < n_steps, got n_burnin={n_burnin}, n_steps={n_steps}")

    # split the current state (moving it to the requested device if any)
    current_state1 = torch.as_tensor(current_state[0])
    current_state2 = torch.as_tensor(current_state[1])
    if device is not None:
        current_state1 = current_state1.to(device)
        current_state2 = current_state2.to(device)
    device, dtype = current_state1.device, current_state1.dtype

    # pull out the number of walkers, batch size, and parameters
    n_walkers, n_batch, n_params = current_state1.shape

    # initial target log prob for the walkers
    logp_current1 = _nan_to_neginf(log_prob(current_state1, *args))
    logp_current2 = _nan_to_neginf(log_prob(current_state2, *args))

    # the state at steps n_burnin, n_burnin + thin, ... is stored
    def keep(step):
        return step >= n_burnin and (step - n_burnin) % thin == 0

    chain, lp_chain = [], []
    if keep(0):
        chain.append(torch.cat([current_state1, current_state2], dim=0))
        if save_lp:
            lp_chain.append(torch.cat([logp_current1, logp_current2], dim=0))

    # progress bar?
    loop = trange if progress else range

    # MCMC loop
    for step in loop(1, n_steps):

        # first set of walkers:

        # proposals
        partners1 = current_state2[torch.randint(0, n_walkers, (n_walkers,), device=device)]
        z1 = 0.5 * (torch.rand((n_walkers, n_batch), dtype=dtype, device=device) + 1) ** 2
        proposed_state1 = partners1 + z1[:, :, None] * (current_state1 - partners1)

        # target log prob at proposed points
        logp_proposed1 = _nan_to_neginf(log_prob(proposed_state1, *args))

        # acceptance in log space: log u < (d - 1) log z + log p(proposed) - log p(current)
        log_accept1 = (n_params - 1) * torch.log(z1).to(logp_proposed1.dtype) + logp_proposed1 - logp_current1
        u1 = torch.rand((n_walkers, n_batch), dtype=logp_proposed1.dtype, device=device)
        accept1 = torch.log(u1) < log_accept1

        # update the state
        current_state1 = torch.where(accept1[:, :, None], proposed_state1, current_state1)
        logp_current1 = torch.where(accept1, logp_proposed1, logp_current1)

        # second set of walkers:

        # proposals (partners drawn from the updated first set)
        partners2 = current_state1[torch.randint(0, n_walkers, (n_walkers,), device=device)]
        z2 = 0.5 * (torch.rand((n_walkers, n_batch), dtype=dtype, device=device) + 1) ** 2
        proposed_state2 = partners2 + z2[:, :, None] * (current_state2 - partners2)

        # target log prob at proposed points
        logp_proposed2 = _nan_to_neginf(log_prob(proposed_state2, *args))

        # acceptance in log space
        log_accept2 = (n_params - 1) * torch.log(z2).to(logp_proposed2.dtype) + logp_proposed2 - logp_current2
        u2 = torch.rand((n_walkers, n_batch), dtype=logp_proposed2.dtype, device=device)
        accept2 = torch.log(u2) < log_accept2

        # update the state
        current_state2 = torch.where(accept2[:, :, None], proposed_state2, current_state2)
        logp_current2 = torch.where(accept2, logp_proposed2, logp_current2)

        # append to chain
        if keep(step):
            chain.append(torch.cat([current_state1, current_state2], dim=0))
            if save_lp:
                lp_chain.append(torch.cat([logp_current1, logp_current2], dim=0))

    # stack up the chain and return
    chain = torch.stack(chain, dim=0)
    if save_lp:
        return chain, torch.stack(lp_chain, dim=0)
    return chain
