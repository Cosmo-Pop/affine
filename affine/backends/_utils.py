"""Framework-independent helpers shared by the sampler backends."""


def validate_steps(n_steps, n_burnin, thin):
    """Validate step settings and return the number of states that will be stored.

    States are stored at steps ``n_burnin, n_burnin + thin, n_burnin + 2*thin, ...``,
    where step 0 is the initial state. With the defaults (``n_burnin=0, thin=1``)
    every step, including the initial state, is stored, so the chain has exactly
    ``n_steps`` entries.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")
    if not 0 <= n_burnin < n_steps:
        raise ValueError(
            f"n_burnin must satisfy 0 <= n_burnin < n_steps, got n_burnin={n_burnin}, n_steps={n_steps}"
        )
    return (n_steps - n_burnin + thin - 1) // thin


def keep_step(step, n_burnin, thin):
    """Whether the state at ``step`` should be stored in the chain."""
    return step >= n_burnin and (step - n_burnin) % thin == 0
