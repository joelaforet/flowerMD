"""Chain-length distributions for polydisperse systems."""

import numpy as np


def schulz_zimm_lengths(mean_length, pdi, n_chains, seed=0, min_length=1):
    """Sample chain lengths from a Schulz-Zimm distribution.

    Returns the two lists `flowermd.base.Polymer` takes as ``lengths`` and
    ``num_mols``: the distinct chain lengths and how many chains have each.

    The Schulz-Zimm (gamma) distribution of degree of polymerization ``N``
    has shape ``k = 1 / (PDI - 1)`` and mean ``mean_length``; ``PDI = 1``
    returns ``n_chains`` chains of ``mean_length``.

    Parameters
    ----------
    mean_length : float, required
        Number-average degree of polymerization.
    pdi : float, required
        Polydispersity index ``M_w / M_n``, at least 1.
    n_chains : int, required
        Number of chains to sample.
    seed : int, default 0
        Random seed.
    min_length : int, default 1
        Shortest allowed chain; samples below it are raised to it.

    Returns
    -------
    lengths : list of int
        Distinct chain lengths, ascending.
    num_mols : list of int
        Number of chains of each length, same order.

    Examples
    --------
    ::

        lengths, num_mols = schulz_zimm_lengths(mean_length=50, pdi=1.5, n_chains=40)
        chains = PolyEthylene(lengths=lengths, num_mols=num_mols)

    """
    if mean_length <= 0:
        raise ValueError("mean_length must be positive.")
    if pdi < 1.0:
        raise ValueError("pdi cannot be below 1.")
    if n_chains < 1:
        raise ValueError("n_chains must be at least 1.")
    if pdi == 1.0:
        samples = np.full(n_chains, round(mean_length), dtype=int)
    else:
        rng = np.random.default_rng(seed)
        shape = 1.0 / (pdi - 1.0)
        scale = mean_length / shape
        samples = np.rint(rng.gamma(shape, scale, size=n_chains)).astype(int)
    samples = np.maximum(samples, min_length)
    lengths, counts = np.unique(samples, return_counts=True)
    return [int(x) for x in lengths], [int(c) for c in counts]
