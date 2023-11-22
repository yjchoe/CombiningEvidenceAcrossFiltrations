"""
data generators for exchangeability
"""


import numpy as np


def generate_binary_iid(
        p: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate an i.i.d. Bernoulli(p) sequence."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    return rng.binomial(1, p=p, size=size)


def generate_binary_exch_noniid(
        p: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate an exchangeable but non-i.i.d. sequence."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    b0 = rng.binomial(1, p=p)
    b = rng.binomial(1, p=p, size=size)
    return np.minimum(b0, b)  # b0 AND bt


def generate_binary_changepoint(
        p: float,
        q: float,
        size: int,
        rng: np.random.Generator = None,
):
    """Generate a Bernoulli sequence with a changepoint in the middle,
    that is, Ber(p) for [size/2] and Ber(q) for the next [size/2]."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    half = size // 2
    return np.concatenate([
        rng.binomial(1, p=p, size=half),
        rng.binomial(1, p=q, size=size - half),
    ])


def generate_binary_markov(
        p_10: float,
        p_11: float,
        size: int,
        p_init: float = 0.5,
        rng: np.random.Generator = None,
):
    """Generate a binary sequence from a first-order Markov process.

    `p_kj` is the transition probability from state `j` to state `k`.
    The resulting sequence is NOT exchangeable."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    probs = np.array([p_10, p_11])

    seq = np.zeros(size, dtype=int)
    seq[0] = rng.binomial(1, p=p_init)
    for t in range(1, size):
        seq[t] = rng.binomial(1, p=probs[seq[t - 1]])
    return seq


def generate_ar(order: int, size: int, rng: np.random.Generator = None):
    """Generate a sequence from an AR(order) process.

    This is an example for a _continuous_ process.
    The resulting sequence is NOT exchangeable."""
    raise NotImplementedError


def generate_urn(n, m, rng: np.random.Generator = None):
    """Generate a sequence of marbles from an urn.

    This is an example for a _non-binary but discrete_ process.
    The resulting sequence is exchangeable (but not i.i.d.).
    """
    raise NotImplementedError
