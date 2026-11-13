"""Exact binomial and hypergeometric probabilities in pure Python.

The binomial model describes records drawn from a process (or a batch much
larger than the sample); the hypergeometric model describes a sample drawn
without replacement from a finite batch with a fixed number of bad records.

Tails are summed term by term away from the mode with the pmf recurrence, so a
small tail is computed directly and never as ``1 - (something close to 1)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

_EXACT_COMB_LIMIT = 5000
_REL_STOP = 1e-17


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    if n <= _EXACT_COMB_LIMIT:
        return math.log(math.comb(n, k))
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _check_binom(n: int, p: float) -> None:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")


def _check_hypergeom(lot: int, bad: int, n: int) -> None:
    if lot < 0:
        raise ValueError("batch size must be non-negative")
    if not 0 <= bad <= lot:
        raise ValueError("number of bad records must be between 0 and the batch size")
    if not 0 <= n <= lot:
        raise ValueError("sample size must be between 0 and the batch size")


def _tail(
    start: int, step: int, end: int, first: float, up_ratio: Callable[[int], float]
) -> float:
    """Sum pmf terms from ``start`` to ``end`` inclusive, moving by ``step``.

    ``up_ratio(j)`` is pmf(j + 1) / pmf(j). The caller guarantees the terms do
    not increase in the direction of travel, which makes the early stop safe.
    """
    term = first
    if term == 0.0:
        return 0.0
    total = term
    j = start
    while j != end:
        if step > 0:
            term *= up_ratio(j)
        else:
            term /= up_ratio(j - 1)
        j += step
        total += term
        if term < total * _REL_STOP:
            break
    return total


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    _check_binom(n, p)
    if k < 0 or k > n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(_log_comb(n, k) + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    _check_binom(n, p)
    if k < 0:
        return 0.0
    if k >= n or p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    odds = p / (1.0 - p)

    def up(j: int) -> float:
        return (n - j) / (j + 1) * odds

    mode = int((n + 1) * p)
    if k < mode:
        return min(1.0, _tail(k, -1, 0, binom_pmf(k, n, p), up))
    return max(0.0, 1.0 - _tail(k + 1, 1, n, binom_pmf(k + 1, n, p), up))


def binom_upper(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    _check_binom(n, p)
    if k <= 0:
        return 1.0
    if k > n or p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    odds = p / (1.0 - p)

    def up(j: int) -> float:
        return (n - j) / (j + 1) * odds

    mode = int((n + 1) * p)
    if k > mode:
        return min(1.0, _tail(k, 1, n, binom_pmf(k, n, p), up))
    return max(0.0, 1.0 - _tail(k - 1, -1, 0, binom_pmf(k - 1, n, p), up))


def hypergeom_support(lot: int, bad: int, n: int) -> tuple[int, int]:
    """Smallest and largest possible number of bad records in the sample."""
    _check_hypergeom(lot, bad, n)
    return max(0, n - (lot - bad)), min(n, bad)


def hypergeom_pmf(k: int, lot: int, bad: int, n: int) -> float:
    """P(X = k) when ``n`` records are drawn from ``lot`` of which ``bad`` are bad."""
    lo, hi = hypergeom_support(lot, bad, n)
    if k < lo or k > hi:
        return 0.0
    if lot <= _EXACT_COMB_LIMIT:
        return math.comb(bad, k) * math.comb(lot - bad, n - k) / math.comb(lot, n)
    return math.exp(_log_comb(bad, k) + _log_comb(lot - bad, n - k) - _log_comb(lot, n))


def _hypergeom_up(lot: int, bad: int, n: int) -> Callable[[int], float]:
    def up(j: int) -> float:
        return (bad - j) * (n - j) / ((j + 1) * (lot - bad - n + j + 1))

    return up


def hypergeom_cdf(k: int, lot: int, bad: int, n: int) -> float:
    """P(X <= k) for the hypergeometric count of bad records in the sample."""
    lo, hi = hypergeom_support(lot, bad, n)
    if k < lo:
        return 0.0
    if k >= hi:
        return 1.0
    up = _hypergeom_up(lot, bad, n)
    mode = (n + 1) * (bad + 1) // (lot + 2)
    if k < mode:
        return min(1.0, _tail(k, -1, lo, hypergeom_pmf(k, lot, bad, n), up))
    return max(0.0, 1.0 - _tail(k + 1, 1, hi, hypergeom_pmf(k + 1, lot, bad, n), up))


def hypergeom_upper(k: int, lot: int, bad: int, n: int) -> float:
    """P(X >= k) for the hypergeometric count of bad records in the sample."""
    lo, hi = hypergeom_support(lot, bad, n)
    if k <= lo:
        return 1.0
    if k > hi:
        return 0.0
    up = _hypergeom_up(lot, bad, n)
    mode = (n + 1) * (bad + 1) // (lot + 2)
    if k > mode:
        return min(1.0, _tail(k, 1, hi, hypergeom_pmf(k, lot, bad, n), up))
    return max(0.0, 1.0 - _tail(k - 1, -1, lo, hypergeom_pmf(k - 1, lot, bad, n), up))
