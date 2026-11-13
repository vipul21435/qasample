"""What an inspected sample says about the batch it came from."""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist

from .dist import binom_cdf, binom_upper, hypergeom_cdf, hypergeom_upper
from .single import bad_count_at_most

SIDES = ("two-sided", "upper", "lower")


def _check_sample(bad: int, n: int) -> None:
    if n < 1:
        raise ValueError("the sample must hold at least one record")
    if not 0 <= bad <= n:
        raise ValueError("the number of bad records must be between 0 and the sample size")


def _tail_alpha(confidence: float, side: str) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    return (1.0 - confidence) / 2.0 if side == "two-sided" else 1.0 - confidence


def _bisect(f: Callable[[float], float], lo: float, hi: float) -> float:
    """Root of an increasing function f on [lo, hi] with f(lo) <= 0 <= f(hi)."""
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson(
    bad: int, n: int, confidence: float = 0.95, side: str = "two-sided"
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) bounds on the error rate from ``bad`` of ``n`` records.

    ``side="upper"`` returns ``(0, U)`` with a one-sided upper bound,
    ``side="lower"`` returns ``(L, 1)``; two-sided splits the risk equally.
    """
    _check_sample(bad, n)
    alpha = _tail_alpha(confidence, side)
    lower, upper = 0.0, 1.0
    if side != "upper" and bad > 0:
        if bad == n:
            lower = alpha ** (1.0 / n)
        else:
            lower = _bisect(lambda p: binom_upper(bad, n, p) - alpha, 0.0, 1.0)
    if side != "lower" and bad < n:
        if bad == 0:
            upper = 1.0 - alpha ** (1.0 / n)
        else:
            upper = _bisect(lambda p: alpha - binom_cdf(bad, n, p), 0.0, 1.0)
    return lower, upper


def wilson(bad: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval (two-sided, approximate) for the error rate."""
    _check_sample(bad, n)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = bad / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def rule_of_three(n: int, confidence: float = 0.95) -> dict[str, float]:
    """Upper bound on the error rate when ``n`` records were inspected and none was bad.

    ``exact`` is ``1 - (1 - confidence) ** (1 / n)``; ``approx`` is
    ``-ln(1 - confidence) / n``, which is the familiar ``3 / n`` at 95 %.
    """
    if n < 1:
        raise ValueError("the sample must hold at least one record")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    alpha = 1.0 - confidence
    return {"exact": 1.0 - alpha ** (1.0 / n), "approx": min(1.0, -math.log(alpha) / n)}


def zero_defect_sample_size(rate: float, confidence: float = 0.95) -> int:
    """Smallest clean sample that puts the upper bound on the error rate at or below ``rate``."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must be strictly between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    n = max(1, math.ceil(math.log(1.0 - confidence) / math.log1p(-rate) - 1e-12))
    while (1.0 - rate) ** n > 1.0 - confidence:
        n += 1
    return n


def hypergeom_bound(
    bad: int, n: int, lot_size: int, confidence: float = 0.95, side: str = "two-sided"
) -> tuple[int, int]:
    """Exact bounds on the number of bad records in a finite batch.

    ``bad`` of the ``n`` sampled records were bad; the batch holds ``lot_size``
    records. The bounds are found by inverting the hypergeometric tail tests:
    the upper bound is the largest total D for which seeing ``bad`` or fewer
    is not rarer than the tail risk, and the lower bound likewise. Subtract
    ``bad`` from both to bound the bad records left among the uninspected.
    """
    _check_sample(bad, n)
    if n > lot_size:
        raise ValueError("the sample is larger than the batch")
    alpha = _tail_alpha(confidence, side)
    d_min, d_max = bad, lot_size - (n - bad)
    lower, upper = d_min, d_max
    if side != "lower":
        # P(X <= bad | D) falls as D grows: find the largest D that keeps it above alpha
        lo, hi = d_min, d_max
        if hypergeom_cdf(bad, lot_size, hi, n) > alpha:
            upper = hi
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if hypergeom_cdf(bad, lot_size, mid, n) > alpha:
                    lo = mid
                else:
                    hi = mid
            upper = lo
    if side != "upper":
        # P(X >= bad | D) rises with D: find the smallest D that brings it above alpha
        lo, hi = d_min, d_max
        if hypergeom_upper(bad, lot_size, lo, n) > alpha:
            lower = lo
        else:
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if hypergeom_upper(bad, lot_size, mid, n) > alpha:
                    hi = mid
                else:
                    lo = mid
            lower = hi
    return lower, upper


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz iteration)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 100_000):
        m2 = 2 * m
        for aa in (
            m * (b - m) * x / ((qam + m2) * (a + m2)),
            -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)),
        ):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            delta = d * c
            h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b), the Beta(a, b) cdf at x."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return min(1.0, math.exp(log_front) * _betacf(a, b, x) / a)
    return max(0.0, 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def betabinom_cdf(k: int, m: int, a: float, b: float) -> float:
    """P(R <= k) for R ~ BetaBinomial(m, a, b)."""
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    base = _log_beta(a, b)
    total = 0.0
    for j in range(k + 1):
        log_comb = math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1)
        total += math.exp(log_comb + _log_beta(j + a, m - j + b) - base)
    return min(1.0, total)


def prob_meets_target(
    bad: int,
    n: int,
    target: float,
    lot_size: int | None = None,
    prior: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Posterior probability that the batch error rate is at most ``target``.

    The error rate gets a Beta(``prior``) prior (uniform by default), so after
    the sample it is Beta(a + bad, b + n - bad). Without ``lot_size`` the
    answer is that distribution's cdf at ``target``. With ``lot_size`` the
    bad records among the uninspected ones follow a beta-binomial
    distribution and the answer is the probability that the whole batch,
    sampled part included, holds at most ``floor(lot_size * target)`` bad
    records.
    """
    _check_sample(bad, n)
    if not 0.0 <= target <= 1.0:
        raise ValueError("target must be in [0, 1]")
    a, b = prior
    if a <= 0.0 or b <= 0.0:
        raise ValueError("prior parameters must be positive")
    post_a, post_b = a + bad, b + (n - bad)
    if lot_size is None:
        return betainc(post_a, post_b, target)
    if n > lot_size:
        raise ValueError("the sample is larger than the batch")
    allowed = bad_count_at_most(lot_size, target) - bad
    if allowed < 0:
        return 0.0
    return betabinom_cdf(allowed, lot_size - n, post_a, post_b)
