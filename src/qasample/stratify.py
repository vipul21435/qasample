"""Stratified audits: split a sample across strata and combine what comes back."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

from .bounds import hypergeom_bound

METHODS = ("proportional", "neyman")


def allocate(
    sizes: Mapping[str, int],
    total: int,
    method: str = "proportional",
    prior_rates: Mapping[str, float] | None = None,
    minimum: int = 1,
) -> dict[str, int]:
    """Split ``total`` inspections across strata.

    ``proportional`` gives each stratum a share proportional to its size N_h.
    ``neyman`` uses N_h * sqrt(p_h * (1 - p_h)) with the prior error rates
    p_h, which minimises the variance of the combined estimate for a fixed
    total. Every stratum gets at least ``minimum`` records (or all of its
    records if it is smaller) and never more than it holds; shares are
    rounded by largest remainder so that they add up to ``total``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not sizes:
        raise ValueError("no strata given")
    if any(size < 0 for size in sizes.values()):
        raise ValueError("stratum sizes must be non-negative")
    if minimum < 0 or total < 0:
        raise ValueError("total and minimum must be non-negative")
    names = list(sizes)
    lower = {h: min(minimum, sizes[h]) for h in names}
    if total > sum(sizes.values()):
        raise ValueError("the sample is larger than the batch")
    if total < sum(lower.values()):
        raise ValueError("the sample is too small to give every stratum its minimum")
    if method == "neyman":
        if prior_rates is None:
            raise ValueError("neyman allocation needs prior error rates")
        missing = [h for h in names if h not in prior_rates]
        if missing:
            raise ValueError(f"no prior rate for strata: {', '.join(missing)}")
        for h in names:
            if not 0.0 <= prior_rates[h] <= 1.0:
                raise ValueError("prior rates must be in [0, 1]")
        weight = {h: sizes[h] * math.sqrt(prior_rates[h] * (1.0 - prior_rates[h])) for h in names}
        if sum(weight.values()) == 0.0:
            weight = {h: float(sizes[h]) for h in names}
    else:
        weight = {h: float(sizes[h]) for h in names}

    # constrained proportional shares: clamp to [lower, size] and re-share the rest
    share: dict[str, float] = {}
    free = set(names)
    budget = float(total)
    while free:
        wsum = sum(weight[h] for h in free)
        if wsum <= 0.0:
            ideal = {h: budget / len(free) for h in free}
        else:
            ideal = {h: budget * weight[h] / wsum for h in free}
        low = [h for h in free if ideal[h] < lower[h]]
        high = [h for h in free if ideal[h] > sizes[h]]
        if low:
            clamp = {h: float(lower[h]) for h in low}
        elif high:
            clamp = {h: float(sizes[h]) for h in high}
        else:
            share.update(ideal)
            break
        for h, value in clamp.items():
            share[h] = value
            budget -= value
            free.discard(h)

    out = {h: min(sizes[h], max(lower[h], math.floor(share[h] + 1e-9))) for h in names}
    left = total - sum(out.values())
    order = sorted(names, key=lambda h: (-(share[h] - math.floor(share[h] + 1e-9)), names.index(h)))
    while left > 0:
        moved = False
        for h in order:
            if left > 0 and out[h] < sizes[h]:
                out[h] += 1
                left -= 1
                moved = True
        if not moved:  # pragma: no cover - total <= sum(sizes) guarantees room
            break
    return out


@dataclass(frozen=True)
class StratumSample:
    """One stratum: ``size`` records in the batch, ``n`` inspected, ``bad`` found bad."""

    name: str
    size: int
    n: int
    bad: int

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"stratum {self.name}: at least one record must be inspected")
        if self.n > self.size:
            raise ValueError(f"stratum {self.name}: the sample is larger than the stratum")
        if not 0 <= self.bad <= self.n:
            raise ValueError(f"stratum {self.name}: bad must be between 0 and n")


@dataclass(frozen=True)
class StratifiedEstimate:
    rate: float
    std_error: float
    normal_interval: tuple[float, float]
    exact_interval: tuple[float, float]
    bad_bounds: tuple[int, int]
    confidence: float
    strata: tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "rate": self.rate,
            "std_error": self.std_error,
            "normal_interval": list(self.normal_interval),
            "exact_interval": list(self.exact_interval),
            "bad_bounds": list(self.bad_bounds),
            "confidence": self.confidence,
            "strata": list(self.strata),
        }


def stratified_estimate(
    strata: Sequence[StratumSample], confidence: float = 0.95
) -> StratifiedEstimate:
    """Combined error rate of the batch from per-stratum samples.

    The estimate is sum(W_h * p_h) with W_h = N_h / N. Two intervals are
    returned. ``normal_interval`` uses the usual variance estimate
    sum(W_h^2 * (1 - n_h / N_h) * p_h * (1 - p_h) / (n_h - 1)); it is
    approximate and collapses to a point when no stratum shows a mixed
    sample. ``exact_interval`` adds up exact hypergeometric bounds on the bad
    records of each stratum, each taken at confidence 1 - alpha / H so that
    all H hold together with probability at least ``confidence``; it is
    conservative (wider than needed) but its coverage is guaranteed.
    """
    if not strata:
        raise ValueError("no strata given")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    total = sum(s.size for s in strata)
    rate = 0.0
    variance = 0.0
    per_level = 1.0 - (1.0 - confidence) / len(strata)
    low_bad = high_bad = 0
    rows = []
    for s in strata:
        w = s.size / total
        p = s.bad / s.n
        rate += w * p
        variance += w * w * (1.0 - s.n / s.size) * p * (1.0 - p) / max(s.n - 1, 1)
        lo, hi = hypergeom_bound(s.bad, s.n, s.size, per_level)
        low_bad += lo
        high_bad += hi
        rows.append(
            {"name": s.name, "size": s.size, "n": s.n, "bad": s.bad, "rate": p,
             "bad_low": lo, "bad_high": hi}
        )
    se = math.sqrt(variance)
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return StratifiedEstimate(
        rate=rate,
        std_error=se,
        normal_interval=(max(0.0, rate - z * se), min(1.0, rate + z * se)),
        exact_interval=(low_bad / total, high_bad / total),
        bad_bounds=(low_bad, high_bad),
        confidence=confidence,
        strata=tuple(rows),
    )
