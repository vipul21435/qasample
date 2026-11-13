"""Single sampling plans: inspect n records, accept the batch if at most c are bad."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .dist import binom_cdf, hypergeom_cdf, hypergeom_pmf, hypergeom_support

_GUARD = 1e-9
_MAX_ACCEPT_NUMBER = 100_000
_MAX_SAMPLE = 10**9


def bad_count_at_most(lot_size: int, rate: float) -> int:
    """Largest number of bad records a batch can hold with an error rate <= ``rate``."""
    return min(lot_size, math.floor(lot_size * rate + _GUARD))


def bad_count_at_least(lot_size: int, rate: float) -> int:
    """Smallest number of bad records a batch can hold with an error rate >= ``rate``."""
    return max(0, math.ceil(lot_size * rate - _GUARD))


@dataclass(frozen=True)
class SinglePlan:
    """Inspect ``n`` records; accept if the number found bad is at most ``c``."""

    n: int
    c: int

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be non-negative")
        if self.c < 0:
            raise ValueError("c must be non-negative")

    def accept_prob(self, rate: float) -> float:
        """Probability of acceptance when each record is bad with probability ``rate``."""
        return binom_cdf(self.c, self.n, rate)

    def accept_prob_lot(self, lot_size: int, bad: int) -> float:
        """Probability of acceptance for a batch of ``lot_size`` holding ``bad`` bad records."""
        if self.n > lot_size:
            raise ValueError("the sample is larger than the batch")
        return hypergeom_cdf(self.c, lot_size, bad, self.n)

    def aoq(self, rate: float, lot_size: int | None = None) -> float:
        """Average outgoing error rate under rectification, binomial model.

        Rejected batches are inspected in full and every bad record found is
        replaced by a good one, so only accepted batches pass errors on, and
        only in the part that was not inspected.
        """
        frac = 1.0 if lot_size is None else (lot_size - self.n) / lot_size
        return rate * self.accept_prob(rate) * frac

    def aoq_lot(self, lot_size: int, bad: int) -> float:
        """Average outgoing error rate under rectification for a finite batch (exact)."""
        if self.n > lot_size:
            raise ValueError("the sample is larger than the batch")
        lo, hi = hypergeom_support(lot_size, bad, self.n)
        left = 0.0
        for d in range(lo, min(self.c, hi) + 1):
            left += hypergeom_pmf(d, lot_size, bad, self.n) * (bad - d)
        return left / lot_size

    def ati(self, rate: float, lot_size: int) -> float:
        """Average total inspection per batch under rectification, binomial model."""
        return self.n + (1.0 - self.accept_prob(rate)) * (lot_size - self.n)

    def aoql(self, lot_size: int | None = None) -> tuple[float, float]:
        """Largest average outgoing error rate and the incoming rate where it occurs."""
        if self.n == 0:
            return 1.0, 1.0
        best_x = _maximise(lambda p: self.aoq(p, lot_size), 0.0, 1.0)
        return self.aoq(best_x, lot_size), best_x

    def aoql_lot(self, lot_size: int) -> tuple[float, int]:
        """Exact finite-batch AOQL and the number of bad records where it occurs."""
        best_bad = _maximise_int(lambda bad: self.aoq_lot(lot_size, bad), 0, lot_size)
        return self.aoq_lot(lot_size, best_bad), best_bad


@dataclass(frozen=True)
class SinglePlanDesign:
    """A plan together with the requirement it was designed for and the risks it achieves."""

    plan: SinglePlan
    model: str
    acceptable: float
    rejectable: float
    producer_risk: float
    consumer_risk: float
    achieved_producer_risk: float
    achieved_consumer_risk: float
    lot_size: int | None = None
    acceptable_bad: int | None = None
    rejectable_bad: int | None = None

    def to_dict(self) -> dict:
        out = {
            "n": self.plan.n,
            "c": self.plan.c,
            "model": self.model,
            "acceptable": self.acceptable,
            "rejectable": self.rejectable,
            "producer_risk": self.producer_risk,
            "consumer_risk": self.consumer_risk,
            "achieved_producer_risk": self.achieved_producer_risk,
            "achieved_consumer_risk": self.achieved_consumer_risk,
        }
        if self.lot_size is not None:
            out["lot_size"] = self.lot_size
            out["acceptable_bad"] = self.acceptable_bad
            out["rejectable_bad"] = self.rejectable_bad
        return out


def _check_requirement(
    acceptable: float, rejectable: float, producer_risk: float, consumer_risk: float
) -> None:
    if not 0.0 <= acceptable < rejectable <= 1.0:
        raise ValueError("need 0 <= acceptable < rejectable <= 1")
    if not 0.0 < producer_risk < 1.0 or not 0.0 < consumer_risk < 1.0:
        raise ValueError("risks must be strictly between 0 and 1")
    if producer_risk + consumer_risk >= 1.0:
        raise ValueError("producer_risk + consumer_risk must be below 1")


def _smallest_n(accept_at: Callable[[int], float], lo: int, hi_cap: int, target: float) -> int:
    """Smallest n in [lo, hi_cap] with accept_at(n) <= target; accept_at is non-increasing."""
    if accept_at(lo) <= target:
        return lo
    hi = lo
    while True:
        prev = hi
        hi = min(hi_cap, max(hi * 2, hi + 1))
        if accept_at(hi) <= target:
            break
        if hi == hi_cap:
            return -1
    left = prev
    while hi - left > 1:
        mid = (left + hi) // 2
        if accept_at(mid) <= target:
            hi = mid
        else:
            left = mid
    return hi


def find_single_plan(
    acceptable: float,
    rejectable: float,
    producer_risk: float = 0.05,
    consumer_risk: float = 0.10,
    lot_size: int | None = None,
) -> SinglePlanDesign:
    """Smallest single plan (n, c) that meets both risks.

    A batch at the ``acceptable`` error rate must be accepted with probability
    at least ``1 - producer_risk``; a batch at the ``rejectable`` error rate
    must be accepted with probability at most ``consumer_risk``.

    With ``lot_size`` the hypergeometric model is used: the acceptable batch
    holds ``floor(lot_size * acceptable)`` bad records and the rejectable one
    ``ceil(lot_size * rejectable)``. Without it the binomial model is used.

    For a fixed c the acceptance probability falls as n grows, so the smallest
    n that meets the consumer risk is also the n that is kindest to the
    producer risk; c is raised until that n meets the producer risk too.
    """
    _check_requirement(acceptable, rejectable, producer_risk, consumer_risk)
    if lot_size is None:
        n_floor = 1
        for c in range(_MAX_ACCEPT_NUMBER):
            n = _smallest_n(
                lambda m, c=c: binom_cdf(c, m, rejectable),
                max(n_floor, c + 1),
                _MAX_SAMPLE,
                consumer_risk,
            )
            if n < 0:
                break
            n_floor = n
            pa_good = binom_cdf(c, n, acceptable)
            if pa_good >= 1.0 - producer_risk:
                return SinglePlanDesign(
                    plan=SinglePlan(n, c),
                    model="binomial",
                    acceptable=acceptable,
                    rejectable=rejectable,
                    producer_risk=producer_risk,
                    consumer_risk=consumer_risk,
                    achieved_producer_risk=1.0 - pa_good,
                    achieved_consumer_risk=binom_cdf(c, n, rejectable),
                )
        raise ValueError("no plan found: the two error rates are too close together")

    if lot_size < 1:
        raise ValueError("lot_size must be at least 1")
    bad_ok = bad_count_at_most(lot_size, acceptable)
    bad_no = bad_count_at_least(lot_size, rejectable)
    n_floor = 1
    for c in range(bad_no):
        n = _smallest_n(
            lambda m, c=c: hypergeom_cdf(c, lot_size, bad_no, m),
            max(n_floor, c + 1),
            lot_size,
            consumer_risk,
        )
        if n < 0:
            continue
        n_floor = n
        pa_good = hypergeom_cdf(c, lot_size, bad_ok, n)
        if pa_good >= 1.0 - producer_risk:
            return SinglePlanDesign(
                plan=SinglePlan(n, c),
                model="hypergeometric",
                acceptable=acceptable,
                rejectable=rejectable,
                producer_risk=producer_risk,
                consumer_risk=consumer_risk,
                achieved_producer_risk=1.0 - pa_good,
                achieved_consumer_risk=hypergeom_cdf(c, lot_size, bad_no, n),
                lot_size=lot_size,
                acceptable_bad=bad_ok,
                rejectable_bad=bad_no,
            )
    raise ValueError("no plan found")  # pragma: no cover - c = bad_ok, n = lot_size always works


def oc_curve(
    plan: SinglePlan, rates: Sequence[float], lot_size: int | None = None
) -> list[dict[str, float]]:
    """Operating characteristic: acceptance probability at each incoming error rate.

    With ``lot_size`` each rate is turned into a whole number of bad records
    (rounded to nearest) and the exact hypergeometric probability is reported.
    """
    rows = []
    for rate in rates:
        if lot_size is None:
            row = {
                "rate": rate,
                "accept": plan.accept_prob(rate),
                "aoq": plan.aoq(rate),
            }
        else:
            bad = min(lot_size, max(0, round(rate * lot_size)))
            row = {
                "rate": bad / lot_size,
                "bad": bad,
                "accept": plan.accept_prob_lot(lot_size, bad),
                "aoq": plan.aoq_lot(lot_size, bad),
            }
        rows.append(row)
    return rows


def _maximise(f: Callable[[float], float], lo: float, hi: float, grid: int = 400) -> float:
    """Maximiser of f on [lo, hi]: coarse grid, then golden-section in the best cell."""
    xs = [lo + (hi - lo) * i / grid for i in range(grid + 1)]
    best = max(range(grid + 1), key=lambda i: f(xs[i]))
    a, b = xs[max(0, best - 1)], xs[min(grid, best + 1)]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = b - ratio * (b - a), a + ratio * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(80):
        if f1 < f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + ratio * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - ratio * (b - a)
            f1 = f(x1)
    return (a + b) / 2.0


def _maximise_int(f: Callable[[int], float], lo: int, hi: int, grid: int = 200) -> int:
    """Integer maximiser of f on [lo, hi]: coarse grid, refined around the best point."""
    while hi - lo > grid:
        step = (hi - lo) / grid
        xs = sorted({lo + round(step * i) for i in range(grid + 1)})
        best = max(range(len(xs)), key=lambda i: f(xs[i]))
        lo, hi = xs[max(0, best - 1)], xs[min(len(xs) - 1, best + 1)]
    return max(range(lo, hi + 1), key=f)


__all__ = [
    "SinglePlan",
    "SinglePlanDesign",
    "bad_count_at_least",
    "bad_count_at_most",
    "find_single_plan",
    "oc_curve",
]
