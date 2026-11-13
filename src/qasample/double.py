"""Double sampling plans: a second sample is drawn only when the first is inconclusive."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .dist import binom_cdf, binom_pmf, hypergeom_cdf, hypergeom_pmf
from .single import (
    _check_requirement,
    _smallest_n,
    bad_count_at_least,
    bad_count_at_most,
    find_single_plan,
)


@dataclass(frozen=True)
class DoublePlan:
    """Two-stage plan.

    Inspect ``n1`` records and count the bad ones, d1. Accept if d1 <= ``c1``,
    reject if d1 >= ``r1``; otherwise inspect ``n2`` more and accept if the
    total d1 + d2 <= ``c2``.
    """

    n1: int
    c1: int
    r1: int
    n2: int
    c2: int

    def __post_init__(self) -> None:
        if self.n1 < 1 or self.n2 < 1:
            raise ValueError("both sample sizes must be at least 1")
        if not 0 <= self.c1 < self.r1 <= self.c2 + 1:
            raise ValueError("need 0 <= c1 < r1 <= c2 + 1")

    def _stage(self, lot_size: int | None, rate_or_bad: float) -> Callable[[int], float]:
        if lot_size is None:
            return lambda d: binom_pmf(d, self.n1, rate_or_bad)
        if self.n1 + self.n2 > lot_size:
            raise ValueError("the two samples together are larger than the batch")
        return lambda d: hypergeom_pmf(d, lot_size, int(rate_or_bad), self.n1)

    def accept_prob(self, rate: float) -> float:
        """Exact probability of acceptance under the binomial model."""
        total = binom_cdf(self.c1, self.n1, rate)
        for d1 in range(self.c1 + 1, self.r1):
            total += binom_pmf(d1, self.n1, rate) * binom_cdf(self.c2 - d1, self.n2, rate)
        return min(1.0, total)

    def accept_prob_lot(self, lot_size: int, bad: int) -> float:
        """Exact probability of acceptance for a finite batch with ``bad`` bad records."""
        first = self._stage(lot_size, bad)
        total = hypergeom_cdf(self.c1, lot_size, bad, self.n1)
        for d1 in range(self.c1 + 1, self.r1):
            p1 = first(d1)
            if p1 > 0.0:
                total += p1 * hypergeom_cdf(self.c2 - d1, lot_size - self.n1, bad - d1, self.n2)
        return min(1.0, total)

    def second_sample_prob(self, rate: float) -> float:
        """Probability that the second sample is needed, binomial model."""
        return sum(binom_pmf(d, self.n1, rate) for d in range(self.c1 + 1, self.r1))

    def second_sample_prob_lot(self, lot_size: int, bad: int) -> float:
        first = self._stage(lot_size, bad)
        return sum(first(d) for d in range(self.c1 + 1, self.r1))

    def asn(self, rate: float) -> float:
        """Average number of records inspected (no curtailment), binomial model."""
        return self.n1 + self.n2 * self.second_sample_prob(rate)

    def asn_lot(self, lot_size: int, bad: int) -> float:
        return self.n1 + self.n2 * self.second_sample_prob_lot(lot_size, bad)

    def run(self, records: Iterator[bool]) -> tuple[bool, int]:
        """Apply the plan to a stream of flags (True = bad); returns (accepted, inspected)."""
        d1 = sum(1 for _, flag in zip(range(self.n1), records, strict=False) if flag)
        if d1 <= self.c1:
            return True, self.n1
        if d1 >= self.r1:
            return False, self.n1
        d2 = sum(1 for _, flag in zip(range(self.n2), records, strict=False) if flag)
        return d1 + d2 <= self.c2, self.n1 + self.n2

    def to_dict(self) -> dict:
        return {"n1": self.n1, "c1": self.c1, "r1": self.r1, "n2": self.n2, "c2": self.c2}


@dataclass(frozen=True)
class DoublePlanDesign:
    plan: DoublePlan
    model: str
    achieved_producer_risk: float
    achieved_consumer_risk: float
    asn_acceptable: float
    asn_rejectable: float
    single_n: int

    def to_dict(self) -> dict:
        out = self.plan.to_dict()
        out.update(
            model=self.model,
            achieved_producer_risk=self.achieved_producer_risk,
            achieved_consumer_risk=self.achieved_consumer_risk,
            asn_acceptable=self.asn_acceptable,
            asn_rejectable=self.asn_rejectable,
            single_n=self.single_n,
        )
        return out


def find_double_plan(
    acceptable: float,
    rejectable: float,
    producer_risk: float = 0.05,
    consumer_risk: float = 0.10,
    lot_size: int | None = None,
    ratios: tuple[int, ...] = (1, 2),
    max_c2: int = 12,
) -> DoublePlanDesign:
    """Double plan meeting both risks with the smallest average sample at the acceptable rate.

    The search covers plans with n2 = ratio * n1 for each ratio given, r1 = c2 + 1
    and c2 up to three above the acceptance number of the smallest single plan
    (capped at ``max_c2``). For each (c1, c2, ratio) the smallest n1 that meets
    the consumer risk is found by bisection, since the acceptance probability
    falls as the samples grow; the plan is kept if it also meets the producer
    risk. Among the kept plans the one with the smallest average sample number
    at the acceptable error rate wins. Risks are exact, not approximated.
    """
    _check_requirement(acceptable, rejectable, producer_risk, consumer_risk)
    single = find_single_plan(acceptable, rejectable, producer_risk, consumer_risk, lot_size)
    c_top = min(single.plan.c + 3, max_c2)
    if lot_size is not None:
        bad_ok = bad_count_at_most(lot_size, acceptable)
        bad_no = bad_count_at_least(lot_size, rejectable)
    best: tuple[float, DoublePlan] | None = None
    for ratio in ratios:
        if ratio < 1:
            raise ValueError("ratios must be positive integers")
        n1_cap = 10**9 if lot_size is None else lot_size // (1 + ratio)
        if n1_cap < 1:
            continue
        for c2 in range(1, c_top + 1):
            for c1 in range(0, c2):
                r1 = c2 + 1

                def make(n1: int, c1: int = c1, r1: int = r1, c2: int = c2) -> DoublePlan:
                    return DoublePlan(n1, c1, r1, ratio * n1, c2)  # noqa: B023

                if lot_size is None:
                    n1 = _smallest_n(
                        lambda m: make(m).accept_prob(rejectable), 1, n1_cap, consumer_risk
                    )
                else:
                    n1 = _smallest_n(
                        lambda m: make(m).accept_prob_lot(lot_size, bad_no),
                        1,
                        n1_cap,
                        consumer_risk,
                    )
                if n1 < 0:
                    continue
                plan = make(n1)
                if lot_size is None:
                    pa_good = plan.accept_prob(acceptable)
                    asn = plan.asn(acceptable)
                else:
                    pa_good = plan.accept_prob_lot(lot_size, bad_ok)
                    asn = plan.asn_lot(lot_size, bad_ok)
                if pa_good < 1.0 - producer_risk:
                    continue
                if best is None or asn < best[0] - 1e-12:
                    best = (asn, plan)
    if best is None:
        raise ValueError(
            "no double plan found in the search range; raise max_c2 or use a single plan"
        )
    plan = best[1]
    if lot_size is None:
        return DoublePlanDesign(
            plan=plan,
            model="binomial",
            achieved_producer_risk=1.0 - plan.accept_prob(acceptable),
            achieved_consumer_risk=plan.accept_prob(rejectable),
            asn_acceptable=plan.asn(acceptable),
            asn_rejectable=plan.asn(rejectable),
            single_n=single.plan.n,
        )
    return DoublePlanDesign(
        plan=plan,
        model="hypergeometric",
        achieved_producer_risk=1.0 - plan.accept_prob_lot(lot_size, bad_ok),
        achieved_consumer_risk=plan.accept_prob_lot(lot_size, bad_no),
        asn_acceptable=plan.asn_lot(lot_size, bad_ok),
        asn_rejectable=plan.asn_lot(lot_size, bad_no),
        single_n=single.plan.n,
    )
