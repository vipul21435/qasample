"""Wald's sequential probability ratio plan for a pass/fail inspection.

Records are inspected one at a time. After n records with d bad ones the
log likelihood ratio of the rejectable rate p1 against the acceptable rate p0 is

    d * ln(p1 / p0) + (n - d) * ln((1 - p1) / (1 - p0)).

Inspection stops with acceptance when it falls to ln(beta / (1 - alpha)) or
below and with rejection when it reaches ln((1 - beta) / alpha) or above.
In the (n, d) plane these are two parallel lines,

    accept when d <= -h_accept + s * n,   reject when d >= h_reject + s * n.

Wald's thresholds ignore the overshoot past the lines, so alpha and beta are
nominal. ``exact`` computes the real operating characteristic and average
sample number by carrying the probability of every (n, d) state forward.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property

_EPS = 1e-9
_MASS_STOP = 1e-13
_HARD_LIMIT = 2_000_000


@dataclass(frozen=True)
class SequentialResult:
    """Exact behaviour of a sequential plan at one error rate."""

    accept: float
    reject: float
    undecided: float
    asn: float
    max_n: int

    def to_dict(self) -> dict:
        return {
            "accept": self.accept,
            "reject": self.reject,
            "undecided": self.undecided,
            "asn": self.asn,
            "max_n": self.max_n,
        }


@dataclass(frozen=True)
class SequentialPlan:
    """Wald's plan for acceptable rate p0 and rejectable rate p1 with nominal risks.

    ``max_n`` truncates the inspection: a run still undecided after ``max_n``
    records is settled by the sign of the log likelihood ratio (accept when it
    is not positive, that is when d <= s * max_n), or by ``truncation_accept``
    when that acceptance number is given.
    """

    acceptable: float
    rejectable: float
    producer_risk: float = 0.05
    consumer_risk: float = 0.10
    max_n: int | None = None
    truncation_accept: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.acceptable < self.rejectable < 1.0:
            raise ValueError("need 0 < acceptable < rejectable < 1")
        if not 0.0 < self.producer_risk < 1.0 or not 0.0 < self.consumer_risk < 1.0:
            raise ValueError("risks must be strictly between 0 and 1")
        if self.producer_risk + self.consumer_risk >= 1.0:
            raise ValueError("producer_risk + consumer_risk must be below 1")
        if self.max_n is not None and self.max_n < 1:
            raise ValueError("max_n must be at least 1")

    # --- the two lines -------------------------------------------------

    @cached_property
    def _g1(self) -> float:
        return math.log(self.rejectable / self.acceptable)

    @cached_property
    def _g2(self) -> float:
        return math.log((1.0 - self.acceptable) / (1.0 - self.rejectable))

    @cached_property
    def slope(self) -> float:
        return self._g2 / (self._g1 + self._g2)

    @cached_property
    def h_accept(self) -> float:
        return math.log((1.0 - self.producer_risk) / self.consumer_risk) / (self._g1 + self._g2)

    @cached_property
    def h_reject(self) -> float:
        return math.log((1.0 - self.consumer_risk) / self.producer_risk) / (self._g1 + self._g2)

    def accept_number(self, n: int) -> int:
        """Largest bad count that accepts after n records; -1 while none can."""
        return max(-1, math.floor(-self.h_accept + self.slope * n + _EPS))

    def reject_number(self, n: int) -> int:
        """Smallest bad count that rejects after n records (may exceed n early on)."""
        return math.ceil(self.h_reject + self.slope * n - _EPS)

    def _final_accept_number(self, n: int) -> int:
        if self.truncation_accept is not None:
            return self.truncation_accept
        return math.floor(self.slope * n + _EPS)

    def decide(self, n: int, bad: int) -> str:
        """``"accept"``, ``"reject"`` or ``"continue"`` after n records with ``bad`` bad ones."""
        if bad <= self.accept_number(n):
            return "accept"
        if bad >= self.reject_number(n):
            return "reject"
        if self.max_n is not None and n >= self.max_n:
            return "accept" if bad <= self._final_accept_number(n) else "reject"
        return "continue"

    def table(self, upto: int) -> list[dict[str, int | None]]:
        """Acceptance and rejection numbers for n = 1..upto (None where not possible)."""
        rows: list[dict[str, int | None]] = []
        for n in range(1, upto + 1):
            a, r = self.accept_number(n), self.reject_number(n)
            rows.append({"n": n, "accept": a if a >= 0 else None, "reject": r if r <= n else None})
        return rows

    def first_accept_n(self) -> int:
        """Fewest records after which a clean run is accepted."""
        return max(1, math.ceil(self.h_accept / self.slope - _EPS))

    def run(self, records: Iterator[bool]) -> tuple[bool, int]:
        """Apply the plan to a stream of flags (True = bad); returns (accepted, inspected).

        If the stream ends before a decision, the run is settled as at truncation.
        """
        n = bad = 0
        for flag in records:
            n += 1
            bad += 1 if flag else 0
            verdict = self.decide(n, bad)
            if verdict != "continue":
                return verdict == "accept", n
        return bad <= self._final_accept_number(n), n

    # --- exact operating characteristic --------------------------------

    def exact(
        self, rate: float | None = None, lot_size: int | None = None, bad: int | None = None
    ) -> SequentialResult:
        """Exact acceptance probability and average sample number.

        Give ``rate`` for the binomial model, or ``lot_size`` and ``bad`` for a
        finite batch sampled without replacement (which ends at the latest
        when the batch is exhausted).
        """
        if lot_size is None:
            if rate is None or not 0.0 <= rate <= 1.0:
                raise ValueError("rate must be in [0, 1]")
            limit = self.max_n if self.max_n is not None else _HARD_LIMIT
        else:
            if bad is None or not 0 <= bad <= lot_size:
                raise ValueError("bad must be between 0 and lot_size")
            limit = lot_size if self.max_n is None else min(self.max_n, lot_size)
        forced_end = self.max_n is not None or lot_size is not None
        state: dict[int, float] = {0: 1.0}
        accept = reject = asn = 0.0
        n = 0
        while state and n < limit:
            n += 1
            nxt: dict[int, float] = {}
            for d, mass in state.items():
                if lot_size is None:
                    p_bad = rate
                else:
                    p_bad = (bad - d) / (lot_size - (n - 1))
                if p_bad > 0.0:
                    nxt[d + 1] = nxt.get(d + 1, 0.0) + mass * p_bad
                if p_bad < 1.0:
                    nxt[d] = nxt.get(d, 0.0) + mass * (1.0 - p_bad)
            a, r = self.accept_number(n), self.reject_number(n)
            last = forced_end and n == limit
            final_a = self._final_accept_number(n) if last else None
            state = {}
            for d, mass in nxt.items():
                if d <= a:
                    accept += mass
                    asn += n * mass
                elif d >= r:
                    reject += mass
                    asn += n * mass
                elif final_a is not None:
                    if d <= final_a:
                        accept += mass
                    else:
                        reject += mass
                    asn += n * mass
                else:
                    state[d] = mass
            if not forced_end and sum(state.values()) < _MASS_STOP:
                break
        undecided = sum(state.values())
        asn += n * undecided
        return SequentialResult(accept, reject, undecided, asn, n)

    # --- Wald's closed-form approximations -----------------------------

    def _rate_at(self, h: float) -> float:
        if abs(h) < 1e-12:
            return self.slope
        if h * self._g1 > 700.0:
            return 0.0
        if -h * self._g2 > 700.0:
            return 1.0
        num = -math.expm1(-h * self._g2)
        den = math.exp(h * self._g1) - math.exp(-h * self._g2)
        return num / den

    def wald_oc(self, rate: float) -> float:
        """Wald's approximation to the acceptance probability (overshoot ignored)."""
        if rate <= 0.0:
            return 1.0
        if rate >= 1.0:
            return 0.0
        lo, hi = -1.0, 1.0
        while self._rate_at(lo) < rate and lo > -1e6:
            lo *= 2.0
        while self._rate_at(hi) > rate and hi < 1e6:
            hi *= 2.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if self._rate_at(mid) > rate:
                lo = mid
            else:
                hi = mid
        h = (lo + hi) / 2.0
        log_a = math.log((1.0 - self.consumer_risk) / self.producer_risk)
        log_b = math.log(self.consumer_risk / (1.0 - self.producer_risk))
        if abs(h) < 1e-9:
            return log_a / (log_a - log_b)
        if h * log_a > 700.0:
            return 1.0
        if h * log_b > 700.0:
            return 0.0
        return math.expm1(h * log_a) / (math.exp(h * log_a) - math.exp(h * log_b))

    def wald_asn(self, rate: float) -> float:
        """Wald's approximation to the average sample number (overshoot ignored)."""
        log_a = math.log((1.0 - self.consumer_risk) / self.producer_risk)
        log_b = math.log(self.consumer_risk / (1.0 - self.producer_risk))
        drift = rate * self._g1 - (1.0 - rate) * self._g2
        if abs(drift) < 1e-9 * (self._g1 + self._g2):
            return self.h_accept * self.h_reject / (self.slope * (1.0 - self.slope))
        accept = self.wald_oc(rate)
        return (accept * log_b + (1.0 - accept) * log_a) / drift

    def to_dict(self) -> dict:
        return {
            "acceptable": self.acceptable,
            "rejectable": self.rejectable,
            "producer_risk": self.producer_risk,
            "consumer_risk": self.consumer_risk,
            "slope": self.slope,
            "h_accept": self.h_accept,
            "h_reject": self.h_reject,
            "max_n": self.max_n,
            "truncation_accept": self.truncation_accept,
        }
