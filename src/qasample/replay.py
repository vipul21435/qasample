"""Replay a plan on a batch whose records carry a known good/bad flag."""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .double import DoublePlan
from .sequential import SequentialPlan
from .single import SinglePlan

Plan = SinglePlan | DoublePlan | SequentialPlan


@dataclass(frozen=True)
class ReplayResult:
    audits: int
    accepted: int
    accept_rate: float
    accept_se: float
    mean_inspected: float
    lot_size: int
    bad: int
    true_rate: float
    expected_accept: float
    expected_inspected: float
    binomial_accept: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _random_order(flags: Sequence[bool], rng: random.Random, chunk: int) -> Iterator[bool]:
    """Flags in a uniformly random order without replacement, drawn lazily in chunks."""
    size = len(flags)
    first = rng.sample(range(size), min(chunk, size))
    for i in first:
        yield flags[i]
    if len(first) == size:
        return
    seen = set(first)
    while len(seen) * 2 < size:
        i = rng.randrange(size)
        if i not in seen:
            seen.add(i)
            yield flags[i]
    rest = [i for i in range(size) if i not in seen]
    rng.shuffle(rest)
    for i in rest:
        yield flags[i]


def _single_run(plan: SinglePlan, records: Iterator[bool]) -> tuple[bool, int]:
    bad = sum(1 for _, flag in zip(range(plan.n), records, strict=False) if flag)
    return bad <= plan.c, plan.n


def expected(plan: Plan, lot_size: int, bad: int) -> tuple[float, float, float]:
    """Exact (finite-batch accept probability, average inspected, binomial accept probability)."""
    rate = bad / lot_size
    if isinstance(plan, SinglePlan):
        return plan.accept_prob_lot(lot_size, bad), float(plan.n), plan.accept_prob(rate)
    if isinstance(plan, DoublePlan):
        return (
            plan.accept_prob_lot(lot_size, bad),
            plan.asn_lot(lot_size, bad),
            plan.accept_prob(rate),
        )
    finite = plan.exact(lot_size=lot_size, bad=bad)
    return finite.accept, finite.asn, plan.exact(rate).accept


def replay(flags: Sequence[bool], plan: Plan, audits: int = 10_000, seed: int = 0) -> ReplayResult:
    """Draw ``audits`` independent random audits of the batch under ``plan``.

    ``flags`` holds one entry per record of the batch, True for a bad record.
    Each audit inspects records in a fresh random order without replacement,
    exactly as the plan prescribes. The observed accept rate is reported next
    to the exact probability computed for a batch of this size with this many
    bad records.
    """
    if audits < 1:
        raise ValueError("audits must be at least 1")
    size = len(flags)
    if size == 0:
        raise ValueError("the batch is empty")
    if isinstance(plan, SinglePlan):
        need = plan.n
    elif isinstance(plan, DoublePlan):
        need = plan.n1 + plan.n2
    else:
        need = plan.max_n if plan.max_n is not None else 4 * plan.first_accept_n()
    if not isinstance(plan, SequentialPlan) and need > size:
        raise ValueError("the plan inspects more records than the batch holds")
    rng = random.Random(seed)
    accepted = inspected = 0
    for _ in range(audits):
        order = _random_order(flags, rng, need)
        if isinstance(plan, SinglePlan):
            ok, used = _single_run(plan, order)
        else:
            ok, used = plan.run(order)
        accepted += 1 if ok else 0
        inspected += used
    bad = sum(1 for flag in flags if flag)
    exp_accept, exp_inspected, binom_accept = expected(plan, size, bad)
    rate = accepted / audits
    return ReplayResult(
        audits=audits,
        accepted=accepted,
        accept_rate=rate,
        accept_se=math.sqrt(exp_accept * (1.0 - exp_accept) / audits),
        mean_inspected=inspected / audits,
        lot_size=size,
        bad=bad,
        true_rate=bad / size,
        expected_accept=exp_accept,
        expected_inspected=exp_inspected,
        binomial_accept=binom_accept,
    )
