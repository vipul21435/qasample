"""An inspector who sometimes gets it wrong.

Two error rates describe the inspector: ``miss_rate`` is the chance that a
bad record is passed, ``false_alarm_rate`` the chance that a good record is
flagged. If the true error rate of the batch is p, the rate the inspector
reports is

    q = p * (1 - miss_rate) + (1 - p) * false_alarm_rate,

and solving for p gives p = (q - false_alarm_rate) / (1 - miss_rate - false_alarm_rate).
Only the binomial model is handled here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bounds import clopper_pearson
from .single import SinglePlan, SinglePlanDesign, find_single_plan


@dataclass(frozen=True)
class Inspector:
    miss_rate: float = 0.0
    false_alarm_rate: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.miss_rate <= 1.0 or not 0.0 <= self.false_alarm_rate <= 1.0:
            raise ValueError("inspector error rates must be in [0, 1]")
        if self.miss_rate + self.false_alarm_rate >= 1.0:
            raise ValueError(
                "miss_rate + false_alarm_rate must be below 1: such an inspector carries no "
                "information about the records"
            )

    def apparent(self, true_rate: float) -> float:
        """Rate of flagged records when the true error rate is ``true_rate``."""
        return true_rate * (1.0 - self.miss_rate) + (1.0 - true_rate) * self.false_alarm_rate

    def true_rate(self, apparent: float) -> float:
        """True error rate behind an apparent rate, clipped to [0, 1]."""
        value = (apparent - self.false_alarm_rate) / (1.0 - self.miss_rate - self.false_alarm_rate)
        return min(1.0, max(0.0, value))

    def accept_prob(self, plan: SinglePlan, true_rate: float) -> float:
        """Acceptance probability of a single plan when this inspector does the counting."""
        return plan.accept_prob(self.apparent(true_rate))

    def corrected_bounds(
        self, flagged: int, n: int, confidence: float = 0.95, side: str = "two-sided"
    ) -> tuple[float, float]:
        """Exact bounds on the true error rate when the inspector's error rates are known.

        The Clopper-Pearson bounds on the apparent rate are mapped through
        ``true_rate``; the map is increasing, so the confidence level carries over.
        """
        low, high = clopper_pearson(flagged, n, confidence, side)
        return self.true_rate(low), self.true_rate(high)


def find_single_plan_with_inspector(
    acceptable: float,
    rejectable: float,
    inspector: Inspector,
    producer_risk: float = 0.05,
    consumer_risk: float = 0.10,
) -> SinglePlanDesign:
    """Smallest single plan that meets both risks on the true rates despite the inspector.

    The plan is designed on the apparent rates the inspector would report at
    the acceptable and rejectable true rates. The acceptance number then
    counts flagged records.
    """
    return find_single_plan(
        inspector.apparent(acceptable),
        inspector.apparent(rejectable),
        producer_risk,
        consumer_risk,
    )


def risks_under_inspector(
    plan: SinglePlan, acceptable: float, rejectable: float, inspector: Inspector
) -> dict[str, float]:
    """Real producer and consumer risks of a plan that was designed for a perfect inspector."""
    return {
        "producer_risk": 1.0 - inspector.accept_prob(plan, acceptable),
        "consumer_risk": inspector.accept_prob(plan, rejectable),
    }


def estimate_inspector(
    missed: int,
    known_bad: int,
    false_alarms: int,
    known_good: int,
    confidence: float = 0.95,
) -> dict:
    """Inspector error rates from a calibration set with known answers.

    The inspector saw ``known_bad`` records known to be bad and passed
    ``missed`` of them, and saw ``known_good`` records known to be good and
    flagged ``false_alarms`` of them. Returns the two point estimates with
    Clopper-Pearson intervals.
    """
    miss_interval = clopper_pearson(missed, known_bad, confidence)
    alarm_interval = clopper_pearson(false_alarms, known_good, confidence)
    return {
        "miss_rate": missed / known_bad,
        "miss_interval": miss_interval,
        "false_alarm_rate": false_alarms / known_good,
        "false_alarm_interval": alarm_interval,
    }


def corrected_bounds_estimated(
    flagged: int,
    n: int,
    missed: int,
    known_bad: int,
    false_alarms: int,
    known_good: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Conservative bounds on the true rate when the inspector's rates are themselves estimated.

    Three two-sided Clopper-Pearson intervals (apparent rate, miss rate, false
    alarm rate) are taken at confidence 1 - alpha / 3 each, so all three hold
    together with probability at least ``confidence``; the bounds are the
    smallest and largest true rate compatible with them.
    """
    level = 1.0 - (1.0 - confidence) / 3.0
    q_low, q_high = clopper_pearson(flagged, n, level)
    m_low, m_high = clopper_pearson(missed, known_bad, level)
    f_low, f_high = clopper_pearson(false_alarms, known_good, level)
    candidates = []
    for q in (q_low, q_high):
        for m in (m_low, m_high):
            for f in (f_low, f_high):
                if m + f >= 1.0:
                    return 0.0, 1.0
                candidates.append(min(1.0, max(0.0, (q - f) / (1.0 - m - f))))
    return min(candidates), max(candidates)
