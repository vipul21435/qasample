import random

import pytest

from qasample.bounds import clopper_pearson
from qasample.inspector import (
    Inspector,
    corrected_bounds_estimated,
    estimate_inspector,
    find_single_plan_with_inspector,
    risks_under_inspector,
)
from qasample.single import find_single_plan


def test_apparent_rate_and_its_inverse():
    insp = Inspector(miss_rate=0.2, false_alarm_rate=0.01)
    assert insp.apparent(0.0) == pytest.approx(0.01)
    assert insp.apparent(1.0) == pytest.approx(0.8)
    assert insp.apparent(0.05) == pytest.approx(0.05 * 0.8 + 0.95 * 0.01)
    for rate in (0.0, 0.03, 0.4, 1.0):
        assert insp.true_rate(insp.apparent(rate)) == pytest.approx(rate)
    # apparent rates outside the reachable range are clipped
    assert insp.true_rate(0.0) == 0.0
    assert insp.true_rate(0.95) == 1.0


def test_perfect_inspector_changes_nothing():
    insp = Inspector()
    assert insp.apparent(0.07) == pytest.approx(0.07)
    assert insp.corrected_bounds(3, 100) == pytest.approx(clopper_pearson(3, 100))


def test_misses_raise_the_consumer_risk_of_a_naive_plan():
    naive = find_single_plan(0.01, 0.05)
    insp = Inspector(miss_rate=0.3)
    risks = risks_under_inspector(naive.plan, 0.01, 0.05, insp)
    assert risks["consumer_risk"] > 0.10
    assert risks["producer_risk"] < naive.achieved_producer_risk
    fixed = find_single_plan_with_inspector(0.01, 0.05, insp)
    again = risks_under_inspector(fixed.plan, 0.01, 0.05, insp)
    assert again["consumer_risk"] <= 0.10 and again["producer_risk"] <= 0.05
    assert fixed.plan.n > naive.plan.n


def test_false_alarms_raise_the_producer_risk_of_a_naive_plan():
    naive = find_single_plan(0.01, 0.05)
    insp = Inspector(false_alarm_rate=0.02)
    risks = risks_under_inspector(naive.plan, 0.01, 0.05, insp)
    assert risks["producer_risk"] > 0.05


def test_acceptance_under_inspector_matches_seeded_simulation():
    plan = find_single_plan(0.02, 0.10).plan
    insp = Inspector(miss_rate=0.25, false_alarm_rate=0.03)
    rng = random.Random(9)
    rate, trials, accepted = 0.06, 4000, 0
    for _ in range(trials):
        flagged = 0
        for _ in range(plan.n):
            is_bad = rng.random() < rate
            says_bad = rng.random() >= 0.25 if is_bad else rng.random() < 0.03
            flagged += says_bad
        accepted += flagged <= plan.c
    expect = insp.accept_prob(plan, rate)
    assert abs(accepted / trials - expect) < 4 * (expect * (1 - expect) / trials) ** 0.5


def test_corrected_bounds_cover_the_true_rate_in_simulation():
    insp = Inspector(miss_rate=0.2, false_alarm_rate=0.02)
    rng = random.Random(2)
    rate, n, trials, hits = 0.08, 150, 1000, 0
    q = insp.apparent(rate)
    for _ in range(trials):
        flagged = sum(rng.random() < q for _ in range(n))
        low, high = insp.corrected_bounds(flagged, n, 0.9)
        hits += low <= rate <= high
    assert hits / trials >= 0.9 - 0.03


def test_estimated_inspector():
    est = estimate_inspector(missed=6, known_bad=40, false_alarms=2, known_good=200)
    assert est["miss_rate"] == pytest.approx(0.15)
    assert est["false_alarm_rate"] == pytest.approx(0.01)
    assert est["miss_interval"] == pytest.approx(clopper_pearson(6, 40))
    low, high = corrected_bounds_estimated(12, 200, 6, 40, 2, 200)
    known = Inspector(0.15, 0.01).corrected_bounds(12, 200)
    assert low <= known[0] and high >= known[1]
    point = Inspector(0.15, 0.01).true_rate(12 / 200)
    assert low <= point <= high
    # a calibration set too small to say anything gives the trivial interval
    assert corrected_bounds_estimated(5, 50, 1, 2, 1, 2) == (0.0, 1.0)
    with pytest.raises(ValueError):
        estimate_inspector(1, 0, 1, 10)


def test_validation():
    with pytest.raises(ValueError):
        Inspector(miss_rate=1.2)
    with pytest.raises(ValueError):
        Inspector(miss_rate=0.6, false_alarm_rate=0.4)
