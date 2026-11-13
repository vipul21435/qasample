import math

import numpy as np
import pytest
from scipy import stats

from qasample.single import (
    SinglePlan,
    bad_count_at_least,
    bad_count_at_most,
    find_single_plan,
    oc_curve,
)


def brute_force_binomial(acceptable, rejectable, alpha, beta, n_max):
    for n in range(1, n_max + 1):
        cs = np.arange(n)
        ok = (stats.binom.cdf(cs, n, rejectable) <= beta) & (
            stats.binom.cdf(cs, n, acceptable) >= 1 - alpha
        )
        if ok.any():
            return n, int(cs[ok][0])
    return None


def brute_force_hypergeometric(lot, bad_ok, bad_no, alpha, beta):
    for n in range(1, lot + 1):
        cs = np.arange(n)
        ok = (stats.hypergeom.cdf(cs, lot, bad_no, n) <= beta) & (
            stats.hypergeom.cdf(cs, lot, bad_ok, n) >= 1 - alpha
        )
        if ok.any():
            return n, int(cs[ok][0])
    return None


@pytest.mark.parametrize(
    "acceptable,rejectable,alpha,beta",
    [
        (0.01, 0.06, 0.05, 0.10),
        (0.02, 0.10, 0.05, 0.10),
        (0.05, 0.20, 0.10, 0.10),
        (0.001, 0.05, 0.05, 0.05),
        (0.03, 0.09, 0.01, 0.20),
        (0.0, 0.10, 0.05, 0.10),
    ],
)
def test_binomial_plan_is_the_smallest_by_enumeration(acceptable, rejectable, alpha, beta):
    design = find_single_plan(acceptable, rejectable, alpha, beta)
    assert (design.plan.n, design.plan.c) == brute_force_binomial(
        acceptable, rejectable, alpha, beta, 400
    )
    assert design.achieved_producer_risk <= alpha
    assert design.achieved_consumer_risk <= beta
    assert design.model == "binomial"


@pytest.mark.parametrize(
    "lot,acceptable,rejectable,alpha,beta",
    [
        (200, 0.01, 0.06, 0.05, 0.10),
        (150, 0.02, 0.10, 0.05, 0.10),
        (60, 0.05, 0.20, 0.10, 0.10),
        (40, 0.0, 0.10, 0.05, 0.05),
        (300, 0.03, 0.09, 0.01, 0.20),
        (25, 0.04, 0.08, 0.05, 0.10),
    ],
)
def test_hypergeometric_plan_is_the_smallest_by_enumeration(
    lot, acceptable, rejectable, alpha, beta
):
    design = find_single_plan(acceptable, rejectable, alpha, beta, lot_size=lot)
    bad_ok = bad_count_at_most(lot, acceptable)
    bad_no = bad_count_at_least(lot, rejectable)
    assert (design.acceptable_bad, design.rejectable_bad) == (bad_ok, bad_no)
    assert (design.plan.n, design.plan.c) == brute_force_hypergeometric(
        lot, bad_ok, bad_no, alpha, beta
    )
    assert design.achieved_producer_risk <= alpha
    assert design.achieved_consumer_risk <= beta
    assert design.plan.n <= lot


def test_finite_batch_needs_no_more_than_the_binomial_plan():
    big = find_single_plan(0.01, 0.05)
    for lot in (500, 2000, 10000):
        small = find_single_plan(0.01, 0.05, lot_size=lot)
        assert small.plan.n <= big.plan.n


def test_bad_count_rounding_is_guarded_against_float_noise():
    assert bad_count_at_most(100, 0.29) == 29
    assert bad_count_at_least(100, 0.29) == 29
    assert bad_count_at_most(1000, 0.0015) == 1
    assert bad_count_at_least(1000, 0.0015) == 2


def test_requirement_validation():
    with pytest.raises(ValueError):
        find_single_plan(0.05, 0.05)
    with pytest.raises(ValueError):
        find_single_plan(0.01, 0.05, producer_risk=0.0)
    with pytest.raises(ValueError):
        find_single_plan(0.01, 0.05, producer_risk=0.6, consumer_risk=0.5)
    with pytest.raises(ValueError):
        find_single_plan(0.01, 0.05, lot_size=0)
    with pytest.raises(ValueError):
        SinglePlan(-1, 0)
    with pytest.raises(ValueError):
        SinglePlan(5, -1)


def test_oc_curve_matches_scipy_and_falls_with_the_error_rate():
    plan = SinglePlan(89, 2)
    rates = [0.0, 0.005, 0.01, 0.02, 0.04, 0.06, 0.1, 1.0]
    rows = oc_curve(plan, rates)
    for row, rate in zip(rows, rates, strict=True):
        assert row["accept"] == pytest.approx(stats.binom.cdf(2, 89, rate), rel=1e-9, abs=1e-12)
    accepts = [row["accept"] for row in rows]
    assert accepts == sorted(accepts, reverse=True)
    assert accepts[0] == 1.0 and accepts[-1] == 0.0


def test_oc_curve_finite_batch():
    plan = SinglePlan(50, 1)
    rows = oc_curve(plan, [0.0, 0.01, 0.033, 0.2], lot_size=400)
    assert [row["bad"] for row in rows] == [0, 4, 13, 80]
    for row in rows:
        assert row["accept"] == pytest.approx(
            stats.hypergeom.cdf(1, 400, row["bad"], 50), rel=1e-9
        )
    with pytest.raises(ValueError):
        SinglePlan(50, 1).accept_prob_lot(40, 3)


def test_aoq_and_aoql_binomial():
    plan = SinglePlan(89, 2)
    lot = 10000
    grid = [i / 20000 for i in range(1, 4000)]
    best = max(grid, key=lambda p: plan.aoq(p, lot))
    value, where = plan.aoql(lot)
    assert value == pytest.approx(plan.aoq(best, lot), rel=1e-6)
    assert where == pytest.approx(best, abs=1e-4)
    assert value >= plan.aoq(best, lot) - 1e-15
    # with no batch size the factor (N - n) / N is dropped
    assert plan.aoq(0.02) == pytest.approx(0.02 * stats.binom.cdf(2, 89, 0.02))
    assert plan.aoq(0.02, lot) == pytest.approx(plan.aoq(0.02) * (lot - 89) / lot)


def test_aoq_finite_batch_by_direct_expectation():
    plan = SinglePlan(20, 1)
    lot, bad = 120, 9
    expect = 0.0
    for d in range(0, 2):
        expect += stats.hypergeom.pmf(d, lot, bad, 20) * (bad - d)
    assert plan.aoq_lot(lot, bad) == pytest.approx(expect / lot, rel=1e-9)
    value, where = plan.aoql_lot(lot)
    brute = max(range(lot + 1), key=lambda b: plan.aoq_lot(lot, b))
    assert where == brute
    assert value == pytest.approx(plan.aoq_lot(lot, brute))
    with pytest.raises(ValueError):
        plan.aoq_lot(10, 1)


def test_aoql_int_search_on_a_large_batch():
    plan = SinglePlan(80, 2)
    lot = 3000
    value, where = plan.aoql_lot(lot)
    brute = max(range(lot + 1), key=lambda b: plan.aoq_lot(lot, b))
    assert value == pytest.approx(plan.aoq_lot(lot, brute), rel=1e-12)
    assert where == brute


def test_ati_between_sample_and_batch():
    plan = SinglePlan(89, 2)
    assert plan.ati(0.0, 5000) == pytest.approx(89)
    assert plan.ati(1.0, 5000) == pytest.approx(5000)
    assert 89 < plan.ati(0.03, 5000) < 5000


def test_tiny_batch_still_gets_a_plan():
    # 10 records: "at most 1 %" means 0 bad, "at least 5 %" means 1 bad or more
    # with one bad record, a sample of n misses it with probability (10 - n) / 10
    design = find_single_plan(0.01, 0.05, consumer_risk=0.15, lot_size=10)
    assert (design.acceptable_bad, design.rejectable_bad) == (0, 1)
    assert (design.plan.n, design.plan.c) == (9, 0)
    assert design.achieved_consumer_risk == pytest.approx(0.1)
    assert design.achieved_producer_risk == pytest.approx(0.0)


def test_zero_sample_plan_always_accepts():
    plan = SinglePlan(0, 0)
    assert plan.accept_prob(0.5) == 1.0
    assert plan.aoql() == (1.0, 1.0)


def test_design_to_dict_round_trip():
    d = find_single_plan(0.01, 0.06, lot_size=500).to_dict()
    assert d["model"] == "hypergeometric" and d["lot_size"] == 500
    assert math.isclose(d["acceptable"], 0.01)
    d2 = find_single_plan(0.01, 0.06).to_dict()
    assert "lot_size" not in d2
