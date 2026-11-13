import itertools
import math
import random

import pytest
from scipy import stats

from qasample.double import DoublePlan, find_double_plan
from qasample.single import bad_count_at_least, bad_count_at_most, find_single_plan


def brute_accept(plan, rate):
    total = 0.0
    for d1 in range(plan.n1 + 1):
        p1 = stats.binom.pmf(d1, plan.n1, rate)
        if d1 <= plan.c1:
            total += p1
        elif d1 < plan.r1:
            total += p1 * stats.binom.cdf(plan.c2 - d1, plan.n2, rate)
    return total


def brute_accept_lot(plan, lot, bad):
    total = 0.0
    for d1 in range(plan.n1 + 1):
        p1 = stats.hypergeom.pmf(d1, lot, bad, plan.n1)
        if p1 == 0.0:
            continue
        if d1 <= plan.c1:
            total += p1
        elif d1 < plan.r1:
            total += p1 * stats.hypergeom.cdf(plan.c2 - d1, lot - plan.n1, bad - d1, plan.n2)
    return total


@pytest.mark.parametrize("rate", [0.0, 0.005, 0.02, 0.05, 0.12, 1.0])
def test_double_oc_matches_enumeration(rate):
    plan = DoublePlan(50, 1, 4, 100, 3)
    assert plan.accept_prob(rate) == pytest.approx(brute_accept(plan, rate), rel=1e-9, abs=1e-14)
    need_second = sum(stats.binom.pmf(d, 50, rate) for d in (2, 3))
    assert plan.asn(rate) == pytest.approx(50 + 100 * need_second, rel=1e-9)


@pytest.mark.parametrize("bad", [0, 3, 9, 30, 300])
def test_double_oc_finite_batch_matches_enumeration(bad):
    plan = DoublePlan(40, 0, 3, 40, 2)
    lot = 300
    assert plan.accept_prob_lot(lot, bad) == pytest.approx(
        brute_accept_lot(plan, lot, bad), rel=1e-8, abs=1e-14
    )
    need_second = sum(stats.hypergeom.pmf(d, lot, bad, 40) for d in (1, 2))
    assert plan.asn_lot(lot, bad) == pytest.approx(40 + 40 * need_second, rel=1e-8)


def test_double_oc_against_seeded_simulation():
    plan = DoublePlan(30, 0, 3, 60, 2)
    lot, bad = 500, 20
    flags = [True] * bad + [False] * (lot - bad)
    rng = random.Random(5)
    trials = 4000
    accepted = used = 0
    for _ in range(trials):
        ok, n = plan.run(iter(rng.sample(flags, 90)))
        accepted += ok
        used += n
    expect = plan.accept_prob_lot(lot, bad)
    se = (expect * (1 - expect) / trials) ** 0.5
    assert abs(accepted / trials - expect) < 4 * se
    assert used / trials == pytest.approx(plan.asn_lot(lot, bad), rel=0.03)


def test_run_stops_after_the_first_sample_when_it_can():
    plan = DoublePlan(5, 0, 2, 5, 1)
    assert plan.run(iter([False] * 10)) == (True, 5)
    assert plan.run(iter([True, True, False, False, False] + [False] * 5)) == (False, 5)
    assert plan.run(iter([True, False, False, False, False] + [False] * 5)) == (True, 10)
    assert plan.run(iter([True, False, False, False, False] + [True] + [False] * 4)) == (False, 10)


@pytest.mark.parametrize(
    "acceptable,rejectable,alpha,beta",
    [(0.01, 0.06, 0.05, 0.10), (0.02, 0.10, 0.05, 0.10), (0.05, 0.20, 0.10, 0.10)],
)
def test_found_double_plan_meets_both_risks_and_saves_inspection(
    acceptable, rejectable, alpha, beta
):
    design = find_double_plan(acceptable, rejectable, alpha, beta)
    plan = design.plan
    assert brute_accept(plan, acceptable) >= 1 - alpha - 1e-12
    assert brute_accept(plan, rejectable) <= beta + 1e-12
    assert design.achieved_producer_risk == pytest.approx(1 - brute_accept(plan, acceptable))
    single = find_single_plan(acceptable, rejectable, alpha, beta)
    assert design.single_n == single.plan.n
    assert design.asn_acceptable < single.plan.n
    assert plan.n2 in (plan.n1, 2 * plan.n1)


def test_found_double_plan_is_the_best_in_its_search_space():
    acceptable, rejectable, alpha, beta = 0.02, 0.12, 0.05, 0.10
    design = find_double_plan(acceptable, rejectable, alpha, beta, ratios=(1,))
    c_top = find_single_plan(acceptable, rejectable, alpha, beta).plan.c + 3
    best = None
    for c2 in range(1, c_top + 1):
        for c1 in range(c2):
            for n1 in range(1, 150):
                plan = DoublePlan(n1, c1, c2 + 1, n1, c2)
                if brute_accept(plan, rejectable) <= beta:
                    if brute_accept(plan, acceptable) >= 1 - alpha:
                        asn = plan.asn(acceptable)
                        if best is None or asn < best[0] - 1e-12:
                            best = (asn, plan)
                    break
    assert best is not None
    assert design.asn_acceptable == pytest.approx(best[0], rel=1e-9)


def test_found_double_plan_finite_batch():
    lot = 400
    design = find_double_plan(0.01, 0.08, lot_size=lot)
    plan = design.plan
    assert design.model == "hypergeometric"
    assert plan.n1 + plan.n2 <= lot
    assert brute_accept_lot(plan, lot, bad_count_at_most(lot, 0.01)) >= 0.95 - 1e-12
    assert brute_accept_lot(plan, lot, bad_count_at_least(lot, 0.08)) <= 0.10 + 1e-12
    assert set(design.to_dict()) >= {"n1", "c1", "r1", "n2", "c2", "asn_acceptable"}


def test_validation():
    with pytest.raises(ValueError):
        DoublePlan(0, 0, 1, 5, 1)
    with pytest.raises(ValueError):
        DoublePlan(5, 2, 2, 5, 3)
    with pytest.raises(ValueError):
        DoublePlan(5, 0, 5, 5, 3)
    with pytest.raises(ValueError):
        DoublePlan(30, 0, 2, 30, 1).accept_prob_lot(50, 3)
    with pytest.raises(ValueError):
        find_double_plan(0.01, 0.06, ratios=(0,))
    with pytest.raises(ValueError):
        find_double_plan(0.01, 0.012, max_c2=1)


def test_double_plan_matches_enumeration_of_every_arrangement():
    plan = DoublePlan(4, 0, 2, 5, 1)
    lot, bad = 13, 3
    accepted = used = count = 0
    for places in itertools.combinations(range(lot), bad):
        order = [i in places for i in range(lot)]
        ok, n = plan.run(iter(order))
        accepted += ok
        used += n
        count += 1
    assert count == math.comb(lot, bad)
    assert plan.accept_prob_lot(lot, bad) == pytest.approx(accepted / count, rel=1e-10)
    assert plan.asn_lot(lot, bad) == pytest.approx(used / count, rel=1e-10)
