import math
import random

import pytest
from scipy import special, stats

from qasample.bounds import (
    betabinom_cdf,
    betainc,
    clopper_pearson,
    hypergeom_bound,
    prob_meets_target,
    rule_of_three,
    wilson,
    zero_defect_sample_size,
)

CASES = [(0, 10), (1, 10), (10, 10), (3, 50), (0, 300), (7, 1000), (250, 500), (4999, 5000)]


@pytest.mark.parametrize("bad,n", CASES)
@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99])
def test_clopper_pearson_matches_scipy(bad, n, confidence):
    ref = stats.binomtest(bad, n).proportion_ci(confidence_level=confidence, method="exact")
    low, high = clopper_pearson(bad, n, confidence)
    assert low == pytest.approx(ref.low, rel=1e-8, abs=1e-12)
    assert high == pytest.approx(ref.high, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("bad,n", CASES)
def test_one_sided_bounds_match_scipy(bad, n):
    up = stats.binomtest(bad, n, alternative="less").proportion_ci(0.95, method="exact")
    lo = stats.binomtest(bad, n, alternative="greater").proportion_ci(0.95, method="exact")
    assert clopper_pearson(bad, n, 0.95, "upper") == pytest.approx((0.0, up.high), rel=1e-8)
    assert clopper_pearson(bad, n, 0.95, "lower") == pytest.approx((lo.low, 1.0), rel=1e-8)


@pytest.mark.parametrize("bad,n", CASES)
def test_wilson_matches_scipy(bad, n):
    ref = stats.binomtest(bad, n).proportion_ci(confidence_level=0.95, method="wilson")
    low, high = wilson(bad, n)
    assert low == pytest.approx(ref.low, rel=1e-9, abs=1e-12)
    assert high == pytest.approx(ref.high, rel=1e-9, abs=1e-12)


def test_rule_of_three():
    out = rule_of_three(100)
    assert out["approx"] == pytest.approx(-math.log(0.05) / 100)
    assert out["approx"] == pytest.approx(3 / 100, rel=0.01)
    assert out["exact"] == pytest.approx(1 - 0.05 ** (1 / 100))
    assert out["exact"] == pytest.approx(clopper_pearson(0, 100, 0.95, "upper")[1])
    assert out["exact"] < out["approx"]
    with pytest.raises(ValueError):
        rule_of_three(0)
    with pytest.raises(ValueError):
        rule_of_three(10, 1.0)


def test_zero_defect_sample_size():
    n = zero_defect_sample_size(0.01)
    assert n == 299
    assert 0.99**n <= 0.05 < 0.99 ** (n - 1)
    assert zero_defect_sample_size(0.5, 0.5) == 1
    with pytest.raises(ValueError):
        zero_defect_sample_size(0.0)
    with pytest.raises(ValueError):
        zero_defect_sample_size(0.1, 0.0)


def test_clopper_pearson_coverage_by_seeded_simulation():
    rng = random.Random(11)
    n = 60
    for rate in (0.01, 0.07, 0.3):
        hits = 0
        trials = 1500
        for _ in range(trials):
            bad = sum(rng.random() < rate for _ in range(n))
            low, high = clopper_pearson(bad, n, 0.9)
            hits += low <= rate <= high
        # exact intervals are conservative: coverage should not fall below nominal
        # by more than simulation noise (3 standard errors is about 0.023)
        assert hits / trials >= 0.9 - 0.025


def test_clopper_pearson_exact_coverage_by_enumeration():
    n = 40
    table = [clopper_pearson(k, n, 0.95) for k in range(n + 1)]
    for rate in [i / 200 for i in range(1, 200)]:
        cover = sum(
            stats.binom.pmf(k, n, rate) for k, (lo, hi) in enumerate(table) if lo <= rate <= hi
        )
        assert cover >= 0.95 - 1e-9


def brute_hypergeom_bound(bad, n, lot, alpha_tail):
    possible = range(bad, lot - (n - bad) + 1)
    upper = max(d for d in possible if stats.hypergeom.cdf(bad, lot, d, n) > alpha_tail)
    lower = min(d for d in possible if stats.hypergeom.sf(bad - 1, lot, d, n) > alpha_tail)
    return lower, upper


@pytest.mark.parametrize(
    "bad,n,lot", [(0, 20, 100), (2, 30, 200), (5, 5, 40), (1, 50, 60), (0, 60, 60), (7, 80, 900)]
)
def test_hypergeom_bound_matches_brute_force(bad, n, lot):
    assert hypergeom_bound(bad, n, lot, 0.95) == brute_hypergeom_bound(bad, n, lot, 0.025)
    one_up = hypergeom_bound(bad, n, lot, 0.95, "upper")
    one_lo = hypergeom_bound(bad, n, lot, 0.95, "lower")
    brute = brute_hypergeom_bound(bad, n, lot, 0.05)
    assert one_up == (bad, brute[1])
    assert one_lo == (brute[0], lot - (n - bad))


def test_hypergeom_bound_exact_coverage():
    lot, n = 120, 25
    table = [hypergeom_bound(k, n, lot, 0.9) for k in range(n + 1)]
    for total in range(lot + 1):
        cover = 0.0
        for k, (lo, hi) in enumerate(table):
            if lo <= total <= hi:
                cover += stats.hypergeom.pmf(k, lot, total, n)
        assert cover >= 0.9 - 1e-9


def test_whole_batch_inspected_leaves_no_doubt():
    assert hypergeom_bound(4, 50, 50) == (4, 4)
    with pytest.raises(ValueError):
        hypergeom_bound(1, 60, 50)


def test_finite_batch_bound_is_tighter_than_binomial():
    lot, n, bad = 400, 200, 6
    _, upper = hypergeom_bound(bad, n, lot, 0.95, "upper")
    assert upper / lot < clopper_pearson(bad, n, 0.95, "upper")[1]


@pytest.mark.parametrize(
    "a,b,x",
    [(1, 1, 0.3), (0.5, 0.5, 0.01), (3, 200, 0.02), (40, 7, 0.9), (2500.5, 47500.5, 0.05),
     (1, 3000, 0.001), (12, 12, 0.5)],
)
def test_betainc_matches_scipy(a, b, x):
    assert betainc(a, b, x) == pytest.approx(special.betainc(a, b, x), rel=1e-9, abs=1e-14)


def test_betainc_edges():
    assert betainc(2, 3, 0.0) == 0.0
    assert betainc(2, 3, 1.0) == 1.0
    with pytest.raises(ValueError):
        betainc(0, 1, 0.5)


def test_prob_meets_target_uniform_prior_is_a_binomial_tail():
    # with a uniform prior, P(p <= t | bad of n) = P(Binomial(n + 1, t) >= bad + 1)
    for bad, n, target in [(0, 100, 0.03), (3, 200, 0.02), (10, 150, 0.1)]:
        expect = stats.binom.sf(bad, n + 1, target)
        assert prob_meets_target(bad, n, target) == pytest.approx(expect, rel=1e-9)


def test_prob_meets_target_other_prior_matches_scipy():
    got = prob_meets_target(2, 120, 0.04, prior=(0.5, 0.5))
    assert got == pytest.approx(stats.beta.cdf(0.04, 2.5, 118.5), rel=1e-9)


def test_prob_meets_target_finite_batch_matches_scipy_betabinom():
    bad, n, lot, target = 2, 80, 500, 0.04
    allowed = math.floor(lot * target) - bad
    expect = stats.betabinom.cdf(allowed, lot - n, 1 + bad, 1 + n - bad)
    assert prob_meets_target(bad, n, target, lot_size=lot) == pytest.approx(expect, rel=1e-8)
    assert betabinom_cdf(-1, 10, 1, 1) == 0.0
    assert betabinom_cdf(10, 10, 1, 1) == 1.0
    # already more bad records in the sample than the target allows in the whole batch
    assert prob_meets_target(30, 80, 0.04, lot_size=500) == 0.0
    # the whole batch inspected: the answer is certain
    assert prob_meets_target(2, 100, 0.02, lot_size=100) == 1.0
    assert prob_meets_target(3, 100, 0.02, lot_size=100) == 0.0


def test_argument_checks():
    with pytest.raises(ValueError):
        clopper_pearson(5, 4)
    with pytest.raises(ValueError):
        clopper_pearson(0, 0)
    with pytest.raises(ValueError):
        clopper_pearson(1, 4, 1.2)
    with pytest.raises(ValueError):
        clopper_pearson(1, 4, 0.9, "sideways")
    with pytest.raises(ValueError):
        wilson(1, 4, 0.0)
    with pytest.raises(ValueError):
        prob_meets_target(1, 4, 1.5)
    with pytest.raises(ValueError):
        prob_meets_target(1, 4, 0.5, prior=(0, 1))
    with pytest.raises(ValueError):
        prob_meets_target(1, 40, 0.5, lot_size=10)
