import math

import pytest
from scipy import stats

from qasample.dist import (
    binom_cdf,
    binom_pmf,
    binom_upper,
    hypergeom_cdf,
    hypergeom_pmf,
    hypergeom_support,
    hypergeom_upper,
)

BINOM_CASES = [
    (n, p)
    for n in (1, 7, 50, 300, 4000, 60000)
    for p in (1e-6, 0.003, 0.05, 0.31, 0.5, 0.87, 0.9999)
]


@pytest.mark.parametrize("n,p", BINOM_CASES)
def test_binomial_matches_scipy(n, p):
    ks = sorted({0, 1, n // 7, int(n * p), int(n * p) + 1, n // 2, n - 1, n})
    for k in ks:
        assert binom_pmf(k, n, p) == pytest.approx(stats.binom.pmf(k, n, p), rel=1e-9, abs=1e-300)
        assert binom_cdf(k, n, p) == pytest.approx(stats.binom.cdf(k, n, p), rel=1e-9, abs=1e-300)
        assert binom_upper(k, n, p) == pytest.approx(
            stats.binom.sf(k - 1, n, p), rel=1e-9, abs=1e-300
        )


def test_binomial_small_tails_are_not_lost():
    # far tails must come out as small positive numbers, not as 0 or 1 - 1
    assert binom_cdf(2, 5000, 0.05) == pytest.approx(stats.binom.cdf(2, 5000, 0.05), rel=1e-9)
    assert binom_upper(400, 5000, 0.01) == pytest.approx(stats.binom.sf(399, 5000, 0.01), rel=1e-9)


def test_binomial_edges():
    assert binom_pmf(0, 0, 0.3) == 1.0
    assert binom_cdf(-1, 10, 0.3) == 0.0
    assert binom_cdf(10, 10, 0.3) == 1.0
    assert binom_cdf(3, 10, 0.0) == 1.0
    assert binom_cdf(3, 10, 1.0) == 0.0
    assert binom_upper(0, 10, 0.3) == 1.0
    assert binom_upper(11, 10, 0.3) == 0.0
    assert binom_upper(3, 10, 0.0) == 0.0
    assert binom_upper(3, 10, 1.0) == 1.0
    assert binom_pmf(3, 10, 0.0) == 0.0
    assert binom_pmf(10, 10, 1.0) == 1.0
    with pytest.raises(ValueError):
        binom_pmf(1, -1, 0.5)
    with pytest.raises(ValueError):
        binom_cdf(1, 5, 1.5)


def test_binomial_pmf_sums_to_one():
    for n, p in [(13, 0.2), (200, 0.7), (6000, 0.01)]:
        assert math.fsum(binom_pmf(k, n, p) for k in range(n + 1)) == pytest.approx(1.0, abs=1e-9)


HYPER_CASES = [
    (20, 5, 8),
    (20, 0, 8),
    (20, 20, 8),
    (100, 30, 100),
    (500, 7, 60),
    (500, 480, 60),
    (5000, 150, 400),
    (50000, 2916, 300),
    (200000, 1000, 5000),
]


@pytest.mark.parametrize("lot,bad,n", HYPER_CASES)
def test_hypergeometric_matches_scipy(lot, bad, n):
    lo, hi = hypergeom_support(lot, bad, n)
    mean = n * bad // lot
    ks = sorted({lo - 1, lo, lo + 1, mean, mean + 1, (lo + hi) // 2, hi - 1, hi, hi + 1})
    ref = stats.hypergeom(lot, bad, n)
    for k in ks:
        assert hypergeom_pmf(k, lot, bad, n) == pytest.approx(ref.pmf(k), rel=1e-8, abs=1e-300)
        assert hypergeom_cdf(k, lot, bad, n) == pytest.approx(ref.cdf(k), rel=1e-8, abs=1e-300)
        assert hypergeom_upper(k, lot, bad, n) == pytest.approx(
            ref.sf(k - 1), rel=1e-8, abs=1e-300
        )


def test_hypergeometric_exact_small_case():
    # 2 bad in a batch of 5, draw 2: P(0) = C(3,2)/C(5,2) = 3/10, P(1) = 6/10, P(2) = 1/10
    assert hypergeom_pmf(0, 5, 2, 2) == pytest.approx(0.3)
    assert hypergeom_pmf(1, 5, 2, 2) == pytest.approx(0.6)
    assert hypergeom_pmf(2, 5, 2, 2) == pytest.approx(0.1)
    assert hypergeom_cdf(1, 5, 2, 2) == pytest.approx(0.9)
    assert hypergeom_upper(1, 5, 2, 2) == pytest.approx(0.7)


def test_hypergeometric_rejects_bad_arguments():
    with pytest.raises(ValueError):
        hypergeom_pmf(0, 10, 11, 3)
    with pytest.raises(ValueError):
        hypergeom_cdf(0, 10, 3, 11)
    with pytest.raises(ValueError):
        hypergeom_pmf(0, -1, 0, 0)


def test_whole_batch_sample_is_deterministic():
    assert hypergeom_pmf(4, 30, 4, 30) == 1.0
    assert hypergeom_cdf(3, 30, 4, 30) == 0.0
    assert hypergeom_upper(4, 30, 4, 30) == 1.0
