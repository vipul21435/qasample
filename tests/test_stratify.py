import math
import random

import pytest

from qasample.stratify import StratumSample, allocate, stratified_estimate


def test_proportional_allocation_adds_up_and_follows_sizes():
    sizes = {"a": 5000, "b": 3000, "c": 2000}
    out = allocate(sizes, 200)
    assert out == {"a": 100, "b": 60, "c": 40}
    out = allocate({"a": 3333, "b": 3333, "c": 3334}, 100)
    assert sum(out.values()) == 100
    assert max(out.values()) - min(out.values()) <= 1


def test_neyman_allocation_matches_the_formula():
    sizes = {"easy": 6000, "hard": 4000}
    rates = {"easy": 0.01, "hard": 0.20}
    out = allocate(sizes, 300, "neyman", rates)
    w = {h: sizes[h] * math.sqrt(rates[h] * (1 - rates[h])) for h in sizes}
    ideal = {h: 300 * w[h] / sum(w.values()) for h in sizes}
    assert sum(out.values()) == 300
    for h in sizes:
        assert abs(out[h] - ideal[h]) <= 1
    assert out["hard"] > out["easy"]


def test_neyman_beats_proportional_on_variance():
    sizes = {"easy": 6000, "hard": 4000}
    rates = {"easy": 0.01, "hard": 0.20}

    def variance(alloc):
        total = sum(sizes.values())
        return sum(
            (sizes[h] / total) ** 2 * rates[h] * (1 - rates[h]) / alloc[h] for h in sizes
        )

    assert variance(allocate(sizes, 300, "neyman", rates)) < variance(allocate(sizes, 300))


def test_allocation_respects_minimum_and_stratum_size():
    sizes = {"big": 10000, "tiny": 3, "small": 40}
    out = allocate(sizes, 50, minimum=5)
    assert out["tiny"] == 3
    assert out["small"] >= 5
    assert sum(out.values()) == 50
    # a stratum with prior rate 0 still gets its minimum under Neyman
    out = allocate({"a": 500, "b": 500}, 40, "neyman", {"a": 0.0, "b": 0.1}, minimum=2)
    assert out == {"a": 2, "b": 38}
    # a stratum cannot be given more than it holds
    out = allocate({"a": 10, "b": 1000}, 500, "neyman", {"a": 0.5, "b": 0.0001})
    assert out["a"] == 10 and out["b"] == 490
    # all prior rates zero: fall back to proportional
    assert allocate({"a": 100, "b": 300}, 40, "neyman", {"a": 0.0, "b": 0.0}) == {"a": 10, "b": 30}
    # the whole batch
    assert allocate({"a": 7, "b": 5}, 12) == {"a": 7, "b": 5}


def test_allocation_errors():
    with pytest.raises(ValueError):
        allocate({"a": 10}, 5, "optimal")
    with pytest.raises(ValueError):
        allocate({}, 5)
    with pytest.raises(ValueError):
        allocate({"a": 10}, 11)
    with pytest.raises(ValueError):
        allocate({"a": 10, "b": 10}, 1)
    with pytest.raises(ValueError):
        allocate({"a": 10}, 5, "neyman")
    with pytest.raises(ValueError):
        allocate({"a": 10, "b": 5}, 5, "neyman", {"a": 0.1})
    with pytest.raises(ValueError):
        allocate({"a": 10}, 5, "neyman", {"a": 1.5})
    with pytest.raises(ValueError):
        allocate({"a": -1}, 0)


def test_combined_estimate_by_hand():
    strata = [StratumSample("a", 600, 50, 5), StratumSample("b", 400, 100, 2)]
    est = stratified_estimate(strata)
    assert est.rate == pytest.approx(0.6 * 0.1 + 0.4 * 0.02)
    var = 0.6**2 * (1 - 50 / 600) * 0.1 * 0.9 / 49 + 0.4**2 * (1 - 100 / 400) * 0.02 * 0.98 / 99
    assert est.std_error == pytest.approx(math.sqrt(var))
    low, high = est.normal_interval
    assert low == pytest.approx(est.rate - 1.959964 * math.sqrt(var), rel=1e-5)
    assert est.exact_interval[0] <= est.rate <= est.exact_interval[1]
    assert est.bad_bounds == tuple(round(x * 1000) for x in est.exact_interval)
    assert high > est.rate > low
    assert est.to_dict()["strata"][0]["name"] == "a"


def test_clean_samples_still_give_an_upper_bound():
    est = stratified_estimate([StratumSample("a", 500, 60, 0), StratumSample("b", 500, 60, 0)])
    assert est.rate == 0.0
    assert est.normal_interval == (0.0, 0.0)
    assert est.exact_interval[0] == 0.0 and est.exact_interval[1] > 0.0


def test_full_inspection_is_exact():
    est = stratified_estimate([StratumSample("a", 30, 30, 3), StratumSample("b", 20, 20, 1)])
    assert est.rate == pytest.approx(4 / 50)
    assert est.std_error == 0.0
    assert est.bad_bounds == (4, 4)


def test_exact_interval_coverage_by_seeded_simulation():
    rng = random.Random(41)
    sizes = {"a": 400, "b": 250, "c": 150}
    bad = {"a": 12, "b": 30, "c": 3}
    truth = sum(bad.values()) / sum(sizes.values())
    pools = {h: [True] * bad[h] + [False] * (sizes[h] - bad[h]) for h in sizes}
    alloc = allocate(sizes, 160)
    trials, exact_hits, normal_hits = 800, 0, 0
    for _ in range(trials):
        strata = [
            StratumSample(h, sizes[h], alloc[h], sum(rng.sample(pools[h], alloc[h]))) for h in sizes
        ]
        est = stratified_estimate(strata, 0.9)
        exact_hits += est.exact_interval[0] <= truth <= est.exact_interval[1]
        normal_hits += est.normal_interval[0] <= truth <= est.normal_interval[1]
    assert exact_hits / trials >= 0.9
    assert normal_hits / trials >= 0.82  # approximate interval: allow some shortfall


def test_estimate_errors():
    with pytest.raises(ValueError):
        stratified_estimate([])
    with pytest.raises(ValueError):
        stratified_estimate([StratumSample("a", 10, 5, 1)], confidence=1.0)
    with pytest.raises(ValueError):
        StratumSample("a", 10, 0, 0)
    with pytest.raises(ValueError):
        StratumSample("a", 10, 11, 0)
    with pytest.raises(ValueError):
        StratumSample("a", 10, 5, 6)
