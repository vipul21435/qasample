import itertools
import math
import random

import pytest

from qasample.sequential import SequentialPlan

PLAN = SequentialPlan(0.01, 0.06, 0.05, 0.10)


def test_lines_match_the_textbook_formulas():
    p1, p2, alpha, beta = 0.01, 0.06, 0.05, 0.10
    k = math.log(p2 * (1 - p1) / (p1 * (1 - p2)))
    assert PLAN.h_accept == pytest.approx(math.log((1 - alpha) / beta) / k)
    assert PLAN.h_reject == pytest.approx(math.log((1 - beta) / alpha) / k)
    assert PLAN.slope == pytest.approx(math.log((1 - p1) / (1 - p2)) / k)
    assert p1 < PLAN.slope < p2


def test_decisions_agree_with_the_likelihood_ratio():
    log_a = math.log((1 - 0.10) / 0.05)
    log_b = math.log(0.10 / (1 - 0.05))
    for n in range(1, 250):
        for d in range(0, min(n, 12) + 1):
            llr = d * math.log(0.06 / 0.01) + (n - d) * math.log(0.94 / 0.99)
            if llr <= log_b:
                expect = "accept"
            elif llr >= log_a:
                expect = "reject"
            else:
                expect = "continue"
            if min(abs(llr - log_a), abs(llr - log_b)) > 1e-6:
                assert PLAN.decide(n, d) == expect


def test_wald_closed_forms_at_the_five_standard_points():
    h0, h1, s = PLAN.h_accept, PLAN.h_reject, PLAN.slope
    p0, p1, alpha, beta = 0.01, 0.06, 0.05, 0.10
    assert PLAN.wald_oc(0.0) == 1.0
    assert PLAN.wald_oc(1.0) == 0.0
    assert PLAN.wald_oc(p0) == pytest.approx(1 - alpha, abs=1e-9)
    assert PLAN.wald_oc(p1) == pytest.approx(beta, abs=1e-9)
    assert PLAN.wald_oc(s) == pytest.approx(h1 / (h0 + h1), abs=1e-6)
    assert PLAN.wald_asn(0.0) == pytest.approx(h0 / s)
    assert PLAN.wald_asn(1.0) == pytest.approx(h1 / (1 - s))
    assert PLAN.wald_asn(p0) == pytest.approx(((1 - alpha) * h0 - alpha * h1) / (s - p0))
    assert PLAN.wald_asn(p1) == pytest.approx(((1 - beta) * h1 - beta * h0) / (p1 - s))
    assert PLAN.wald_asn(s) == pytest.approx(h0 * h1 / (s * (1 - s)))
    # continuity of the approximation around the slope
    assert PLAN.wald_asn(s + 1e-5) == pytest.approx(PLAN.wald_asn(s), rel=1e-2)
    assert PLAN.wald_oc(1e-9) == pytest.approx(1.0, abs=1e-6)
    assert PLAN.wald_oc(1 - 1e-9) == pytest.approx(0.0, abs=1e-6)


def test_exact_risks_respect_walds_inequalities():
    for p0, p1, alpha, beta in [(0.01, 0.06, 0.05, 0.10), (0.02, 0.08, 0.10, 0.05),
                                (0.05, 0.15, 0.01, 0.01), (0.002, 0.02, 0.05, 0.20)]:
        plan = SequentialPlan(p0, p1, alpha, beta)
        at_p0, at_p1 = plan.exact(p0), plan.exact(p1)
        assert at_p0.undecided < 1e-12 and at_p1.undecided < 1e-12
        real_alpha, real_beta = at_p0.reject, at_p1.accept
        assert real_alpha <= alpha / (1 - beta) + 1e-12
        assert real_beta <= beta / (1 - alpha) + 1e-12
        assert real_alpha + real_beta <= alpha + beta + 1e-12


def test_exact_is_close_to_walds_approximation():
    for rate in (0.005, 0.01, 0.02, PLAN.slope, 0.04, 0.06, 0.1):
        exact = PLAN.exact(rate)
        assert exact.accept + exact.reject == pytest.approx(1.0, abs=1e-10)
        assert exact.accept == pytest.approx(PLAN.wald_oc(rate), abs=0.06)
        # the approximation ignores the overshoot past the lines, so the real
        # average sample number is somewhat larger, never much smaller
        assert 0.98 * PLAN.wald_asn(rate) <= exact.asn <= 1.35 * PLAN.wald_asn(rate)


def test_exact_at_the_extremes():
    clean = PLAN.exact(0.0)
    assert clean.accept == 1.0
    assert clean.asn == PLAN.first_accept_n()
    assert PLAN.decide(PLAN.first_accept_n(), 0) == "accept"
    assert PLAN.decide(PLAN.first_accept_n() - 1, 0) == "continue"
    rotten = PLAN.exact(1.0)
    assert rotten.reject == 1.0
    assert rotten.asn == math.ceil(PLAN.h_reject / (1 - PLAN.slope) - 1e-9)


def simulate(plan, rate, trials, seed):
    rng = random.Random(seed)
    accepted = used = 0

    def stream():
        while True:
            yield rng.random() < rate

    for _ in range(trials):
        ok, n = plan.run(stream())
        accepted += ok
        used += n
    return accepted / trials, used / trials


@pytest.mark.parametrize("rate", [0.01, 0.03, 0.06])
def test_exact_matches_seeded_simulation(rate):
    trials = 3000
    got_accept, got_asn = simulate(PLAN, rate, trials, seed=17)
    exact = PLAN.exact(rate)
    se = max((exact.accept * (1 - exact.accept) / trials) ** 0.5, 1e-3)
    assert abs(got_accept - exact.accept) < 4 * se
    assert got_asn == pytest.approx(exact.asn, rel=0.08)


def test_truncation_forces_a_decision_and_changes_the_risks():
    cut = SequentialPlan(0.01, 0.06, 0.05, 0.10, max_n=80)
    for rate in (0.0, 0.01, 0.03, 0.06, 0.5):
        res = cut.exact(rate)
        assert res.undecided == 0.0
        assert res.accept + res.reject == pytest.approx(1.0, abs=1e-12)
        assert res.max_n <= 80
        assert res.asn <= 80
    # cutting the run short costs something at one end or the other
    full_alpha, full_beta = PLAN.exact(0.01).reject, PLAN.exact(0.06).accept
    cut_alpha, cut_beta = cut.exact(0.01).reject, cut.exact(0.06).accept
    assert cut_alpha + cut_beta > full_alpha + full_beta
    got_accept, got_asn = simulate(cut, 0.03, 3000, seed=3)
    res = cut.exact(0.03)
    assert abs(got_accept - res.accept) < 4 * (res.accept * (1 - res.accept) / 3000) ** 0.5
    assert got_asn == pytest.approx(res.asn, rel=0.08)


def test_truncation_rule_and_override():
    cut = SequentialPlan(0.01, 0.06, max_n=50)
    limit = math.floor(cut.slope * 50 + 1e-9)
    assert cut.decide(50, limit) == "accept"
    assert cut.decide(50, limit + 1) == "reject"
    assert cut.decide(49, limit + 1) == "continue"
    forced = SequentialPlan(0.01, 0.06, max_n=50, truncation_accept=0)
    assert forced.decide(50, 1) == "reject"
    assert forced.exact(0.02).accept < cut.exact(0.02).accept


def test_finite_batch_exact_matches_simulation_and_ends_with_the_batch():
    lot, bad = 300, 9
    res = PLAN.exact(lot_size=lot, bad=bad)
    assert res.undecided == 0.0
    assert res.accept + res.reject == pytest.approx(1.0, abs=1e-12)
    flags = [True] * bad + [False] * (lot - bad)
    rng = random.Random(23)
    trials = 3000
    accepted = used = 0
    for _ in range(trials):
        order = flags[:]
        rng.shuffle(order)
        ok, n = PLAN.run(iter(order))
        accepted += ok
        used += n
    se = (res.accept * (1 - res.accept) / trials) ** 0.5
    assert abs(accepted / trials - res.accept) < 4 * se
    assert used / trials == pytest.approx(res.asn, rel=0.08)
    tiny = PLAN.exact(lot_size=20, bad=0)
    assert tiny.max_n == 20 and tiny.accept == 1.0


def test_table_lists_numbers_only_where_a_decision_is_possible():
    rows = PLAN.table(60)
    assert rows[0] == {"n": 1, "accept": None, "reject": None}
    first = PLAN.first_accept_n()
    assert rows[first - 1]["accept"] == 0
    assert rows[first - 2]["accept"] is None
    assert all(r["reject"] is None or r["reject"] <= r["n"] for r in rows)
    assert any(r["reject"] is not None for r in rows)


def test_validation_and_dict():
    with pytest.raises(ValueError):
        SequentialPlan(0.06, 0.01)
    with pytest.raises(ValueError):
        SequentialPlan(0.0, 0.01)
    with pytest.raises(ValueError):
        SequentialPlan(0.01, 0.06, 0.0, 0.1)
    with pytest.raises(ValueError):
        SequentialPlan(0.01, 0.06, 0.6, 0.6)
    with pytest.raises(ValueError):
        SequentialPlan(0.01, 0.06, max_n=0)
    with pytest.raises(ValueError):
        PLAN.exact(1.5)
    with pytest.raises(ValueError):
        PLAN.exact(lot_size=10, bad=11)
    assert PLAN.to_dict()["slope"] == pytest.approx(PLAN.slope)
    assert set(PLAN.exact(0.02).to_dict()) == {"accept", "reject", "undecided", "asn", "max_n"}


def test_exact_matches_enumeration_of_every_path():
    # every pass/fail sequence of length 12, weighted by its binomial probability
    plan = SequentialPlan(0.05, 0.40, 0.10, 0.10, max_n=12)
    for rate in (0.05, 0.2, 0.4):
        accept = asn = 0.0
        for path in itertools.product((False, True), repeat=12):
            weight = math.prod(rate if flag else 1 - rate for flag in path)
            ok, used = plan.run(iter(path))
            accept += weight * ok
            asn += weight * used
        res = plan.exact(rate)
        assert res.accept == pytest.approx(accept, rel=1e-10)
        assert res.asn == pytest.approx(asn, rel=1e-10)
        assert res.undecided == pytest.approx(0.0, abs=1e-15)


def test_finite_batch_exact_matches_enumeration_of_every_arrangement():
    # 14 records with 4 bad: every placement of the bad ones is equally likely
    plan = SequentialPlan(0.05, 0.40, 0.10, 0.10)
    lot, bad = 14, 4
    accept = asn = 0.0
    count = 0
    for places in itertools.combinations(range(lot), bad):
        order = [i in places for i in range(lot)]
        ok, used = plan.run(iter(order))
        accept += ok
        asn += used
        count += 1
    res = plan.exact(lot_size=lot, bad=bad)
    assert count == math.comb(lot, bad)
    assert res.accept == pytest.approx(accept / count, rel=1e-10)
    assert res.asn == pytest.approx(asn / count, rel=1e-10)


def test_very_close_rates_stop_after_bounded_work(monkeypatch):
    import qasample.sequential as module

    monkeypatch.setattr(module, "_WORK_LIMIT", 20_000)
    plan = SequentialPlan(0.01, 0.0101)
    res = plan.exact(0.01005)
    assert res.undecided > 0.5
    assert res.accept + res.reject + res.undecided == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError):
        plan.exact(lot_size=0, bad=0)
