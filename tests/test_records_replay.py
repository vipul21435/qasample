import json
import random

import pytest

from qasample.double import DoublePlan
from qasample.records import load_records, read_rows
from qasample.replay import _random_order, expected, replay
from qasample.sequential import SequentialPlan
from qasample.single import SinglePlan


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_jsonl_with_categories_and_strata(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_jsonl(
        path,
        [
            {"id": 1, "result": "pass", "source": "web"},
            {"id": 2, "result": "wrong label", "source": "web"},
            {"id": 3, "result": "OK", "source": "app"},
            {"id": 4, "result": "Wrong Label", "source": "app"},
            {"id": 5, "result": "truncated", "source": "app"},
            {"id": 6, "result": None, "source": "app"},
        ],
    )
    rec = load_records(path, "result", stratum="source")
    assert (rec.n, rec.bad) == (6, 3)
    assert rec.categories == {"wrong label": 2, "truncated": 1}
    assert rec.by_stratum() == {"web": (2, 1), "app": (4, 2)}


def test_load_csv_and_boolean_like_values(tmp_path):
    path = tmp_path / "audit.csv"
    path.write_text("id,is_error\n1,0\n2,1\n3,false\n4,TRUE\n5,\n", encoding="utf-8")
    rec = load_records(path, "is_error")
    assert rec.flags == [False, True, False, True, False]
    with pytest.raises(ValueError):
        rec.by_stratum()


def test_json_booleans_and_numbers(tmp_path):
    path = tmp_path / "audit.jsonl"
    write_jsonl(path, [{"bad": True}, {"bad": False}, {"bad": 0}, {"bad": 1.0}, {"bad": 0.0}])
    assert load_records(path, "bad").flags == [True, False, False, True, False]


def test_explicit_bad_and_good_values(tmp_path):
    path = tmp_path / "audit.csv"
    path.write_text("verdict\nreject\naccept\nunsure\n", encoding="utf-8")
    assert load_records(path, "verdict", bad_values={"Reject"}).flags == [True, False, False]
    assert load_records(path, "verdict", good_values={"accept"}).flags == [True, False, True]


def test_read_errors(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_rows(path)
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        read_rows(path)
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        load_records(path, "a")
    write_jsonl(path, [{"a": 1}])
    with pytest.raises(ValueError, match="no column 'b'"):
        load_records(path, "b")
    with pytest.raises(ValueError, match="no column 's'"):
        load_records(path, "a", stratum="s")


def batch(size, bad):
    flags = [True] * bad + [False] * (size - bad)
    random.Random(1).shuffle(flags)
    return flags


def test_random_order_is_a_permutation_in_every_branch():
    flags = list(range(50))
    for chunk in (5, 30, 50, 80):
        seen = list(_random_order(flags, random.Random(chunk), chunk))
        assert sorted(seen) == flags


@pytest.mark.parametrize(
    "plan",
    [
        SinglePlan(60, 2),
        DoublePlan(30, 0, 3, 60, 2),
        SequentialPlan(0.01, 0.08),
        SequentialPlan(0.01, 0.08, max_n=70),
    ],
)
def test_replay_agrees_with_the_exact_finite_batch_probability(plan):
    flags = batch(2000, 60)
    result = replay(flags, plan, audits=2500, seed=7)
    assert (result.lot_size, result.bad) == (2000, 60)
    assert result.true_rate == pytest.approx(0.03)
    assert 0.02 < result.expected_accept < 0.98
    assert abs(result.accept_rate - result.expected_accept) < 4 * result.accept_se
    assert result.mean_inspected == pytest.approx(result.expected_inspected, rel=0.06)
    assert result.binomial_accept == pytest.approx(result.expected_accept, abs=0.03)
    assert result.accepted == round(result.accept_rate * result.audits)
    assert result.to_dict()["audits"] == 2500


def test_replay_is_reproducible_and_seed_sensitive():
    flags = batch(500, 25)
    a = replay(flags, SinglePlan(40, 1), audits=300, seed=3)
    b = replay(flags, SinglePlan(40, 1), audits=300, seed=3)
    c = replay(flags, SinglePlan(40, 1), audits=300, seed=4)
    assert a == b
    assert a.accepted != c.accepted or a != c


def test_replay_edge_cases():
    clean = replay([False] * 100, SinglePlan(100, 0), audits=5)
    assert clean.accept_rate == 1.0 and clean.expected_accept == 1.0
    with pytest.raises(ValueError):
        replay([False] * 10, SinglePlan(11, 0))
    with pytest.raises(ValueError):
        replay([False] * 10, DoublePlan(6, 0, 2, 6, 1))
    with pytest.raises(ValueError):
        replay([], SinglePlan(1, 0))
    with pytest.raises(ValueError):
        replay([False], SinglePlan(1, 0), audits=0)
    # a sequential plan on a batch smaller than its run simply ends with the batch
    small = replay([False] * 20, SequentialPlan(0.01, 0.08), audits=10)
    assert small.mean_inspected == 20 and small.accept_rate == 1.0


def test_expected_for_each_plan_type():
    accept, used, binom = expected(SinglePlan(50, 1), 1000, 20)
    assert used == 50 and 0 < accept < 1 and 0 < binom < 1
    accept, used, _ = expected(DoublePlan(30, 0, 3, 60, 2), 1000, 20)
    assert 30 < used < 90
    accept, used, _ = expected(SequentialPlan(0.01, 0.08), 1000, 20)
    assert 0 < accept < 1 and used > 1
