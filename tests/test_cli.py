import json

import pytest

from qasample.cli import main
from qasample.single import find_single_plan


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_plan_markdown(capsys):
    code, out, _ = run(capsys, "plan", "--acceptable", "0.01", "--rejectable", "0.06")
    assert code == 0
    design = find_single_plan(0.01, 0.06)
    assert f"n = {design.plan.n} records" in out and f"c = {design.plan.c} are bad" in out
    assert "| batch error rate |" in out


def test_plan_json_finite_batch(capsys):
    code, out, _ = run(
        capsys, "plan", "--acceptable", "0.01", "--rejectable", "0.06", "--batch", "500",
        "--format", "json",
    )
    assert code == 0
    data = json.loads(out)["single"]
    assert data["model"] == "hypergeometric"
    assert data["lot_size"] == 500
    assert data["achieved_consumer_risk"] <= 0.10
    assert 0 < data["aoql"] < 0.06


def test_plan_reports_bad_input(capsys):
    code, _, err = run(capsys, "plan", "--acceptable", "0.06", "--rejectable", "0.01")
    assert code == 2
    assert "error" in err


def test_oc_table_and_json(capsys):
    code, out, _ = run(capsys, "oc", "--n", "89", "--c", "2", "--rates", "0.01,0.06")
    assert code == 0
    assert out.count("\n") >= 4
    code, out, _ = run(
        capsys, "oc", "--n", "50", "--c", "1", "--batch", "400", "--points", "5",
        "--format", "json",
    )
    data = json.loads(out)
    assert len(data["curve"]) == 5
    assert data["curve"][0]["accept"] == pytest.approx(1.0)


def test_rate_argument_is_validated(capsys):
    with pytest.raises(SystemExit):
        main(["plan", "--acceptable", "1.5", "--rejectable", "0.5"])
    capsys.readouterr()
