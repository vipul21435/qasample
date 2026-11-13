import json

import pytest

from qasample.cli import main
from qasample.double import find_double_plan
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


def write_audit(tmp_path, rows, name="audit.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return str(path)


def test_plan_double_and_inspector(capsys):
    code, out, _ = run(
        capsys, "plan", "--acceptable", "0.01", "--rejectable", "0.06", "--double",
        "--miss-rate", "0.3", "--format", "json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["double"]["asn_acceptable"] < data["single"]["n"]
    assert data["inspector"]["naive_risks"]["consumer_risk"] > 0.10
    assert data["inspector"]["adjusted"]["consumer_risk"] <= 0.10
    assert data["inspector"]["adjusted"]["n"] > data["single"]["n"]
    code, out, _ = run(
        capsys, "plan", "--acceptable", "0.01", "--rejectable", "0.06", "--double",
        "--false-alarm-rate", "0.01",
    )
    assert "Double plan" in out and "designed for this inspector" in out
    code, _, err = run(
        capsys, "plan", "--acceptable", "0.01", "--rejectable", "0.06", "--batch", "500",
        "--miss-rate", "0.1",
    )
    assert code == 2 and "binomial" in err


def test_oc_chart(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    target = tmp_path / "oc.png"
    code, out, _ = run(capsys, "oc", "--n", "60", "--c", "1", "--chart", str(target))
    assert code == 0
    assert target.read_bytes()[:4] == b"\x89PNG"
    assert "Chart written" in out
    code, _, err = run(capsys, "oc", "--n", "60", "--c", "1", "--points", "1")
    assert code == 2


def test_bound_from_counts(capsys):
    code, out, _ = run(
        capsys, "bound", "--n", "200", "--bad", "0", "--batch", "5000", "--target", "0.02",
        "--format", "json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["clopper_pearson"][0] == 0.0
    assert data["upper_bound"] == pytest.approx(1 - 0.05 ** (1 / 200))
    assert data["zero_defects"]["approx"] == pytest.approx(0.015, rel=0.01)
    assert data["batch"]["bad_left"][0] == 0
    assert data["batch"]["bad_left_upper"] <= data["batch"]["bad_left"][1]
    assert 0.9 < data["target"]["probability"] <= 1.0
    code, out, _ = run(capsys, "bound", "--n", "200", "--bad", "0")
    assert "rule of three" in out
    code, _, err = run(capsys, "bound", "--n", "200")
    assert code == 2 and "--input" in err


def test_bound_from_file_with_inspector(tmp_path, capsys):
    rows = [{"result": "pass"}] * 45 + [{"result": "wrong label"}] * 4 + [{"result": "empty"}]
    path = write_audit(tmp_path, rows)
    code, out, _ = run(
        capsys, "bound", "--input", path, "--miss-rate", "0.2", "--batch", "400",
    )
    assert code == 0
    assert "5 bad of 50 inspected" in out
    assert "| wrong label | 4 |" in out
    assert "Corrected for an inspector" in out
    assert "Batch of 400" in out
    code, out, _ = run(
        capsys, "bound", "--input", path, "--bad-values", "empty", "--format", "json"
    )
    assert json.loads(out)["bad"] == 1


def test_sequential_command(capsys):
    code, out, _ = run(capsys, "sequential", "--acceptable", "0.01", "--rejectable", "0.06")
    assert code == 0
    assert "Sequential plan" in out and "Wald's approximation" in out
    code, out, _ = run(
        capsys, "sequential", "--acceptable", "0.01", "--rejectable", "0.06", "--truncate", "90",
        "--batch", "600", "--rows", "20", "--format", "json",
    )
    data = json.loads(out)
    assert len(data["numbers"]) == 20
    assert data["plan"]["max_n"] == 90
    for row in data["operating_characteristic"]:
        assert row["longest_run"] <= 90
        assert 0.0 <= row["accept"] <= 1.0
    code, out, _ = run(
        capsys, "sequential", "--acceptable", "0.01", "--rejectable", "0.06", "--truncate", "90",
    )
    assert "Truncated at 90" in out
    code, _, err = run(capsys, "sequential", "--acceptable", "0", "--rejectable", "0.06")
    assert code == 2


def test_stratify_allocation(capsys):
    code, out, _ = run(
        capsys, "stratify", "--sizes", "web=6000,app=4000", "--total", "300", "--method",
        "neyman", "--prior", "web=0.01,app=0.2", "--format", "json",
    )
    assert code == 0
    alloc = json.loads(out)["allocation"]
    assert sum(alloc.values()) == 300 and alloc["app"] > alloc["web"]
    code, out, _ = run(capsys, "stratify", "--sizes", "web=6000,app=4000", "--total", "300")
    assert "| web | 6000 | - | 180 |" in out
    code, _, err = run(capsys, "stratify", "--sizes", "web=6000,app=4000")
    assert code == 2 and "--total" in err
    with pytest.raises(SystemExit):
        main(["stratify", "--sizes", "web", "--total", "5"])
    with pytest.raises(SystemExit):
        main(["stratify", "--sizes", "web=many", "--total", "5"])
    capsys.readouterr()


def test_stratify_estimate_from_file(tmp_path, capsys):
    rows = (
        [{"result": "ok", "stratum": "web"}] * 48
        + [{"result": "bad", "stratum": "web"}] * 2
        + [{"result": "ok", "stratum": "app"}] * 27
        + [{"result": "bad", "stratum": "app"}] * 3
    )
    path = write_audit(tmp_path, rows)
    code, out, _ = run(
        capsys, "stratify", "--sizes", "web=6000,app=4000", "--input", path, "--format", "json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["rate"] == pytest.approx(0.6 * 0.04 + 0.4 * 0.1)
    assert data["exact_interval"][0] <= data["rate"] <= data["exact_interval"][1]
    code, out, _ = run(capsys, "stratify", "--sizes", "web=6000,app=4000", "--input", path)
    assert "Combined error rate" in out and "normal approximation" in out
    code, _, err = run(capsys, "stratify", "--sizes", "web=6000", "--input", path)
    assert code == 2 and "app" in err


def flagged_batch(tmp_path):
    rows = [{"bad": i % 25 == 0} for i in range(1500)]
    return write_audit(tmp_path, rows, "batch.jsonl")


def test_replay_command_for_each_plan_type(tmp_path, capsys):
    path = flagged_batch(tmp_path)
    base = ["replay", "--input", path, "--column", "bad", "--audits", "400", "--format", "json"]
    req = ["--acceptable", "0.01", "--rejectable", "0.08"]
    variants = {
        "single": [*req],
        "single-finite": [*req, "--finite"],
        "double": [*req, "--plan", "double"],
        "sequential": [*req, "--plan", "sequential", "--truncate", "120"],
        "explicit": ["--n", "50", "--c", "2"],
        "explicit-double": ["--double-plan", "30,0,3,60,2"],
    }
    for name, extra in variants.items():
        code, out, _ = run(capsys, *base, *extra)
        assert code == 0, name
        data = json.loads(out)
        assert data["lot_size"] == 1500 and data["bad"] == 60
        assert abs(data["accept_rate"] - data["expected_accept"]) < 5 * data["accept_se"] + 1e-9
    code, out, _ = run(capsys, "replay", "--input", path, "--column", "bad", *req, "--audits", "50")
    assert "accept rate" in out and "single n=" in out
    code, _, err = run(capsys, "replay", "--input", path, "--column", "bad")
    assert code == 2 and "plan" in err
    with pytest.raises(SystemExit):
        main(["replay", "--input", path, "--double-plan", "1,2,3"])
    capsys.readouterr()


def test_missing_file_is_reported(capsys):
    code, _, err = run(capsys, "bound", "--input", "no-such-file.jsonl")
    assert code == 2 and "error" in err


def test_oc_for_a_double_plan(tmp_path, capsys):
    code, out, _ = run(
        capsys, "oc", "--double-plan", "69,0,4,69,3", "--rates", "0.01,0.05", "--format", "json"
    )
    assert code == 0
    data = json.loads(out)
    plan = find_double_plan(0.01, 0.05).plan
    assert data["double"] == plan.to_dict()
    assert data["curve"][0]["accept"] == pytest.approx(plan.accept_prob(0.01))
    assert data["curve"][1]["asn"] == pytest.approx(plan.asn(0.05))
    code, out, _ = run(
        capsys, "oc", "--double-plan", "69,0,4,69,3", "--batch", "1000", "--points", "3"
    )
    assert code == 0 and "double plan n1 = 69" in out and "| bad records |" in out
    code, out, _ = run(capsys, "oc", "--double-plan", "69,0,4,69,3", "--points", "3")
    assert "| error rate | P(accept) | average records inspected |" in out
    code, _, err = run(capsys, "oc", "--n", "50")
    assert code == 2 and "--double-plan" in err
    pytest.importorskip("matplotlib")
    target = tmp_path / "double.png"
    code, _, _ = run(capsys, "oc", "--double-plan", "69,0,4,69,3", "--chart", str(target))
    assert code == 0 and target.read_bytes()[:4] == b"\x89PNG"


def test_sequential_rates_option(capsys):
    code, out, _ = run(
        capsys, "sequential", "--acceptable", "0.01", "--rejectable", "0.06", "--rates",
        "0.02,0.03,0.04", "--format", "json",
    )
    rows = json.loads(out)["operating_characteristic"]
    assert [r["rate"] for r in rows] == [0.02, 0.03, 0.04]
    assert rows[0]["accept"] > rows[1]["accept"] > rows[2]["accept"]


def test_bound_with_inspector_calibration(capsys):
    code, out, _ = run(
        capsys, "bound", "--n", "200", "--bad", "12", "--calibration", "6,40,2,200",
        "--format", "json",
    )
    assert code == 0
    data = json.loads(out)
    est = data["inspector_estimated"]
    assert est["miss_rate"] == pytest.approx(0.15)
    low, high = est["interval"]
    assert low < (12 / 200 - 0.01) / (1 - 0.15 - 0.01) < high
    assert high - low > data["clopper_pearson"][1] - data["clopper_pearson"][0]
    code, out, _ = run(capsys, "bound", "--n", "200", "--bad", "12", "--calibration", "6,40,2,200")
    assert "Inspector calibration" in out
    with pytest.raises(SystemExit):
        main(["bound", "--n", "200", "--bad", "12", "--calibration", "6,40"])
    capsys.readouterr()


def test_plan_double_outside_the_search_range_still_prints_the_single_plan(capsys):
    code, out, _ = run(capsys, "plan", "--acceptable", "0.02", "--rejectable", "0.03", "--double")
    assert code == 0
    assert "Single plan" in out and "Double plan: no double plan found" in out
    code, out, _ = run(
        capsys, "plan", "--acceptable", "0.02", "--rejectable", "0.03", "--double", "--format",
        "json",
    )
    assert json.loads(out)["double"] is None


def test_module_entry_point(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr("sys.argv", ["qasample", "bound", "--n", "50", "--bad", "1"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("qasample", run_name="__main__")
    assert stop.value.code == 0
    assert "1 bad of 50 inspected" in capsys.readouterr().out


def test_sequential_reports_an_unfinished_computation(monkeypatch, capsys):
    import qasample.sequential as module

    monkeypatch.setattr(module, "_WORK_LIMIT", 20_000)
    code, out, _ = run(capsys, "sequential", "--acceptable", "0.01", "--rejectable", "0.0101")
    assert code == 0
    assert "stopped early" in out
