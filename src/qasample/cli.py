"""Command line interface: ``qasample <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .bounds import (
    clopper_pearson,
    hypergeom_bound,
    prob_meets_target,
    rule_of_three,
    wilson,
)
from .double import DoublePlan, find_double_plan
from .inspector import (
    Inspector,
    corrected_bounds_estimated,
    estimate_inspector,
    find_single_plan_with_inspector,
    risks_under_inspector,
)
from .records import load_records
from .replay import replay
from .report import fmt, md_table, to_json
from .sequential import SequentialPlan
from .single import SinglePlan, find_single_plan, oc_curve
from .stratify import METHODS, StratumSample, allocate, stratified_estimate


def _rate(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not a rate between 0 and 1")
    return value


def _rate_list(text: str) -> list[float]:
    return [_rate(part) for part in text.split(",") if part.strip()]


def _pairs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        name, sep, value = part.partition("=")
        if not sep or not name.strip():
            raise argparse.ArgumentTypeError(f"expected name=value, got {part!r}")
        out[name.strip()] = value.strip()
    return out


def _int_pairs(text: str) -> dict[str, int]:
    try:
        return {name: int(value) for name, value in _pairs(text).items()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _rate_pairs(text: str) -> dict[str, float]:
    return {name: _rate(value) for name, value in _pairs(text).items()}


def _double_spec(text: str) -> DoublePlan:
    try:
        n1, c1, r1, n2, c2 = (int(part) for part in text.split(","))
        return DoublePlan(n1, c1, r1, n2, c2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected n1,c1,r1,n2,c2: {exc}") from exc


def _calibration(text: str) -> tuple[int, int, int, int]:
    try:
        missed, known_bad, alarms, known_good = (int(part) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected missed,known_bad,false_alarms,known_good: {exc}"
        ) from exc
    return missed, known_bad, alarms, known_good


def _word_set(text: str) -> set[str]:
    return {part.strip() for part in text.split(",")}


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown", help="output format"
    )


def _add_requirement(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--acceptable", type=_rate, required=required, help="error rate that should be accepted"
    )
    parser.add_argument(
        "--rejectable", type=_rate, required=required, help="error rate that must be caught"
    )
    parser.add_argument(
        "--producer-risk",
        type=float,
        default=0.05,
        help="largest chance of rejecting an acceptable batch (default 0.05)",
    )
    parser.add_argument(
        "--consumer-risk",
        type=float,
        default=0.10,
        help="largest chance of accepting a rejectable batch (default 0.10)",
    )


def _add_inspector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--miss-rate", type=_rate, default=0.0, help="chance the inspector passes a bad record"
    )
    parser.add_argument(
        "--false-alarm-rate",
        type=_rate,
        default=0.0,
        help="chance the inspector flags a good record",
    )


def _add_input(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument("--input", required=required, help="JSONL or CSV file of records")
    parser.add_argument("--column", default="result", help="pass/fail column (default: result)")
    parser.add_argument(
        "--bad-values", type=_word_set, default=None, help="comma-separated values that mean bad"
    )
    parser.add_argument(
        "--good-values", type=_word_set, default=None, help="comma-separated values that mean good"
    )


def _default_rates(max_rate: float, points: int) -> list[float]:
    if points < 2:
        raise ValueError("--points must be at least 2")
    return [max_rate * i / (points - 1) for i in range(points)]


# --- plan -------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> tuple[dict, str]:
    design = find_single_plan(
        args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, args.batch
    )
    payload: dict = {"single": design.to_dict()}
    plan = design.plan
    lines = [
        f"Single plan ({design.model} model): inspect n = {plan.n} records, "
        f"accept if at most c = {plan.c} are bad.",
        "",
        md_table(
            ["batch error rate", "requirement", "P(accept)"],
            [
                [
                    args.acceptable,
                    f">= {fmt(1 - args.producer_risk)}",
                    1 - design.achieved_producer_risk,
                ],
                [
                    args.rejectable,
                    f"<= {fmt(args.consumer_risk)}",
                    design.achieved_consumer_risk,
                ],
            ],
        ),
        "",
    ]
    if design.lot_size is not None:
        lines.append(
            f"Batch of {design.lot_size}: the acceptable batch holds {design.acceptable_bad} "
            f"bad records, the rejectable one {design.rejectable_bad}."
        )
        value, where = plan.aoql_lot(design.lot_size)
        payload["single"]["aoql"] = value
        payload["single"]["aoql_at_bad"] = where
        lines.append(
            f"Under rectification the average outgoing error rate is at most {fmt(value)} "
            f"(reached with {where} bad records in the batch)."
        )
    else:
        value, where_rate = plan.aoql()
        payload["single"]["aoql"] = value
        payload["single"]["aoql_at_rate"] = where_rate
        lines.append(
            f"Under rectification the average outgoing error rate is at most {fmt(value)} "
            f"(reached at an incoming rate of {fmt(where_rate)})."
        )

    if args.double:
        try:
            dd = find_double_plan(
                args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, args.batch
            )
        except ValueError as exc:
            payload["double"] = None
            lines += ["", f"Double plan: {exc}."]
        else:
            dp = dd.plan
            payload["double"] = dd.to_dict()
            lines += [
                "",
                f"Double plan ({dd.model} model): inspect n1 = {dp.n1}; accept if at most "
                f"c1 = {dp.c1} are bad, reject if {dp.r1} or more; otherwise inspect "
                f"n2 = {dp.n2} more and accept if the total is at most c2 = {dp.c2}.",
                "",
                md_table(
                    ["batch error rate", "P(accept)", "average records inspected",
                     "single plan n"],
                    [
                        [args.acceptable, 1 - dd.achieved_producer_risk, dd.asn_acceptable,
                         plan.n],
                        [args.rejectable, dd.achieved_consumer_risk, dd.asn_rejectable, plan.n],
                    ],
                ),
            ]

    if args.miss_rate or args.false_alarm_rate:
        if args.batch is not None:
            raise ValueError("the inspector correction uses the binomial model; drop --batch")
        insp = Inspector(args.miss_rate, args.false_alarm_rate)
        naive = risks_under_inspector(plan, args.acceptable, args.rejectable, insp)
        fixed = find_single_plan_with_inspector(
            args.acceptable, args.rejectable, insp, args.producer_risk, args.consumer_risk
        )
        after = risks_under_inspector(fixed.plan, args.acceptable, args.rejectable, insp)
        payload["inspector"] = {
            "miss_rate": insp.miss_rate,
            "false_alarm_rate": insp.false_alarm_rate,
            "apparent_acceptable": insp.apparent(args.acceptable),
            "apparent_rejectable": insp.apparent(args.rejectable),
            "naive_risks": naive,
            "adjusted": {"n": fixed.plan.n, "c": fixed.plan.c, **after},
        }
        lines += [
            "",
            f"Inspector with miss rate {fmt(insp.miss_rate)} and false alarm rate "
            f"{fmt(insp.false_alarm_rate)}: a batch at {fmt(args.acceptable)} looks like "
            f"{fmt(insp.apparent(args.acceptable))}, one at {fmt(args.rejectable)} looks like "
            f"{fmt(insp.apparent(args.rejectable))}.",
            "",
            md_table(
                ["plan", "n", "c", "real producer risk", "real consumer risk"],
                [
                    ["designed for a perfect inspector", plan.n, plan.c,
                     naive["producer_risk"], naive["consumer_risk"]],
                    ["designed for this inspector", fixed.plan.n, fixed.plan.c,
                     after["producer_risk"], after["consumer_risk"]],
                ],
            ),
        ]
    return payload, "\n".join(lines)


# --- oc ---------------------------------------------------------------------


def _write_chart(path: str, rows: list[dict], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ValueError("the chart needs matplotlib: pip install 'qasample[plot]'") from exc
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([r["rate"] for r in rows], [r["accept"] for r in rows], color="#1f5fa8", lw=1.8)
    ax.set_xlabel("batch error rate")
    ax.set_ylabel("probability of acceptance")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def cmd_oc(args: argparse.Namespace) -> tuple[dict, str]:
    rates = args.rates if args.rates else _default_rates(args.max_rate, args.points)
    if args.double_plan is not None:
        return _oc_double(args, rates)
    if args.n is None or args.c is None:
        raise ValueError("give --n and --c for a single plan, or --double-plan")
    plan = SinglePlan(args.n, args.c)
    rows = oc_curve(plan, rates, args.batch)
    payload = {"n": plan.n, "c": plan.c, "lot_size": args.batch, "curve": rows}
    if args.batch is None:
        table = md_table(
            ["error rate", "P(accept)", "AOQ"],
            [[r["rate"], r["accept"], r["aoq"]] for r in rows],
        )
    else:
        table = md_table(
            ["bad records", "error rate", "P(accept)", "AOQ"],
            [[r["bad"], r["rate"], r["accept"], r["aoq"]] for r in rows],
        )
    head = f"Operating characteristic of the single plan n = {plan.n}, c = {plan.c}"
    head += "." if args.batch is None else f", batch of {args.batch}."
    text = head + "\n\n" + table
    if args.chart:
        fine = oc_curve(plan, _default_rates(max(rates), 201), args.batch)
        _write_chart(args.chart, fine, head)
        payload["chart"] = args.chart
        text += f"\n\nChart written to {args.chart}."
    return payload, text


def _double_rows(plan: DoublePlan, rates: Sequence[float], batch: int | None) -> list[dict]:
    rows = []
    for rate in rates:
        if batch is None:
            rows.append({"rate": rate, "accept": plan.accept_prob(rate), "asn": plan.asn(rate)})
        else:
            bad = min(batch, max(0, round(rate * batch)))
            rows.append(
                {
                    "rate": bad / batch,
                    "bad": bad,
                    "accept": plan.accept_prob_lot(batch, bad),
                    "asn": plan.asn_lot(batch, bad),
                }
            )
    return rows


def _oc_double(args: argparse.Namespace, rates: Sequence[float]) -> tuple[dict, str]:
    plan = args.double_plan
    rows = _double_rows(plan, rates, args.batch)
    payload = {"double": plan.to_dict(), "lot_size": args.batch, "curve": rows}
    if args.batch is None:
        table = md_table(
            ["error rate", "P(accept)", "average records inspected"],
            [[r["rate"], r["accept"], r["asn"]] for r in rows],
        )
    else:
        table = md_table(
            ["bad records", "error rate", "P(accept)", "average records inspected"],
            [[r["bad"], r["rate"], r["accept"], r["asn"]] for r in rows],
        )
    head = (
        f"Operating characteristic of the double plan n1 = {plan.n1}, c1 = {plan.c1}, "
        f"r1 = {plan.r1}, n2 = {plan.n2}, c2 = {plan.c2}"
    )
    head += "." if args.batch is None else f", batch of {args.batch}."
    text = head + "\n\n" + table
    if args.chart:
        fine = _double_rows(plan, _default_rates(max(rates), 201), args.batch)
        _write_chart(args.chart, fine, head)
        payload["chart"] = args.chart
        text += f"\n\nChart written to {args.chart}."
    return payload, text


# --- bound ------------------------------------------------------------------


def cmd_bound(args: argparse.Namespace) -> tuple[dict, str]:
    categories = None
    if args.input:
        records = load_records(
            args.input, args.column, bad_values=args.bad_values, good_values=args.good_values
        )
        n, bad = records.n, records.bad
        categories = dict(records.categories.most_common())
    elif args.n is not None and args.bad is not None:
        n, bad = args.n, args.bad
    else:
        raise ValueError("give --input, or both --n and --bad")
    conf = args.confidence
    two = clopper_pearson(bad, n, conf)
    upper = clopper_pearson(bad, n, conf, "upper")[1]
    score = wilson(bad, n, conf)
    payload: dict = {
        "n": n,
        "bad": bad,
        "rate": bad / n,
        "confidence": conf,
        "clopper_pearson": list(two),
        "upper_bound": upper,
        "wilson": list(score),
    }
    rows = [
        ["Clopper-Pearson, two-sided", two[0], two[1]],
        ["Clopper-Pearson, one-sided upper", 0.0, upper],
        ["Wilson score, two-sided (approximate)", score[0], score[1]],
    ]
    lines = [
        f"{bad} bad of {n} inspected: observed error rate {fmt(bad / n)}.",
        "",
        md_table([f"bound at confidence {fmt(conf)}", "low", "high"], rows, digits=5),
    ]
    if bad == 0:
        r3 = rule_of_three(n, conf)
        payload["zero_defects"] = r3
        lines += [
            "",
            f"No bad record found: the exact upper bound is {fmt(r3['exact'], 5)}; the "
            f"shortcut -ln(1 - confidence) / n gives {fmt(r3['approx'], 5)} (the rule of three "
            "at 95 %).",
        ]
    if args.miss_rate or args.false_alarm_rate:
        insp = Inspector(args.miss_rate, args.false_alarm_rate)
        fixed = insp.corrected_bounds(bad, n, conf)
        payload["inspector_corrected"] = {
            "rate": insp.true_rate(bad / n),
            "interval": list(fixed),
        }
        lines += [
            "",
            f"Corrected for an inspector with miss rate {fmt(insp.miss_rate)} and false alarm "
            f"rate {fmt(insp.false_alarm_rate)}: true error rate about "
            f"{fmt(insp.true_rate(bad / n), 5)}, between {fmt(fixed[0], 5)} and "
            f"{fmt(fixed[1], 5)}.",
        ]
    if args.calibration is not None:
        missed, known_bad, alarms, known_good = args.calibration
        est = estimate_inspector(missed, known_bad, alarms, known_good, conf)
        wide = corrected_bounds_estimated(bad, n, missed, known_bad, alarms, known_good, conf)
        payload["inspector_estimated"] = {
            "miss_rate": est["miss_rate"],
            "miss_interval": list(est["miss_interval"]),
            "false_alarm_rate": est["false_alarm_rate"],
            "false_alarm_interval": list(est["false_alarm_interval"]),
            "interval": list(wide),
        }
        lines += [
            "",
            f"Inspector calibration: missed {missed} of {known_bad} known bad records (miss rate "
            f"{fmt(est['miss_rate'])}), flagged {alarms} of {known_good} known good ones (false "
            f"alarm rate {fmt(est['false_alarm_rate'])}). Allowing for the uncertainty in both, "
            f"the true error rate is between {fmt(wide[0], 5)} and {fmt(wide[1], 5)} "
            "(conservative).",
        ]
    if args.batch is not None:
        low, high = hypergeom_bound(bad, n, args.batch, conf)
        _, high_one = hypergeom_bound(bad, n, args.batch, conf, "upper")
        payload["batch"] = {
            "lot_size": args.batch,
            "bad_total": [low, high],
            "bad_total_upper": high_one,
            "bad_left": [low - bad, high - bad],
            "bad_left_upper": high_one - bad,
        }
        lines += [
            "",
            f"Batch of {args.batch} (hypergeometric): between {low} and {high} bad records in "
            f"all, so between {low - bad} and {high - bad} among the {args.batch - n} not "
            f"inspected; one-sided, at most {high_one - bad} left.",
        ]
    if args.target is not None:
        prob = prob_meets_target(bad, n, args.target, args.batch)
        payload["target"] = {"rate": args.target, "probability": prob}
        lines += [
            "",
            f"Probability that the batch error rate is at most {fmt(args.target)} "
            f"(uniform prior): {fmt(prob)}.",
        ]
    if categories:
        payload["categories"] = categories
        lines += ["", md_table(["error category", "count"], list(categories.items()))]
    return payload, "\n".join(lines)


# --- sequential -------------------------------------------------------------


def cmd_sequential(args: argparse.Namespace) -> tuple[dict, str]:
    plan = SequentialPlan(
        args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, args.truncate
    )
    rates = args.rates or [0.0, args.acceptable, plan.slope, args.rejectable]
    rows = []
    for rate in rates:
        if args.batch is None:
            res = plan.exact(rate)
            shown = rate
        else:
            bad = round(rate * args.batch)
            res = plan.exact(lot_size=args.batch, bad=bad)
            shown = bad / args.batch
        rows.append(
            {
                "rate": shown,
                "accept": res.accept,
                "asn": res.asn,
                "wald_accept": plan.wald_oc(rate),
                "wald_asn": plan.wald_asn(rate),
                "longest_run": res.max_n,
            }
        )
    upto = args.rows if args.rows else min(args.truncate or 10**9, 3 * plan.first_accept_n())
    table = plan.table(upto)
    changes = [
        row
        for i, row in enumerate(table)
        if i == 0
        or i == len(table) - 1
        or (row["accept"], row["reject"]) != (table[i - 1]["accept"], table[i - 1]["reject"])
    ]
    payload = {"plan": plan.to_dict(), "operating_characteristic": rows, "numbers": table}
    lines = [
        f"Sequential plan: after n records with d bad, accept if d <= {fmt(-plan.h_accept)} + "
        f"{fmt(plan.slope, 5)} n, reject if d >= {fmt(plan.h_reject)} + {fmt(plan.slope, 5)} n, "
        "otherwise inspect another record.",
        f"A clean run is accepted after {plan.first_accept_n()} records."
        + (
            f" Truncated at {args.truncate} records (then decided by the sign of the log "
            "likelihood ratio)."
            if args.truncate
            else ""
        ),
        "",
        "Decision numbers, listed where they change:",
        "",
        md_table(
            ["n", "accept if d <=", "reject if d >="],
            [[r["n"], r["accept"], r["reject"]] for r in changes],
        ),
        "",
        "Exact operating characteristic"
        + (f" for a batch of {args.batch}" if args.batch is not None else "")
        + ", with Wald's approximation beside it:",
        "",
        md_table(
            ["error rate", "P(accept)", "average records", "approx. P(accept)",
             "approx. average"],
            [[r["rate"], r["accept"], r["asn"], r["wald_accept"], r["wald_asn"]] for r in rows],
        ),
    ]
    return payload, "\n".join(lines)


# --- stratify ---------------------------------------------------------------


def cmd_stratify(args: argparse.Namespace) -> tuple[dict, str]:
    if args.input:
        records = load_records(
            args.input,
            args.column,
            stratum=args.stratum,
            bad_values=args.bad_values,
            good_values=args.good_values,
        )
        counts = records.by_stratum()
        missing = [name for name in counts if name not in args.sizes]
        if missing:
            raise ValueError(f"no size given for strata: {', '.join(missing)}")
        strata = [
            StratumSample(name, args.sizes[name], n, bad) for name, (n, bad) in counts.items()
        ]
        est = stratified_estimate(strata, args.confidence)
        payload = est.to_dict()
        lines = [
            md_table(
                ["stratum", "size", "inspected", "bad", "rate", "bad records, low", "high"],
                [[r["name"], r["size"], r["n"], r["bad"], r["rate"], r["bad_low"], r["bad_high"]]
                 for r in est.strata],
            ),
            "",
            f"Combined error rate {fmt(est.rate, 5)} (standard error {fmt(est.std_error, 5)}).",
            "",
            md_table(
                [f"interval at confidence {fmt(est.confidence)}", "low", "high"],
                [
                    ["normal approximation", *est.normal_interval],
                    ["exact per stratum, combined (conservative)", *est.exact_interval],
                ],
                digits=5,
            ),
        ]
        return payload, "\n".join(lines)
    if args.total is None:
        raise ValueError("give --total to allocate a sample, or --input to combine an audit")
    alloc = allocate(args.sizes, args.total, args.method, args.prior, args.minimum)
    payload = {"method": args.method, "total": args.total, "allocation": alloc}
    rows = [
        [name, args.sizes[name], (args.prior or {}).get(name), alloc[name]] for name in args.sizes
    ]
    text = f"Allocation of {args.total} inspections ({args.method}):\n\n" + md_table(
        ["stratum", "size", "prior rate", "inspect"], rows
    )
    return payload, text


# --- replay -----------------------------------------------------------------


def cmd_replay(args: argparse.Namespace) -> tuple[dict, str]:
    records = load_records(
        args.input, args.column, bad_values=args.bad_values, good_values=args.good_values
    )
    lot = records.n if args.finite else None
    have_requirement = args.acceptable is not None and args.rejectable is not None
    plan: SinglePlan | DoublePlan | SequentialPlan
    if args.n is not None and args.c is not None:
        plan = SinglePlan(args.n, args.c)
    elif args.double_plan is not None:
        plan = args.double_plan
    elif not have_requirement:
        raise ValueError(
            "give a plan (--n and --c, or --double-plan) or --acceptable and --rejectable"
        )
    elif args.plan == "single":
        plan = find_single_plan(
            args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, lot
        ).plan
    elif args.plan == "double":
        plan = find_double_plan(
            args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, lot
        ).plan
    else:
        plan = SequentialPlan(
            args.acceptable, args.rejectable, args.producer_risk, args.consumer_risk, args.truncate
        )
    if isinstance(plan, SinglePlan):
        label = f"single n={plan.n} c={plan.c}"
    elif isinstance(plan, DoublePlan):
        label = f"double n1={plan.n1} c1={plan.c1} r1={plan.r1} n2={plan.n2} c2={plan.c2}"
    else:
        label = "sequential" + (f" truncated at {plan.max_n}" if plan.max_n else "")
    result = replay(records.flags, plan, args.audits, args.seed)
    payload = {"plan": label, **result.to_dict()}
    lines = [
        f"Batch of {result.lot_size} records, {result.bad} bad (error rate "
        f"{fmt(result.true_rate, 5)}). Plan: {label}. {result.audits} audits, seed {args.seed}.",
        "",
        md_table(
            ["quantity", "replay", "computed"],
            [
                ["accept rate", result.accept_rate, result.expected_accept],
                ["records inspected per audit", result.mean_inspected, result.expected_inspected],
            ],
        ),
        "",
        f"Computed values are exact for this batch (sampling without replacement); the "
        f"binomial model gives P(accept) = {fmt(result.binomial_accept)}. One standard error "
        f"of the replay accept rate is {fmt(result.accept_se)}.",
    ]
    return payload, "\n".join(lines)


# --- parser -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qasample",
        description="Acceptance sampling for data quality: plan an inspection and state its risks.",
    )
    parser.add_argument("--version", action="version", version=f"qasample {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="smallest sampling plan meeting two risks")
    _add_requirement(plan)
    plan.add_argument(
        "--batch", type=int, default=None, help="batch size; omit for the binomial model"
    )
    plan.add_argument("--double", action="store_true", help="also search for a double plan")
    _add_inspector(plan)
    _add_format(plan)
    plan.set_defaults(func=cmd_plan)

    oc = sub.add_parser("oc", help="operating characteristic of a single or double plan")
    oc.add_argument("--n", type=int, default=None, help="sample size")
    oc.add_argument("--c", type=int, default=None, help="acceptance number")
    oc.add_argument(
        "--double-plan", type=_double_spec, default=None, help="double plan n1,c1,r1,n2,c2"
    )
    oc.add_argument("--batch", type=int, default=None, help="batch size (hypergeometric model)")
    oc.add_argument("--rates", type=_rate_list, default=None, help="comma-separated error rates")
    oc.add_argument("--max-rate", type=_rate, default=0.10, help="largest rate in the table")
    oc.add_argument("--points", type=int, default=11, help="number of rows in the table")
    oc.add_argument("--chart", default=None, help="write the curve to this image file")
    _add_format(oc)
    oc.set_defaults(func=cmd_oc)

    bound = sub.add_parser("bound", help="bounds on the batch error rate from a sample")
    _add_input(bound, required=False)
    bound.add_argument("--n", type=int, default=None, help="records inspected")
    bound.add_argument("--bad", type=int, default=None, help="records found bad")
    bound.add_argument("--batch", type=int, default=None, help="batch size (finite-batch bound)")
    bound.add_argument("--confidence", type=float, default=0.95, help="confidence level")
    bound.add_argument("--target", type=_rate, default=None, help="error rate the batch must meet")
    _add_inspector(bound)
    bound.add_argument(
        "--calibration",
        type=_calibration,
        default=None,
        help="inspector checked on known records: missed,known_bad,false_alarms,known_good",
    )
    _add_format(bound)
    bound.set_defaults(func=cmd_bound)

    seq = sub.add_parser("sequential", help="Wald's sequential plan with its exact behaviour")
    _add_requirement(seq)
    seq.add_argument("--truncate", type=int, default=None, help="stop after this many records")
    seq.add_argument("--batch", type=int, default=None, help="batch size (finite batch)")
    seq.add_argument("--rows", type=int, default=None, help="list decision numbers up to this n")
    seq.add_argument("--rates", type=_rate_list, default=None, help="error rates to tabulate")
    _add_format(seq)
    seq.set_defaults(func=cmd_sequential)

    strat = sub.add_parser("stratify", help="allocate a sample across strata or combine an audit")
    strat.add_argument(
        "--sizes", type=_int_pairs, required=True, help="stratum sizes, e.g. web=6000,app=4000"
    )
    strat.add_argument("--total", type=int, default=None, help="inspections to allocate")
    strat.add_argument("--method", choices=METHODS, default="proportional")
    strat.add_argument("--prior", type=_rate_pairs, default=None, help="prior rates for neyman")
    strat.add_argument("--minimum", type=int, default=1, help="least inspections per stratum")
    _add_input(strat, required=False)
    strat.add_argument("--stratum", default="stratum", help="stratum column (default: stratum)")
    strat.add_argument("--confidence", type=float, default=0.95, help="confidence level")
    _add_format(strat)
    strat.set_defaults(func=cmd_stratify)

    rep = sub.add_parser("replay", help="replay a plan on records with a known good/bad flag")
    _add_input(rep, required=True)
    _add_requirement(rep, required=False)
    rep.add_argument("--plan", choices=("single", "double", "sequential"), default="single")
    rep.add_argument("--n", type=int, default=None, help="explicit single plan: sample size")
    rep.add_argument("--c", type=int, default=None, help="explicit single plan: acceptance number")
    rep.add_argument(
        "--double-plan", type=_double_spec, default=None, help="explicit double plan n1,c1,r1,n2,c2"
    )
    rep.add_argument("--truncate", type=int, default=None, help="truncate a sequential plan")
    rep.add_argument("--finite", action="store_true", help="design the plan for this batch size")
    rep.add_argument("--audits", type=int, default=10_000, help="number of audits to draw")
    rep.add_argument("--seed", type=int, default=0, help="random seed")
    _add_format(rep)
    rep.set_defaults(func=cmd_replay)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload, text = args.func(args)
    except (ValueError, OSError) as exc:
        print(f"qasample: error: {exc}", file=sys.stderr)
        return 2
    print(to_json(payload) if args.format == "json" else text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
