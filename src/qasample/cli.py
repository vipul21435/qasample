"""Command line interface: ``qasample <command> ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from .report import fmt, md_table, to_json
from .single import SinglePlan, find_single_plan, oc_curve


def _rate(text: str) -> float:
    value = float(text)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"{text} is not a rate between 0 and 1")
    return value


def _rate_list(text: str) -> list[float]:
    return [_rate(part) for part in text.split(",") if part.strip()]


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("markdown", "json"), default="markdown", help="output format"
    )


def _add_requirement(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--acceptable", type=_rate, required=True, help="error rate that should be accepted"
    )
    parser.add_argument(
        "--rejectable", type=_rate, required=True, help="error rate that must be caught"
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
    parser.add_argument(
        "--batch", type=int, default=None, help="batch size; omit for the binomial model"
    )


def _default_rates(max_rate: float, points: int) -> list[float]:
    return [max_rate * i / (points - 1) for i in range(points)]


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
    ]
    if design.lot_size is not None:
        lines.append("")
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
        lines.append("")
        lines.append(
            f"Under rectification the average outgoing error rate is at most {fmt(value)} "
            f"(reached at an incoming rate of {fmt(where_rate)})."
        )
    return payload, "\n".join(lines)


def cmd_oc(args: argparse.Namespace) -> tuple[dict, str]:
    plan = SinglePlan(args.n, args.c)
    rates = args.rates if args.rates else _default_rates(args.max_rate, args.points)
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
    return payload, head + "\n\n" + table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qasample",
        description="Acceptance sampling for data quality: plan an inspection and state its risks.",
    )
    parser.add_argument("--version", action="version", version=f"qasample {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="smallest single sampling plan meeting two risks")
    _add_requirement(plan)
    _add_format(plan)
    plan.set_defaults(func=cmd_plan)

    oc = sub.add_parser("oc", help="operating characteristic table of a single plan")
    oc.add_argument("--n", type=int, required=True, help="sample size")
    oc.add_argument("--c", type=int, required=True, help="acceptance number")
    oc.add_argument("--batch", type=int, default=None, help="batch size (hypergeometric model)")
    oc.add_argument("--rates", type=_rate_list, default=None, help="comma-separated error rates")
    oc.add_argument("--max-rate", type=_rate, default=0.10, help="largest rate in the table")
    oc.add_argument("--points", type=int, default=11, help="number of rows in the table")
    _add_format(oc)
    oc.set_defaults(func=cmd_oc)
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
