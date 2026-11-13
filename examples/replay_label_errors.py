"""Replay three requirements on the flagged test sets written by fetch_label_errors.py.

Usage:  python examples/replay_label_errors.py DATA_DIR [AUDITS] [SEED]

Prints, in Markdown: the plans each requirement needs, then for every test set the accept
rate over AUDITS random audits next to the probability computed for that batch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qasample.double import find_double_plan
from qasample.records import load_records
from qasample.replay import replay
from qasample.report import md_table
from qasample.sequential import SequentialPlan
from qasample.single import find_single_plan

DATASETS = ("mnist", "cifar10", "20news", "imdb", "imagenet", "cifar100")
REQUIREMENTS = {"A": (0.005, 0.02), "B": (0.01, 0.05), "C": (0.02, 0.08)}
ALPHA, BETA = 0.05, 0.10


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    data_dir = Path(sys.argv[1])
    audits = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    singles, sequentials, rows = {}, {}, []
    for key, (good, poor) in REQUIREMENTS.items():
        single = find_single_plan(good, poor, ALPHA, BETA)
        double = find_double_plan(good, poor, ALPHA, BETA)
        seq = SequentialPlan(good, poor, ALPHA, BETA)
        singles[key], sequentials[key] = single.plan, seq
        dp = double.plan
        rows.append(
            [
                key,
                f"{good:.3f}",
                f"{poor:.3f}",
                f"n={single.plan.n}, c={single.plan.c}",
                f"n1={dp.n1}, c1={dp.c1}, r1={dp.r1}, n2={dp.n2}, c2={dp.c2}",
                double.asn_acceptable,
                seq.exact(good).asn,
                seq.exact(poor).asn,
            ]
        )
    print(
        md_table(
            ["requirement", "acceptable", "rejectable", "single plan", "double plan",
             "double: average records at acceptable", "sequential: average at acceptable",
             "sequential: average at rejectable"],
            rows,
            digits=1,
        )
    )
    print()

    flags = {name: load_records(data_dir / f"{name}.jsonl", "bad").flags for name in DATASETS}
    rows = []
    for name in DATASETS:
        for key, plan in singles.items():
            res = replay(flags[name], plan, audits, seed)
            rows.append(
                [name, res.lot_size, res.bad, f"{res.true_rate:.4f}", key, res.accept_rate,
                 res.expected_accept, res.binomial_accept]
            )
    print(
        md_table(
            ["test set", "records", "bad", "error rate", "requirement", "replay accept rate",
             "computed, finite batch", "computed, binomial"],
            rows,
        )
    )
    print()

    rows = []
    seq = sequentials["B"]
    for name in DATASETS:
        res = replay(flags[name], seq, audits, seed)
        rows.append(
            [name, f"{res.true_rate:.4f}", res.accept_rate, res.expected_accept,
             res.mean_inspected, res.expected_inspected]
        )
    print(
        md_table(
            ["test set", "error rate", "replay accept rate", "computed",
             "replay records per audit", "computed records"],
            rows,
            digits=4,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
