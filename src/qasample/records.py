"""Read inspected records from JSONL or CSV."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOOD_VALUES = frozenset(
    {"pass", "passed", "ok", "good", "correct", "0", "false", "no", "none", "null", ""}
)


def _norm(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value).strip().lower()


@dataclass
class Records:
    """Flags (True = bad) with the raw category of each bad record and an optional stratum."""

    flags: list[bool] = field(default_factory=list)
    categories: Counter = field(default_factory=Counter)
    strata: list[str] | None = None

    @property
    def n(self) -> int:
        return len(self.flags)

    @property
    def bad(self) -> int:
        return sum(self.flags)

    def by_stratum(self) -> dict[str, tuple[int, int]]:
        """(inspected, bad) per stratum, in order of first appearance."""
        if self.strata is None:
            raise ValueError("no stratum column was read")
        out: dict[str, list[int]] = {}
        for flag, name in zip(self.flags, self.strata, strict=True):
            cell = out.setdefault(name, [0, 0])
            cell[0] += 1
            cell[1] += 1 if flag else 0
        return {name: (cell[0], cell[1]) for name, cell in out.items()}


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a ``.jsonl``/``.ndjson`` file (one object per line) or of a CSV with a header."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() in (".jsonl", ".ndjson", ".json"):
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: line {number} is not valid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: line {number} is not a JSON object")
                rows.append(row)
        return rows
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(dict(row) for row in reader)
    return rows


def load_records(
    path: str | Path,
    column: str,
    stratum: str | None = None,
    bad_values: set[str] | None = None,
    good_values: set[str] | None = None,
) -> Records:
    """Load inspected records and decide which are bad.

    By default a value counts as good when, lower-cased, it is one of
    ``pass, passed, ok, good, correct, 0, false, no, none, null`` or empty;
    every other value marks a bad record and is kept as its error category.
    ``bad_values`` reverses the logic (exactly these values are bad);
    ``good_values`` replaces the default list.
    """
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path}: no records")
    bad_set = {v.strip().lower() for v in bad_values} if bad_values is not None else None
    good_set = (
        {v.strip().lower() for v in good_values} if good_values is not None else GOOD_VALUES
    )
    records = Records(strata=[] if stratum is not None else None)
    for number, row in enumerate(rows, start=1):
        if column not in row:
            raise ValueError(f"{path}: record {number} has no column {column!r}")
        value = _norm(row[column])
        is_bad = value in bad_set if bad_set is not None else value not in good_set
        records.flags.append(is_bad)
        if is_bad:
            records.categories[value] += 1
        if stratum is not None:
            if stratum not in row:
                raise ValueError(f"{path}: record {number} has no column {stratum!r}")
            records.strata.append(str(row[stratum]))
    return records
