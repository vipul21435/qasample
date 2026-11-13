"""Small helpers for Markdown and JSON output."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def fmt(value: Any, digits: int = 4) -> str:
    """Format a table cell: integers as they are, floats to ``digits`` significant places."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if value == 0.0:
            return "0"
        if abs(value) < 10 ** (-digits) or abs(value) >= 10**7:
            return f"{value:.{max(digits - 1, 1)}e}"
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], digits: int = 4) -> str:
    """Render a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell, digits) for cell in row) + " |")
    return "\n".join(lines)


def to_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=False)
