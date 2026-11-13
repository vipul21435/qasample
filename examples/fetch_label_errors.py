"""Download the label-error verification files of Northcutt et al. (2021) and write one
JSONL file per test set with a good/bad flag for every record.

Source: https://github.com/cleanlab/label-errors (Apache-2.0). Nothing from it is stored in
this repository; this script fetches the files into a directory of your choice.

A record is marked bad when it was sent for verification and fewer than three of its five
votes kept its given label. Records that were never sent for verification are marked good,
so the error rate of each file is a lower bound on the real one.

Usage:  python examples/fetch_label_errors.py OUT_DIR
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/cleanlab/label-errors/main"

# name -> (verification file, label array whose length is the size of the test set)
DATASETS = {
    "mnist": ("mturk/mnist_mturk.json", "original_test_labels/mnist_test_set_original_labels.npy"),
    "cifar10": (
        "mturk/cifar10_mturk.json",
        "original_test_labels/cifar10_test_set_original_labels.npy",
    ),
    "cifar100": (
        "mturk/cifar100_mturk.json",
        "original_test_labels/cifar100_test_set_original_labels.npy",
    ),
    "imagenet": (
        "mturk/imagenet_mturk.json",
        "original_test_labels/imagenet_val_set_original_labels.npy",
    ),
    "imdb": ("mturk/imdb_mturk.json", "original_test_labels/imdb_test_set_original_labels.npy"),
    "20news": (
        "mturk/20news_mturk.json",
        "original_test_labels/20news_test_set_original_labels.npy",
    ),
}


def fetch(relative: str, out_dir: Path) -> Path:
    target = out_dir / "raw" / relative
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{relative}", timeout=60) as response:
            target.write_bytes(response.read())
    return target


def npy_length(path: Path) -> int:
    """Length of a one-dimensional .npy array, read from its text header only."""
    head = path.read_bytes()[:512].decode("latin-1")
    match = re.search(r"'shape':\s*\((\d+),\s*\)", head)
    if not match:
        raise ValueError(f"{path}: cannot find a one-dimensional shape in the header")
    return int(match.group(1))


def build(name: str, out_dir: Path) -> dict:
    votes_file, labels_file = DATASETS[name]
    entries = json.loads(fetch(votes_file, out_dir).read_text(encoding="utf-8"))
    size = npy_length(fetch(labels_file, out_dir))
    checked = {}
    for entry in entries:
        checked[entry["id"]] = int(entry["mturk"]["given"]) < 3
    rows = []
    if all(isinstance(key, int) for key in checked):
        for index in range(size):
            rows.append(
                {"id": index, "verified": index in checked, "bad": checked.get(index, False)}
            )
    else:  # ids are file names: the unverified records are written without one
        for key, is_bad in checked.items():
            rows.append({"id": key, "verified": True, "bad": is_bad})
        for index in range(size - len(checked)):
            rows.append({"id": f"unverified-{index}", "verified": False, "bad": False})
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    bad = sum(1 for row in rows if row["bad"])
    return {"dataset": name, "records": size, "verified": len(checked), "bad": bad}


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    print("| test set | records | sent for verification | bad | error rate |")
    print("| --- | --- | --- | --- | --- |")
    for name in DATASETS:
        info = build(name, out_dir)
        rate = info["bad"] / info["records"]
        print(
            f"| {name} | {info['records']} | {info['verified']} | {info['bad']} | {rate:.4f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
