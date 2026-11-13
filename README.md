# qasample

Acceptance sampling for data quality: choose how many records of a batch to inspect and what error
count to tolerate, with the risks stated.

This is an early version. The list below is what works today; the rest of the plan (double and
sequential plans, bounds from an inspected sample, stratified audits, an imperfect inspector, replay
on labelled data) is not implemented yet.

## What works so far

- Exact binomial and hypergeometric tail probabilities in pure Python (no runtime dependencies).
- The smallest single sampling plan (n, c) that meets a producer risk and a consumer risk, under the
  binomial model or, with a batch size, under the hypergeometric model.
- The operating characteristic of a plan, its average outgoing quality under rectification and the
  limit of that quantity (AOQL).
- A command line with `plan` and `oc`, printing Markdown or JSON.

## Install

```
pip install git+https://github.com/vipul21435/qasample
```

## Example

```
$ qasample plan --acceptable 0.01 --rejectable 0.05 --batch 2000
Single plan (hypergeometric model): inspect n = 130 records, accept if at most c = 3 are bad.

| batch error rate | requirement | P(accept) |
| --- | --- | --- |
| 0.0100 | >= 0.9500 | 0.9634 |
| 0.0500 | <= 0.1000 | 0.0980 |

Batch of 2000: the acceptable batch holds 20 bad records, the rejectable one 100.
Under rectification the average outgoing error rate is at most 0.0143 (reached with 45 bad records in the batch).
```

## Tests

```
uv run --extra dev pytest -q
```

The plan search is checked against exhaustive enumeration and the tails against scipy, which is a
test-only dependency.

## Licence

MIT. Copyright (c) 2026 Vipul Raj Jha.

Built by Vipul Raj Jha · https://vipul21435.github.io
