# Changelog

## 0.1.0 - 2026-11-13

First release.

- Single sampling plans: the smallest (n, c) meeting a producer and a consumer risk under the
  binomial or the hypergeometric model; operating characteristic, AOQ, AOQL and ATI.
- Double sampling plans with exact operating characteristic and average sample number, and a
  search for the plan with the smallest average sample on an acceptable batch.
- Wald's sequential plan: decision lines, exact operating characteristic and average sample
  number (binomial or finite batch), truncation, and Wald's closed-form approximations.
- Bounds from an inspected sample: Clopper-Pearson, Wilson, the zero-defect case, exact bounds on
  the bad records of a finite batch, and the posterior probability of meeting a target.
- Stratified audits: proportional and Neyman allocation, combined estimate with two intervals.
- Imperfect inspector: apparent versus true error rate, corrected plans and bounds.
- Replay of a plan on a batch with known good/bad flags.
- Command line: `plan`, `oc`, `bound`, `sequential`, `stratify`, `replay`; Markdown and JSON.
