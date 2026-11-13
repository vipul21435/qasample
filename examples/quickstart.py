"""Plan an audit of a labelled batch, then read the result. Runs offline in under a second.

Usage:  python examples/quickstart.py
"""

from __future__ import annotations

import random

from qasample import (
    Inspector,
    SequentialPlan,
    StratumSample,
    allocate,
    clopper_pearson,
    find_double_plan,
    find_single_plan,
    hypergeom_bound,
    prob_meets_target,
    replay,
    risks_under_inspector,
    stratified_estimate,
)

BATCH = 8000
ACCEPTABLE, REJECTABLE = 0.01, 0.05


def main() -> None:
    # 1. how many records, and how many errors to tolerate
    single = find_single_plan(ACCEPTABLE, REJECTABLE, lot_size=BATCH)
    print(f"single plan: inspect {single.plan.n}, accept up to {single.plan.c} bad")
    print(f"  real producer risk {single.achieved_producer_risk:.4f}, "
          f"consumer risk {single.achieved_consumer_risk:.4f}")
    aoql, at_bad = single.plan.aoql_lot(BATCH)
    print(f"  AOQL under rectification {aoql:.4f} (at {at_bad} bad records)")

    double = find_double_plan(ACCEPTABLE, REJECTABLE, lot_size=BATCH)
    dp = double.plan
    print(f"double plan: n1={dp.n1} c1={dp.c1} r1={dp.r1} n2={dp.n2} c2={dp.c2}, "
          f"average {double.asn_acceptable:.1f} records on an acceptable batch")

    seq = SequentialPlan(ACCEPTABLE, REJECTABLE, max_n=3 * single.plan.n)
    for bad in (80, 400):
        res = seq.exact(lot_size=BATCH, bad=bad)
        print(f"sequential plan, {bad} bad in the batch: P(accept) {res.accept:.4f}, "
              f"average {res.asn:.1f} records")

    # 2. what a finished audit says: 2 bad among 130 inspected
    low, high = clopper_pearson(2, 130)
    print(f"2 bad of 130: error rate between {low:.4f} and {high:.4f} (95 %)")
    lo_bad, hi_bad = hypergeom_bound(2, 130, BATCH)
    print(f"  bad records in the batch: between {lo_bad} and {hi_bad}")
    print(f"  P(error rate <= 5 %) = {prob_meets_target(2, 130, 0.05, BATCH):.4f}")

    # 3. an inspector who misses one bad record in five
    risks = risks_under_inspector(single.plan, ACCEPTABLE, REJECTABLE, Inspector(miss_rate=0.2))
    print(f"with a 20 % miss rate the consumer risk becomes {risks['consumer_risk']:.4f}")

    # 4. two sources of different quality
    sizes = {"reviewed": 5000, "scraped": 3000}
    alloc = allocate(sizes, 200, "neyman", {"reviewed": 0.01, "scraped": 0.08})
    print(f"neyman allocation of 200 inspections: {alloc}")
    est = stratified_estimate(
        [StratumSample("reviewed", 5000, alloc["reviewed"], 1),
         StratumSample("scraped", 3000, alloc["scraped"], 9)]
    )
    print(f"  combined rate {est.rate:.4f}, exact interval "
          f"{est.exact_interval[0]:.4f} to {est.exact_interval[1]:.4f}")

    # 5. replay the single plan on a synthetic batch with 3 % bad records
    flags = [i < 240 for i in range(BATCH)]
    random.Random(0).shuffle(flags)
    res = replay(flags, single.plan, audits=2000, seed=1)
    print(f"replay at 3 % bad: accepted {res.accept_rate:.3f} of audits, "
          f"computed {res.expected_accept:.3f}")


if __name__ == "__main__":
    main()
