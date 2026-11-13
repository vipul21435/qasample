"""Acceptance sampling for data quality.

Plan how many records of a batch to inspect and what error count to tolerate,
with the risks stated, and bound the batch's error rate from the sample.
"""

from .bounds import (
    clopper_pearson,
    hypergeom_bound,
    prob_meets_target,
    rule_of_three,
    wilson,
    zero_defect_sample_size,
)
from .double import DoublePlan, DoublePlanDesign, find_double_plan
from .inspector import (
    Inspector,
    corrected_bounds_estimated,
    estimate_inspector,
    find_single_plan_with_inspector,
    risks_under_inspector,
)
from .records import Records, load_records
from .replay import ReplayResult, replay
from .sequential import SequentialPlan, SequentialResult
from .single import SinglePlan, SinglePlanDesign, find_single_plan, oc_curve
from .stratify import StratifiedEstimate, StratumSample, allocate, stratified_estimate

__version__ = "0.1.0"

__all__ = [
    "DoublePlan",
    "DoublePlanDesign",
    "Inspector",
    "Records",
    "ReplayResult",
    "SequentialPlan",
    "SequentialResult",
    "SinglePlan",
    "SinglePlanDesign",
    "StratifiedEstimate",
    "StratumSample",
    "allocate",
    "clopper_pearson",
    "corrected_bounds_estimated",
    "estimate_inspector",
    "find_double_plan",
    "find_single_plan",
    "find_single_plan_with_inspector",
    "hypergeom_bound",
    "load_records",
    "oc_curve",
    "prob_meets_target",
    "replay",
    "risks_under_inspector",
    "rule_of_three",
    "stratified_estimate",
    "wilson",
    "zero_defect_sample_size",
]
