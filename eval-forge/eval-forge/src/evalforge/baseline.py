"""Baselines and the regression gate.

The gate is the point of the whole project. An eval you look at is a number; an
eval wired to a threshold in CI is a regression test.

Two design choices that make it trustworthy rather than annoying:

* **It compares against the baseline's confidence interval, not its point
  estimate.** Gating on "today's score must be >= yesterday's score" fails the
  build on noise roughly half the time, and a gate that cries wolf gets
  disabled within a week.
* **It refuses to run against a baseline from a different dataset.** The
  fingerprint is checked, because a comparison across golden sets produces
  plausible-looking numbers that mean nothing.

It also reports per-item regressions — items that passed in the baseline and
fail now. An unchanged aggregate can hide five fixes and five new breaks, and
those five breaks are what someone will hit in production.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .runner import RunResult, compare
from .stats import min_detectable_effect


@dataclass
class Baseline:
    name: str
    dataset: str
    dataset_fingerprint: str
    target: str
    n: int
    pass_rate: float
    pass_rate_ci: list[float]
    mean_score: float
    mean_score_ci: list[float]
    cost_per_item_usd: float
    latency_p95_ms: float
    # item_id -> passed, so per-item regressions can be identified.
    item_outcomes: dict[str, bool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    git_sha: str = ""
    notes: str = ""

    @classmethod
    def from_run(cls, run: RunResult, *, git_sha: str = "", notes: str = "") -> "Baseline":
        summary = run.summary()
        return cls(
            name=run.name,
            dataset=run.dataset,
            dataset_fingerprint=run.dataset_fingerprint,
            target=run.target,
            n=run.n,
            pass_rate=summary["pass_rate"],
            pass_rate_ci=summary["pass_rate_ci"],
            mean_score=summary["mean_score"],
            mean_score_ci=summary["mean_score_ci"],
            cost_per_item_usd=summary["cost_per_item_usd"],
            latency_p95_ms=summary["latency_ms"]["p95"],
            item_outcomes={r.item_id: r.passed for r in run.results},
            git_sha=git_sha,
            notes=notes,
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class GateResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    regressed_items: list[str] = field(default_factory=list)
    fixed_items: list[str] = field(default_factory=list)
    comparison_summary: str = ""
    min_detectable_effect: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = ["", "=" * 70, "  REGRESSION GATE", "=" * 70]
        for check in self.checks:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"  [{mark}] {check['name']}: {check['detail']}")

        if self.comparison_summary:
            lines += ["", f"  paired comparison: {self.comparison_summary}"]

        lines.append(
            f"\n  this dataset can detect a change of ~{self.min_detectable_effect:.1%} "
            "or larger; smaller changes are indistinguishable from noise"
        )

        if self.fixed_items:
            lines.append(f"\n  newly passing ({len(self.fixed_items)}): {', '.join(self.fixed_items[:10])}")
        if self.regressed_items:
            lines.append(
                f"\n  NEWLY FAILING ({len(self.regressed_items)}): "
                f"{', '.join(self.regressed_items[:10])}"
            )
        for warning in self.warnings:
            lines.append(f"\n  warning: {warning}")

        lines += ["", "  " + ("GATE PASSED" if self.passed else "GATE FAILED"), ""]
        return "\n".join(lines)


def check_regression(
    run: RunResult,
    baseline: Baseline,
    *,
    # How far below the baseline's lower confidence bound is tolerated.
    tolerance: float = 0.0,
    max_new_failures: int = 0,
    max_cost_increase: float = 0.5,
    max_latency_increase: float = 0.5,
    confidence: float = 0.95,
) -> GateResult:
    result = GateResult(passed=True)

    if run.dataset_fingerprint != baseline.dataset_fingerprint:
        result.passed = False
        result.checks.append(
            {
                "name": "dataset identity",
                "passed": False,
                "detail": (
                    f"dataset changed since the baseline "
                    f"(now {run.dataset_fingerprint[:8]}, baseline "
                    f"{baseline.dataset_fingerprint[:8]}). Re-record the baseline; "
                    "comparing across datasets is meaningless."
                ),
            }
        )
        return result

    summary = run.summary(confidence)
    current = summary["pass_rate"]
    baseline_low = baseline.pass_rate_ci[0] if baseline.pass_rate_ci else baseline.pass_rate

    # 1. aggregate pass rate, against the baseline's LOWER bound
    floor = baseline_low - tolerance
    ok = current >= floor
    result.checks.append(
        {
            "name": "pass rate",
            "passed": ok,
            "detail": (
                f"{current:.3f} vs baseline {baseline.pass_rate:.3f} "
                f"(lower bound {baseline_low:.3f}, floor {floor:.3f})"
            ),
        }
    )
    result.passed &= ok

    # 2. per-item regressions
    regressed = sorted(
        item_id
        for r in run.results
        if (item_id := r.item_id) in baseline.item_outcomes
        and baseline.item_outcomes[item_id]
        and not r.passed
    )
    fixed = sorted(
        r.item_id
        for r in run.results
        if r.item_id in baseline.item_outcomes
        and not baseline.item_outcomes[r.item_id]
        and r.passed
    )
    result.regressed_items = regressed
    result.fixed_items = fixed

    ok = len(regressed) <= max_new_failures
    result.checks.append(
        {
            "name": "per-item regressions",
            "passed": ok,
            "detail": (
                f"{len(regressed)} newly failing, {len(fixed)} newly passing "
                f"(limit {max_new_failures})"
            ),
        }
    )
    result.passed &= ok

    if regressed and fixed and len(regressed) == len(fixed):
        result.warnings.append(
            f"the aggregate is unchanged but {len(regressed)} items broke while "
            f"{len(fixed)} were fixed — the headline number is hiding real churn"
        )

    # 3. cost
    if baseline.cost_per_item_usd > 0:
        ratio = summary["cost_per_item_usd"] / baseline.cost_per_item_usd - 1
        ok = ratio <= max_cost_increase
        result.checks.append(
            {
                "name": "cost per item",
                "passed": ok,
                "detail": (
                    f"${summary['cost_per_item_usd']:.6f} vs "
                    f"${baseline.cost_per_item_usd:.6f} ({ratio:+.1%}, "
                    f"limit {max_cost_increase:+.0%})"
                ),
            }
        )
        result.passed &= ok

    # 4. latency
    current_p95 = summary["latency_ms"]["p95"]
    if baseline.latency_p95_ms > 0 and current_p95 > 0:
        ratio = current_p95 / baseline.latency_p95_ms - 1
        ok = ratio <= max_latency_increase
        result.checks.append(
            {
                "name": "p95 latency",
                "passed": ok,
                "detail": (
                    f"{current_p95:.0f}ms vs {baseline.latency_p95_ms:.0f}ms "
                    f"({ratio:+.1%}, limit {max_latency_increase:+.0%})"
                ),
            }
        )
        result.passed &= ok

    # 5. errors never pass silently
    if run.errors:
        result.checks.append(
            {
                "name": "target errors",
                "passed": False,
                "detail": f"{len(run.errors)} item(s) failed with a target error",
            }
        )
        result.passed = False

    result.min_detectable_effect = min_detectable_effect(run.n, baseline=max(current, 0.5))
    if result.min_detectable_effect > 0.1:
        result.warnings.append(
            f"with {run.n} items this gate cannot detect a regression smaller than "
            f"{result.min_detectable_effect:.0%}. A passing gate is weak evidence; "
            "grow the dataset to tighten it."
        )

    # Paired comparison for context, when the baseline recorded item outcomes.
    if baseline.item_outcomes:
        shared = [r for r in run.results if r.item_id in baseline.item_outcomes]
        if shared:
            from .stats import paired_bootstrap

            comparison = paired_bootstrap(
                [1.0 if baseline.item_outcomes[r.item_id] else 0.0 for r in shared],
                [1.0 if r.passed else 0.0 for r in shared],
                name_a="baseline",
                name_b="current",
                confidence=confidence,
            )
            result.comparison_summary = comparison.summary()

    return result
