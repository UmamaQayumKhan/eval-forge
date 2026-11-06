"""Run an eval: score every item, aggregate with confidence intervals,
compare systems, and measure run-to-run variance.

The orchestration is deliberately boring. The interesting decisions are in
`stats.py` and in what this module refuses to report: there is no bare
accuracy number anywhere in the output. Every metric carries an interval,
because a point estimate from 40 items invites exactly the false conclusion
this harness exists to prevent.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .dataset import Dataset, Item
from .providers import Call, Target
from .scorers import Score, Scorer
from .stats import (
    Comparison,
    Interval,
    bootstrap_interval,
    mean,
    min_detectable_effect,
    paired_bootstrap,
    percentile,
    run_variance,
    wilson_interval,
)


@dataclass
class ItemResult:
    item_id: str
    category: str
    input: str
    output: str
    score: float
    passed: bool
    detail: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False
    error: str = ""
    components: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    name: str
    dataset: str
    dataset_fingerprint: str
    target: str
    results: list[ItemResult] = field(default_factory=list)
    wall_seconds: float = 0.0
    started_at: float = field(default_factory=time.time)

    # -- aggregates ------------------------------------------------------
    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.results]

    @property
    def passes(self) -> list[float]:
        return [1.0 if r.passed else 0.0 for r in self.results]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def errors(self) -> list[ItemResult]:
        return [r for r in self.results if r.error]

    def pass_rate(self, confidence: float = 0.95) -> Interval:
        """Wilson rather than bootstrap: the pass rate is a proportion, and
        Wilson has better coverage near 0 and 1 where eval metrics live."""
        return wilson_interval(int(sum(self.passes)), self.n, confidence)

    def mean_score(self, confidence: float = 0.95) -> Interval:
        return bootstrap_interval(self.scores, confidence=confidence)

    def total_cost(self) -> float:
        return round(sum(r.cost_usd for r in self.results), 6)

    def latency(self) -> dict[str, float]:
        live = [r.latency_ms for r in self.results if not r.cached and r.latency_ms > 0]
        if not live:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "n": len(live),
            "mean": round(mean(live), 1),
            "p50": round(percentile(live, 50), 1),
            "p95": round(percentile(live, 95), 1),
            "p99": round(percentile(live, 99), 1),
        }

    def by_category(self, confidence: float = 0.95) -> dict[str, dict[str, Any]]:
        cats: dict[str, list[ItemResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        out: dict[str, dict[str, Any]] = {}
        for cat, rows in sorted(cats.items()):
            passes = [1.0 if r.passed else 0.0 for r in rows]
            interval = wilson_interval(int(sum(passes)), len(rows), confidence)
            out[cat] = {
                "n": len(rows),
                "pass_rate": round(interval.point, 4),
                "ci_low": round(interval.low, 4),
                "ci_high": round(interval.high, 4),
                "mean_score": round(mean([r.score for r in rows]), 4),
            }
        return out

    def failures(self, limit: int = 20) -> list[dict[str, Any]]:
        return [
            {
                "id": r.item_id,
                "category": r.category,
                "input": r.input[:90],
                "detail": r.detail[:160],
                "output": r.output[:140],
            }
            for r in self.results
            if not r.passed
        ][:limit]

    def summary(self, confidence: float = 0.95) -> dict[str, Any]:
        pr = self.pass_rate(confidence)
        ms = self.mean_score(confidence)
        mde = min_detectable_effect(self.n, baseline=max(pr.point, 0.5))
        return {
            "name": self.name,
            "dataset": self.dataset,
            "dataset_fingerprint": self.dataset_fingerprint,
            "target": self.target,
            "n": self.n,
            "pass_rate": round(pr.point, 4),
            "pass_rate_ci": [round(pr.low, 4), round(pr.high, 4)],
            "pass_rate_margin": round(pr.margin, 4),
            "mean_score": round(ms.point, 4),
            "mean_score_ci": [round(ms.low, 4), round(ms.high, 4)],
            # The honesty line: how small a change this set can even see.
            "min_detectable_effect": round(mde, 4),
            "errors": len(self.errors),
            "total_cost_usd": self.total_cost(),
            "cost_per_item_usd": round(self.total_cost() / self.n, 6) if self.n else 0.0,
            "latency_ms": self.latency(),
            "cached_items": sum(1 for r in self.results if r.cached),
            "wall_seconds": round(self.wall_seconds, 2),
            "by_category": self.by_category(confidence),
        }


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


@dataclass
class Runner:
    target: Target
    scorer: Scorer
    name: str = "run"
    # Free tiers rate-limit hard, so this stays low by default.
    concurrency: int = 4
    on_item: Callable[[ItemResult], None] | None = None

    def _run_one(self, item: Item) -> ItemResult:
        call: Call = self.target.run(item.as_dict())
        if call.error:
            # A target failure is recorded as a failed item with the error
            # attached, not raised. One flaky item must not void the run, and
            # the error count surfaces in the summary.
            return ItemResult(
                item_id=item.id,
                category=item.category,
                input=item.input,
                output="",
                score=0.0,
                passed=False,
                detail=f"target error: {call.error}",
                latency_ms=call.latency_ms,
                error=call.error,
            )

        score: Score = self.scorer.score(call.text, item.expected, item.as_dict())
        return ItemResult(
            item_id=item.id,
            category=item.category,
            input=item.input,
            output=call.text,
            score=score.value,
            passed=score.passed,
            detail=score.detail,
            latency_ms=call.latency_ms,
            cost_usd=call.cost_usd + score.cost_usd,
            cached=call.cached,
            components=score.metadata.get("components", {}),
        )

    def run(self, dataset: Dataset) -> RunResult:
        started = time.perf_counter()
        results: list[ItemResult] = []

        if self.concurrency <= 1:
            for item in dataset:
                result = self._run_one(item)
                results.append(result)
                if self.on_item:
                    self.on_item(result)
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {pool.submit(self._run_one, item): item for item in dataset}
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if self.on_item:
                        self.on_item(result)

        # Stable order regardless of completion order, so two runs of the same
        # dataset produce row-aligned output and can be paired.
        order = {item.id: i for i, item in enumerate(dataset)}
        results.sort(key=lambda r: order.get(r.item_id, 0))

        return RunResult(
            name=self.name,
            dataset=dataset.name,
            dataset_fingerprint=dataset.fingerprint,
            target=getattr(self.target, "name", "target"),
            results=results,
            wall_seconds=time.perf_counter() - started,
        )

    def run_repeated(self, dataset: Dataset, runs: int = 3) -> tuple[list[RunResult], Any]:
        """Run the same eval several times to measure run-to-run variance.

        This should be the first thing anyone does with a new eval. If identical
        runs differ by 4 points, no 3-point "improvement" from a prompt change
        means anything, and every comparison after that is theatre.
        """
        all_runs = [self.run(dataset) for _ in range(max(1, runs))]
        variance = run_variance([r.pass_rate().point for r in all_runs])
        return all_runs, variance


def compare(a: RunResult, b: RunResult, *, confidence: float = 0.95) -> Comparison:
    """Paired comparison of two runs on the same dataset.

    Refuses to compare across datasets. Comparing a run on one golden set
    against a baseline taken on another is the most damaging mistake in eval
    tooling precisely because the numbers still look plausible.
    """
    if a.dataset_fingerprint != b.dataset_fingerprint:
        raise ValueError(
            "cannot compare runs on different datasets "
            f"({a.dataset} fp={a.dataset_fingerprint[:8]} vs "
            f"{b.dataset} fp={b.dataset_fingerprint[:8]})"
        )

    by_id_a = {r.item_id: r for r in a.results}
    by_id_b = {r.item_id: r for r in b.results}
    shared = [i for i in by_id_a if i in by_id_b]
    if not shared:
        raise ValueError("runs share no items")

    return paired_bootstrap(
        [1.0 if by_id_a[i].passed else 0.0 for i in shared],
        [1.0 if by_id_b[i].passed else 0.0 for i in shared],
        name_a=a.name,
        name_b=b.name,
        confidence=confidence,
    )


def compare_many(runs: Sequence[RunResult], *, confidence: float = 0.95) -> list[Comparison]:
    """Every pairwise comparison, for a provider or prompt bake-off.

    Note the multiple-comparisons caveat: with k systems there are k(k-1)/2
    comparisons, and at 95% confidence roughly 1 in 20 will look significant by
    chance. The report states this rather than quietly letting the reader draw a
    conclusion from the one pair that happened to clear the bar.
    """
    out: list[Comparison] = []
    for i, a in enumerate(runs):
        for b in runs[i + 1 :]:
            try:
                out.append(compare(a, b, confidence=confidence))
            except ValueError:
                continue
    return out
