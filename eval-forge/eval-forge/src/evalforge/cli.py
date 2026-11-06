"""CLI.

    evalforge validate datasets/support-qa.jsonl
    evalforge power --n 40
    evalforge run --config configs/support-qa.json
    evalforge variance --config configs/support-qa.json --runs 5
    evalforge compare --config configs/provider-bakeoff.json
    evalforge baseline --config configs/support-qa.json --out baselines/support-qa.json
    evalforge gate --config configs/support-qa.json --baseline baselines/support-qa.json

Configs are JSON, not YAML, so the whole tool depends on the standard library
plus httpx.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .baseline import Baseline, check_regression
from .dataset import Dataset, load_jsonl, load_many
from .providers import (
    CallableTarget,
    HttpTarget,
    PromptTarget,
    ResponseCache,
    ScriptedLLM,
    build_llm,
)
from .report import print_comparison, print_run, print_variance, write_reports
from .runner import Runner, compare, compare_many
from .scorers import CompositeScorer, build_scorer
from .stats import min_detectable_effect, required_n


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "dataset" not in data and "datasets" not in data:
        raise ValueError(f"{path}: config needs 'dataset' or 'datasets'")
    return data


def _resolve(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else (base / p)


def build_dataset(config: dict[str, Any], base: Path) -> Dataset:
    if "datasets" in config:
        paths = [_resolve(p, base) for p in config["datasets"]]
        ds = load_many(paths)
    else:
        ds = load_jsonl(_resolve(config["dataset"], base))

    if config.get("category"):
        ds = ds.filter(category=config["category"])
    if config.get("sample"):
        ds = ds.sample(int(config["sample"]))
    return ds


def build_target(spec: dict[str, Any], cache: ResponseCache | None = None) -> Any:
    kind = spec.get("type", "prompt")

    if kind == "prompt":
        llm = build_llm(spec.get("provider", ""), cache=cache)
        if isinstance(llm, ScriptedLLM) and spec.get("scripted_responses"):
            llm.responses = spec["scripted_responses"]
        return PromptTarget(
            llm=llm,
            name=spec.get("name", spec.get("provider", "prompt")),
            system=spec.get("system", ""),
            template=spec.get("template", "{input}"),
            temperature=float(spec.get("temperature", 0.0)),
            max_tokens=int(spec.get("max_tokens", 900)),
            json_mode=bool(spec.get("json_mode", False)),
        )

    if kind == "http":
        return HttpTarget(
            url=spec["url"],
            name=spec.get("name", "http"),
            method=spec.get("method", "POST"),
            body_template=spec.get("body", {"query": "{input}"}),
            response_path=spec.get("response_path", "answer"),
            headers=spec.get("headers", {}),
            timeout=float(spec.get("timeout", 120)),
        )

    raise ValueError(f"unknown target type {kind!r}; use 'prompt' or 'http'")


def build_scorer_from_config(config: dict[str, Any], llm: Any) -> Any:
    specs = config.get("scorers") or ["contains_any"]
    scorers: list[tuple[Any, float, str]] = []
    for spec in specs:
        if isinstance(spec, str):
            scorer, weight, field_name = build_scorer(spec, llm), 1.0, _default_field(spec)
        else:
            weight = float(spec.pop("weight", 1.0))
            field_name = spec.pop("field", _default_field(spec.get("type", "")))
            scorer = build_scorer(spec, llm)
        scorers.append((scorer, weight, field_name))

    return CompositeScorer(
        scorers=scorers,
        require_all=bool(config.get("require_all_scorers", True)),
    )


def _default_field(kind: str) -> str:
    """Which dataset field each scorer reads. Keeping this mapping explicit is
    what lets one item carry several expectations and be scored by several
    scorers without ambiguity."""
    return {
        "contains_any": "contains_any",
        "contains_all": "contains_all",
        "must_not_contain": "must_not_contain",
        "refusal": "should_refuse",
        "json_schema": "required_keys",
        "citation_validity": "n_sources",
        "numeric": "numeric",
        "llm_judge": "rubric",
    }.get(kind, "")


def make_cache(config: dict[str, Any], disabled: bool = False) -> ResponseCache:
    return ResponseCache(
        directory=Path(config.get("cache_dir", ".evalforge_cache")),
        enabled=not disabled and bool(config.get("cache", True)),
    )


def build_runner(config: dict[str, Any], base: Path, *, no_cache: bool = False) -> tuple[Runner, Dataset]:
    cache = make_cache(config, disabled=no_cache)
    target_spec = config.get("target") or {"type": "prompt"}
    target = build_target(target_spec, cache)
    judge_llm = build_llm(config.get("judge_provider", target_spec.get("provider", "")), cache=cache)
    scorer = build_scorer_from_config(config, judge_llm)
    dataset = build_dataset(config, base)
    return (
        Runner(
            target=target,
            scorer=scorer,
            name=config.get("name", target_spec.get("name", "run")),
            concurrency=int(config.get("concurrency", 4)),
        ),
        dataset,
    )


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout.strip()
    except Exception:
        return ""


def warn_dataset(dataset: Dataset) -> None:
    problems = dataset.validate()
    if problems:
        print(f"  dataset warnings ({len(problems)}):", file=sys.stderr)
        for p in problems:
            print(f"    - {p}", file=sys.stderr)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    ds = load_jsonl(args.path)
    problems = ds.validate()
    print(f"\n  {ds.name}: {len(ds)} items, fingerprint {ds.fingerprint[:12]}")
    print(f"  categories: {ds.categories()}")
    negatives = sum(1 for i in ds if i.should_refuse)
    print(f"  negative (should-refuse) cases: {negatives}")
    print(f"  detectable effect at this size: {min_detectable_effect(len(ds)):.1%}")

    if not problems:
        print("\n  no integrity problems\n")
        return 0
    print(f"\n  {len(problems)} problem(s):")
    for p in problems:
        print(f"    - {p}")
    print()
    return 1 if args.strict else 0


def cmd_power(args: argparse.Namespace) -> int:
    print("\n  What an eval set of each size can actually detect")
    print("  (binary metric, baseline ~{:.0%}, 95% confidence, 80% power)\n".format(args.baseline))
    print(f"  {'items':>8}  {'detectable difference':>22}")
    print("  " + "-" * 34)
    for n in (20, 30, 40, 50, 100, 200, 500, 1000, 2000):
        marker = "  <- yours" if args.n and n >= args.n and n < args.n * 2 else ""
        print(f"  {n:>8}  {min_detectable_effect(n, args.baseline):>21.1%}{marker}")

    if args.n:
        mde = min_detectable_effect(args.n, args.baseline)
        print(
            f"\n  With {args.n} items you can detect a change of {mde:.1%} or larger.\n"
            f"  A smaller measured difference is not evidence of anything."
        )
    print("\n  Items needed to detect a given difference:\n")
    print(f"  {'difference':>12}  {'items needed':>14}")
    print("  " + "-" * 30)
    for effect in (0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20):
        print(f"  {effect:>11.0%}  {required_n(effect, args.baseline):>14,}")
    print()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = Path(args.config).parent
    runner, dataset = build_runner(config, base, no_cache=args.no_cache)

    print(f"\n  running {runner.name} on {dataset.name} ({len(dataset)} items)")
    warn_dataset(dataset)

    result = runner.run(dataset)
    print_run(result, args.confidence)

    if args.out:
        paths = write_reports([result], args.out, confidence=args.confidence)
        print(f"\n  reports: {', '.join(str(p) for p in paths.values())}")
    return 0


def cmd_variance(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = Path(args.config).parent
    # Caching would make repeated runs identical and hide the very thing being
    # measured, so it is forced off here.
    runner, dataset = build_runner(config, base, no_cache=True)

    print(f"\n  measuring variance: {args.runs} identical runs of {runner.name}")
    runs, variance = runner.run_repeated(dataset, runs=args.runs)
    print_variance(variance)

    if args.out:
        write_reports(runs[:1], args.out, variance=variance, confidence=args.confidence)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = Path(args.config).parent
    variants = config.get("variants")
    if not variants:
        print("config needs a 'variants' list for compare", file=sys.stderr)
        return 2

    dataset = build_dataset(config, base)
    warn_dataset(dataset)
    cache = make_cache(config, disabled=args.no_cache)

    runs = []
    for variant in variants:
        merged = {**config, **variant}
        target = build_target(merged.get("target", {"type": "prompt", **variant}), cache)
        judge = build_llm(config.get("judge_provider", ""), cache=cache)
        runner = Runner(
            target=target,
            scorer=build_scorer_from_config(config, judge),
            name=variant.get("name", getattr(target, "name", "variant")),
            concurrency=int(config.get("concurrency", 4)),
        )
        print(f"\n  running {runner.name} ...")
        runs.append(runner.run(dataset))

    for run in runs:
        print_run(run, args.confidence)

    comparisons = compare_many(runs, confidence=args.confidence)
    print_comparison(comparisons)

    if args.out:
        paths = write_reports(runs, args.out, comparisons, confidence=args.confidence)
        print(f"\n  reports: {', '.join(str(p) for p in paths.values())}")
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = Path(args.config).parent
    runner, dataset = build_runner(config, base, no_cache=args.no_cache)

    print(f"\n  recording baseline for {runner.name} on {dataset.name}")
    warn_dataset(dataset)
    result = runner.run(dataset)
    print_run(result, args.confidence)

    baseline = Baseline.from_run(result, git_sha=git_sha(), notes=args.notes or "")
    path = baseline.save(args.out)
    print(
        f"\n  baseline written to {path}"
        f"\n  pass rate {baseline.pass_rate:.3f} "
        f"[{baseline.pass_rate_ci[0]:.3f}, {baseline.pass_rate_ci[1]:.3f}]"
        f"\n  dataset fingerprint {baseline.dataset_fingerprint[:12]}\n"
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = Path(args.config).parent

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"\n  no baseline at {baseline_path}."
            "\n  record one first:  evalforge baseline --config ... --out ...\n",
            file=sys.stderr,
        )
        return 2

    baseline = Baseline.load(baseline_path)
    runner, dataset = build_runner(config, base, no_cache=args.no_cache)

    print(f"\n  gating {runner.name} against {baseline_path.name}")
    result = runner.run(dataset)
    print_run(result, args.confidence)

    gate = check_regression(
        result,
        baseline,
        tolerance=args.tolerance,
        max_new_failures=args.max_new_failures,
        max_cost_increase=args.max_cost_increase,
        max_latency_increase=args.max_latency_increase,
        confidence=args.confidence,
    )
    print(gate.report())

    if args.out:
        write_reports([result], args.out, confidence=args.confidence)
    return 0 if gate.passed else 1


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evalforge", description="LLM evaluation harness")
    ap.add_argument("--confidence", type=float, default=0.95)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("validate", help="check dataset integrity")
    p.add_argument("path")
    p.add_argument("--strict", action="store_true", help="exit non-zero on any problem")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("power", help="what can an eval of size N detect?")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--baseline", type=float, default=0.80)
    p.set_defaults(fn=cmd_power)

    p = sub.add_parser("run", help="run an eval")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="directory for json/md/html reports")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("variance", help="measure run-to-run variance")
    p.add_argument("--config", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_variance, no_cache=True)

    p = sub.add_parser("compare", help="paired comparison of variants")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("baseline", help="record a baseline")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_baseline)

    p = sub.add_parser("gate", help="fail if quality regressed")
    p.add_argument("--config", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--max-new-failures", type=int, default=0)
    p.add_argument("--max-cost-increase", type=float, default=0.5)
    p.add_argument("--max-latency-increase", type=float, default=0.5)
    p.add_argument("--out", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(fn=cmd_gate)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n  error: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
