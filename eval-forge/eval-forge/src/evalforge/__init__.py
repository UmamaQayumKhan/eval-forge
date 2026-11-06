"""EvalForge — an LLM evaluation harness that reports uncertainty."""

from .baseline import Baseline, GateResult, check_regression
from .dataset import Dataset, Item, load_jsonl, load_many
from .providers import CallableTarget, HttpTarget, PromptTarget, ResponseCache, ScriptedLLM
from .report import html_report, markdown_report, write_reports
from .runner import ItemResult, Runner, RunResult, compare, compare_many
from .scorers import (
    CitationValidity,
    CompositeScorer,
    ContainsAny,
    JsonSchemaMatch,
    LLMJudge,
    MustNotContain,
    NumericTolerance,
    RefusalMatch,
    Score,
    build_scorer,
)
from .stats import (
    Comparison,
    Interval,
    agreement,
    bootstrap_interval,
    mcnemar_exact,
    min_detectable_effect,
    paired_bootstrap,
    required_n,
    run_variance,
    wilson_interval,
)

__version__ = "1.0.0"

__all__ = [
    "Baseline",
    "CallableTarget",
    "CitationValidity",
    "Comparison",
    "CompositeScorer",
    "ContainsAny",
    "Dataset",
    "GateResult",
    "HttpTarget",
    "Interval",
    "Item",
    "ItemResult",
    "JsonSchemaMatch",
    "LLMJudge",
    "MustNotContain",
    "NumericTolerance",
    "PromptTarget",
    "RefusalMatch",
    "ResponseCache",
    "RunResult",
    "Runner",
    "Score",
    "ScriptedLLM",
    "agreement",
    "bootstrap_interval",
    "build_scorer",
    "check_regression",
    "compare",
    "compare_many",
    "html_report",
    "load_jsonl",
    "load_many",
    "markdown_report",
    "mcnemar_exact",
    "min_detectable_effect",
    "paired_bootstrap",
    "required_n",
    "run_variance",
    "wilson_interval",
    "write_reports",
    "__version__",
]
