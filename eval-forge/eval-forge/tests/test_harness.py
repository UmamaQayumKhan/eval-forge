"""Tests for scorers, datasets, the runner and the regression gate.

The harness is tested against a scripted provider rather than a real model: an
eval harness whose own test suite is flaky and expensive is one nobody runs.
"""

from __future__ import annotations

import json

import pytest

from evalforge.baseline import Baseline, check_regression
from evalforge.dataset import Dataset, Item, load_jsonl
from evalforge.providers import CallableTarget, PromptTarget, ResponseCache, ScriptedLLM
from evalforge.report import html_report, markdown_report
from evalforge.runner import Runner, compare
from evalforge.scorers import (
    CitationValidity,
    CompositeScorer,
    ContainsAll,
    ContainsAny,
    ExactMatch,
    JsonSchemaMatch,
    LLMJudge,
    MustNotContain,
    NumericTolerance,
    RefusalMatch,
)


# --- scorers -------------------------------------------------------------


def test_contains_any_is_case_and_whitespace_insensitive():
    s = ContainsAny().score("Suspended after  14 DAYS.", ["14 days"], {})
    assert s.passed


def test_contains_all_gives_partial_credit():
    s = ContainsAll().score("us-east-1 and eu-central-1", ["us-east-1", "eu-central-1", "ap-southeast-2"], {})
    assert not s.passed
    assert s.value == pytest.approx(2 / 3)


def test_must_not_contain_catches_a_leak():
    s = MustNotContain().score("I've refunded your account.", ["i've refunded"], {})
    assert not s.passed
    assert "i've refunded" in s.detail


def test_numeric_tolerance_parses_currency_and_separators():
    """Models write '$1,364.00' and '1364' for the same answer; exact string
    matching on a number is a false-negative factory."""
    scorer = NumericTolerance()
    assert scorer.score("The refund is $1,364.00", 1364, {}).passed
    assert scorer.score("1364", 1364, {}).passed
    assert not scorer.score("The refund is $1,500", 1364, {}).passed


def test_numeric_picks_the_closest_number_in_a_sentence():
    s = NumericTolerance().score("Retried on days 1, 3 and 7, suspended on day 14.", 14, {})
    assert s.passed


def test_numeric_with_no_number_fails_cleanly():
    s = NumericTolerance().score("It depends on your plan.", 14, {})
    assert not s.passed and "no number" in s.detail


def test_json_schema_gives_partial_credit_for_a_near_miss():
    out = '{"invoiceNumber": "INV-1", "vendorName": "Acme", "total": 100}'
    s = JsonSchemaMatch().score(out, ["invoiceNumber", "vendorName", "total", "currency"], {})
    assert not s.passed
    assert s.value == pytest.approx(0.75)
    assert "currency" in s.detail


def test_json_schema_handles_fenced_output():
    out = 'Here you go:\n```json\n{"a": 1, "b": 2}\n```'
    assert JsonSchemaMatch().score(out, ["a", "b"], {}).passed


def test_json_schema_rejects_prose():
    assert not JsonSchemaMatch().score("no json here", ["a"], {}).passed


def test_citation_validity_catches_a_fabricated_source():
    """The cheapest reliable hallucination signal: citing source [7] when four
    were supplied."""
    s = CitationValidity().score("Suspension is after 14 days [7].", 4, {})
    assert not s.passed and "[7]" in s.detail


def test_citation_validity_accepts_real_citations():
    assert CitationValidity().score("After 14 days [1] and then deleted [2].", 4, {}).passed


def test_citation_validity_fails_an_uncited_answer():
    assert not CitationValidity().score("After 14 days.", 4, {}).passed


@pytest.mark.parametrize(
    "text,refused",
    [
        ("I don't have enough information to answer that.", True),
        ("I'm not able to share that.", True),
        ("I won't repeat a password.", True),
        ("That's not something I can do.", True),
        ("Suspension happens after 14 days.", False),
        ("The ACS URL is https://auth.northwind.cloud/saml/acs", False),
    ],
)
def test_refusal_detection_covers_varied_phrasing(text, refused):
    """The first version of this marker list was narrow and scored genuine
    refusals as non-refusals, making a red-team suite look like a catastrophic
    failure when the model was declining correctly."""
    assert RefusalMatch().refused(text) is refused


def test_refusal_scorer_is_bidirectional():
    scorer = RefusalMatch()
    assert scorer.score("I'm not able to share that.", True, {}).passed
    assert not scorer.score("Here is the answer: 14 days.", True, {}).passed
    assert scorer.score("Here is the answer: 14 days.", False, {}).passed
    assert not scorer.score("I'm not able to share that.", False, {}).passed


# --- composite -----------------------------------------------------------


def test_composite_require_all_fails_on_any_component():
    """A correct answer that also leaks a banned phrase must fail outright, not
    average to 0.8. For safety items a weighted mean is the wrong shape."""
    composite = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (MustNotContain(), 1.0, "must_not_contain")],
        require_all=True,
    )
    item = {"contains_any": ["14 days"], "must_not_contain": ["i've refunded"]}
    s = composite.score("It's 14 days. I've refunded you.", None, item)
    assert not s.passed
    assert s.value > 0.0  # partial credit is still reported


def test_composite_skips_scorers_the_item_does_not_exercise():
    composite = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (RefusalMatch(), 1.0, "should_refuse")],
        require_all=True,
    )
    # should_refuse is absent (None) so the refusal scorer must not be applied
    s = composite.score("14 days", None, {"contains_any": ["14 days"], "should_refuse": None})
    assert s.passed
    assert "refusal" not in s.metadata.get("components", {})


def test_composite_with_no_applicable_scorer_fails_loudly():
    composite = CompositeScorer(scorers=[(ContainsAny(), 1.0, "contains_any")])
    s = composite.score("anything", None, {"contains_any": []})
    assert not s.passed and "no applicable scorer" in s.detail


# --- LLM judge -----------------------------------------------------------


def test_judge_parses_a_verdict():
    llm = ScriptedLLM(
        responses={"RUBRIC": json.dumps({"verdict": "pass", "score": 1.0, "reasons": ["ok"]})}
    )
    s = LLMJudge(llm=llm).score("an answer", None, {"input": "q", "rubric": "must say 14 days"})
    assert s.passed and s.value == 1.0


def test_judge_parse_failure_is_reported_not_silently_zero():
    """A judge outage scoring everything zero looks exactly like a catastrophic
    regression, which is the worst way for a harness to fail."""
    llm = ScriptedLLM(responses={}, default="not json at all")
    s = LLMJudge(llm=llm).score("an answer", None, {"input": "q", "rubric": "r"})
    assert not s.passed
    assert s.metadata.get("judge_error") is True


def test_judge_without_a_rubric_uses_the_fallback():
    s = LLMJudge(llm=ScriptedLLM(), fallback=ContainsAny()).score(
        "14 days", ["14 days"], {"input": "q", "contains_any": ["14 days"]}
    )
    assert s.passed


# --- dataset -------------------------------------------------------------


def test_real_dataset_loads_and_is_clean(support_dataset):
    assert len(support_dataset) > 40
    assert support_dataset.validate() == []


def test_redteam_dataset_has_no_integrity_problems(redteam_dataset):
    """The only acceptable complaint is the size note. A red-team suite is
    deliberately small -- attacks are hand-written -- so the wide-interval
    warning is expected and is informational, not an integrity failure."""
    problems = [p for p in redteam_dataset.validate() if "items; confidence intervals" not in p]
    assert problems == []


def test_fingerprint_changes_when_an_expectation_changes(support_dataset):
    """A gate comparing against a baseline from a different dataset is worse
    than no gate, and without a fingerprint that mistake is invisible."""
    before = support_dataset.fingerprint
    support_dataset.items[0].contains_any = ["something else entirely"]
    assert support_dataset.fingerprint != before


def test_validate_flags_duplicate_inputs():
    ds = Dataset(
        name="t",
        items=[
            Item(id="a", input="Same question?", contains_any=["x"]),
            Item(id="b", input="same   QUESTION?", contains_any=["x"]),
        ],
    )
    problems = " ".join(ds.validate())
    assert "duplicate input" in problems


def test_validate_flags_items_with_no_expectation():
    ds = Dataset(name="t", items=[Item(id="a", input="q")])
    assert any("no expectation" in p for p in ds.validate())


def test_validate_flags_a_set_with_no_negative_cases():
    ds = Dataset(name="t", items=[Item(id=str(i), input=f"q{i}", contains_any=["x"]) for i in range(40)])
    assert any("no negative cases" in p for p in ds.validate())


def test_sample_is_deterministic_and_stratified(support_dataset):
    a = support_dataset.sample(12)
    b = support_dataset.sample(12)
    assert [i.id for i in a] == [i.id for i in b]
    assert len(a.categories()) >= 4


def test_bad_jsonl_reports_the_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","input":"q"}\n{nope\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_jsonl(path)


def test_missing_input_field_is_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","expected":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field 'input'"):
        load_jsonl(path)


# --- runner --------------------------------------------------------------


def _dataset() -> Dataset:
    return Dataset(
        name="tiny",
        items=[
            Item(id="q1", input="suspend", contains_any=["14 days"], category="billing"),
            Item(id="q2", input="refund", contains_any=["not refundable"], category="billing"),
            Item(id="q3", input="unknown topic", should_refuse=True, category="unanswerable"),
        ],
    )


def _target(responses: dict[str, str]) -> PromptTarget:
    return PromptTarget(llm=ScriptedLLM(responses=responses), name="t")


def test_runner_scores_every_item_and_preserves_order():
    target = _target({"suspend": "14 days", "refund": "not refundable"})
    scorer = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (RefusalMatch(), 1.0, "should_refuse")]
    )
    result = Runner(target=target, scorer=scorer, concurrency=2).run(_dataset())
    assert [r.item_id for r in result.results] == ["q1", "q2", "q3"]
    assert result.pass_rate().point == pytest.approx(1.0)


def test_runner_records_a_target_error_without_voiding_the_run():
    def explode(item):
        raise RuntimeError("upstream down")

    scorer = CompositeScorer(scorers=[(ContainsAny(), 1.0, "contains_any")])
    result = Runner(target=CallableTarget(fn=explode), scorer=scorer, concurrency=1).run(_dataset())
    assert len(result.errors) == 3
    assert all(not r.passed for r in result.results)


def test_summary_reports_intervals_and_detectable_effect():
    target = _target({"suspend": "14 days", "refund": "not refundable"})
    scorer = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (RefusalMatch(), 1.0, "should_refuse")]
    )
    summary = Runner(target=target, scorer=scorer).run(_dataset()).summary()
    assert "pass_rate_ci" in summary and len(summary["pass_rate_ci"]) == 2
    assert summary["min_detectable_effect"] > 0
    assert "billing" in summary["by_category"]


def test_compare_refuses_runs_from_different_datasets():
    scorer = CompositeScorer(scorers=[(ContainsAny(), 1.0, "contains_any")])
    a = Runner(target=_target({}), scorer=scorer, name="a").run(_dataset())
    other = Dataset(name="other", items=[Item(id="z", input="x", contains_any=["y"])])
    b = Runner(target=_target({}), scorer=scorer, name="b").run(other)
    with pytest.raises(ValueError, match="different datasets"):
        compare(a, b)


def test_repeated_runs_measure_variance():
    target = _target({"suspend": "14 days", "refund": "not refundable"})
    scorer = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (RefusalMatch(), 1.0, "should_refuse")]
    )
    _, variance = Runner(target=target, scorer=scorer).run_repeated(_dataset(), runs=3)
    assert variance.runs == 3
    assert variance.stable  # scripted provider is deterministic


# --- cache ---------------------------------------------------------------


def test_cache_returns_a_hit_on_the_second_call(tmp_path):
    cache = ResponseCache(directory=tmp_path / "c")
    key = ResponseCache.key("m", [{"role": "user", "content": "hi"}])
    assert cache.get(key) is None
    cache.put(key, {"text": "hello"})
    assert cache.get(key)["text"] == "hello"
    assert cache.stats()["hits"] == 1


def test_cache_key_depends_on_the_request():
    a = ResponseCache.key("m", [{"role": "user", "content": "x"}], temperature=0.0)
    b = ResponseCache.key("m", [{"role": "user", "content": "x"}], temperature=0.7)
    assert a != b


# --- baseline and gate ---------------------------------------------------


def _run(responses: dict[str, str], name: str = "r"):
    scorer = CompositeScorer(
        scorers=[(ContainsAny(), 1.0, "contains_any"), (RefusalMatch(), 1.0, "should_refuse")]
    )
    return Runner(target=_target(responses), scorer=scorer, name=name).run(_dataset())


GOOD = {"suspend": "14 days", "refund": "not refundable"}


def test_gate_passes_when_nothing_changed(tmp_path):
    baseline = Baseline.from_run(_run(GOOD, "base"))
    gate = check_regression(_run(GOOD, "now"), baseline)
    assert gate.passed


def test_gate_catches_a_per_item_regression_the_aggregate_hides():
    """The core argument for per-item tracking: a change that breaks real items
    can leave the aggregate inside its confidence interval and clear every
    statistical test."""
    baseline = Baseline.from_run(_run(GOOD, "base"))
    broken = check_regression(_run({"suspend": "14 days", "refund": "yes, fully refundable"}, "now"), baseline)
    assert not broken.passed
    assert "q2" in broken.regressed_items


def test_gate_refuses_a_baseline_from_a_different_dataset():
    baseline = Baseline.from_run(_run(GOOD, "base"))
    baseline.dataset_fingerprint = "deadbeef" * 4
    gate = check_regression(_run(GOOD, "now"), baseline)
    assert not gate.passed
    assert any("dataset changed" in c["detail"] for c in gate.checks)


def test_gate_warns_when_the_set_is_too_small_to_be_evidence():
    baseline = Baseline.from_run(_run(GOOD, "base"))
    gate = check_regression(_run(GOOD, "now"), baseline)
    assert gate.min_detectable_effect > 0.1
    assert any("cannot detect a regression smaller" in w for w in gate.warnings)


def test_gate_flags_churn_that_cancels_out():
    """An unchanged headline number can hide fixes and breaks in equal measure,
    and the breaks are what someone hits in production."""
    baseline = Baseline.from_run(_run({"suspend": "14 days", "refund": "wrong"}, "base"))
    churned = check_regression(_run({"suspend": "wrong", "refund": "not refundable"}, "now"), baseline)
    assert churned.regressed_items and churned.fixed_items
    assert any("hiding real churn" in w for w in churned.warnings)


def test_gate_fails_on_target_errors():
    def explode(item):
        raise RuntimeError("down")

    baseline = Baseline.from_run(_run(GOOD, "base"))
    scorer = CompositeScorer(scorers=[(ContainsAny(), 1.0, "contains_any")])
    broken = Runner(target=CallableTarget(fn=explode), scorer=scorer, name="now").run(_dataset())
    broken.dataset_fingerprint = baseline.dataset_fingerprint
    gate = check_regression(broken, baseline)
    assert not gate.passed
    assert any(c["name"] == "target errors" for c in gate.checks)


def test_baseline_round_trips(tmp_path):
    baseline = Baseline.from_run(_run(GOOD, "base"), notes="n")
    path = baseline.save(tmp_path / "b.json")
    loaded = Baseline.load(path)
    assert loaded.pass_rate == baseline.pass_rate
    assert loaded.item_outcomes == baseline.item_outcomes


# --- reports -------------------------------------------------------------


def test_markdown_report_shows_the_interval_not_a_bare_number():
    md = markdown_report([_run(GOOD)])
    assert "95% CI" in md
    assert "detect a difference" in md


def test_html_report_is_self_contained_and_escaped():
    html = html_report([_run(GOOD)])
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "http://" not in html.split("</style>")[0]  # no external assets


def test_html_report_escapes_output_text():
    run = _run({"suspend": "<script>alert(1)</script>"})
    html = html_report([run])
    assert "<script>alert(1)</script>" not in html
