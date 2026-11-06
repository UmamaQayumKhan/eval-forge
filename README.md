# EvalForge — an LLM eval harness that reports uncertainty

A golden-dataset evaluation harness with confidence intervals on every metric,
paired significance testing between systems, run-to-run variance measurement,
and a CI gate that fails the build on regression.

Runs with **no API keys** against a scripted provider, which is how the 85 tests
and the CI gate execute.

```bash
pip install -r requirements-dev.txt
make test        # 85 tests
make power       # what can an eval of size N actually detect?
make run         # support-QA eval, 44 items
make redteam     # 20-item adversarial suite
make gate        # the regression gate CI runs
make report      # json + markdown + self-contained HTML
```

---

## The problem

Standard practice is: run the eval once, get 0.84, change a prompt, get 0.87,
ship it. That comparison is usually meaningless, for three compounding reasons.

**The interval is wider than the change.** On 44 items a pass rate of 0.93 has a
95% confidence interval of **[0.818, 0.977]** — a margin of ±8 points. A
3-point "improvement" sits comfortably inside the noise.

**The sample size can't see what you're claiming.** `make power`:

| items | smallest difference detectable |
| --- | --- |
| 20 | 35.4% |
| 40 | 25.1% |
| 100 | 15.8% |
| 500 | 7.1% |
| 1,000 | 5.0% |
| 2,000 | 3.5% |

Detecting a 3-point difference on a binary metric at 80% power needs **over
1,000 items**. Nobody's golden set has that, which is why so many reported
prompt improvements are noise.

**Runs vary.** `make variance` runs the same eval repeatedly. If identical runs
differ by 4 points, no 3-point change is measurable at all, and every comparison
after that is theatre.

So this harness refuses to print a bare number. Every metric carries an
interval, every system comparison is paired, and every gate states how small a
change it is capable of detecting.

---

## What makes the comparisons trustworthy

### Paired comparison, not two independent means

Comparing two independent means carries item-difficulty variance twice.
Resampling item *indices* and taking the difference within each resample cancels
it, which typically shrinks the interval enough to see a few points of
difference on a 44-item set instead of needing hundreds. A difference is
reported as real **only when the interval on the difference excludes zero**.

```
concise-prompt 0.477 vs verbose-prompt 0.455
  delta -0.023  95% CI [-0.068, +0.000]  NOT significant, p=1.0000
  verdict: no detectable difference
  discordant: concise-only 1, verbose-only 0 (both right 20, both wrong 23)
```

Two prompts, a 2-point gap, and the honest answer is "no detectable
difference". That output is the product.

### McNemar on the discordant pairs

For binary metrics the exact binomial McNemar test runs alongside. Only items
where exactly one system was right carry information — items both got right say
nothing about which is better. The exact binomial is used rather than
chi-squared because discordant counts in an LLM eval are routinely under 10,
where the approximation is unreliable.

### Wilson intervals, not the normal approximation

Eval metrics live near 0 and 1, exactly where the normal interval returns bounds
outside [0, 1]. At 19/20 the normal interval runs past 1.0; Wilson gives
[0.764, 0.991]. Verified against textbook values in `test_stats.py`.

### Cohen's kappa for judge validation

Raw agreement is misleading when labels are imbalanced: on a 90%-pass set, a
judge that passes everything scores 0.90 agreement and has learned nothing.
Kappa corrects for chance and shows that as ~0. `test_kappa_exposes_a_judge_that_always_says_pass`
pins it.

---

## The regression gate

`make gate` compares a run against a recorded baseline and exits non-zero on
regression. Two decisions make it trustworthy rather than annoying:

**It gates on the baseline's lower confidence bound, not its point estimate.**
Gating on "today must beat yesterday" fails the build on noise about half the
time, and a gate that cries wolf is disabled within a week.

**It refuses to run against a baseline from a different dataset.** Every
baseline stores a fingerprint of the dataset's inputs and expectations. A
comparison across golden sets produces plausible-looking numbers that mean
nothing, and without a fingerprint that mistake is invisible.

### What per-item tracking catches that aggregates don't

This is the gate run after breaking two previously-correct answers:

```
  [PASS] pass rate: 0.886 vs baseline 0.932 (lower bound 0.818, floor 0.818)
  [FAIL] per-item regressions: 2 newly failing, 0 newly passing (limit 0)

  paired comparison: baseline 0.932 vs current 0.886
      delta -0.045 [-0.114, 0.000] | NOT significant, p=0.5000

  NEWLY FAILING (2): bill-02, sec-01

  GATE FAILED
```

The aggregate check **passed**. The significance test said **not significant**.
Both were right — a 4.5-point drop on 44 items genuinely is inside the noise.
And two real answers had broken, one of them now claiming a non-refundable plan
is refundable and the other claiming HIPAA eligibility the product does not
have.

That is the argument for tracking item outcomes, in one screen. The gate also
warns when an unchanged aggregate is hiding equal numbers of fixes and breaks,
because the breaks are what someone hits in production.

The gate additionally fails on cost-per-item or p95-latency growth beyond a
threshold, and on any target error — a flaky endpoint must never pass silently.

CI contains a **gate self-test** that deliberately breaks two answers and
asserts the gate rejects them. A gate that only ever passes is
indistinguishable from no gate, and that job is the cheapest way to know which
one you have.

---

## Scorers

Deterministic first, LLM judge last. Each judge call costs money, latency and
its own variance, so anything checkable in code is checked in code.

| scorer | what it is for |
| --- | --- |
| `contains_any` / `contains_all` | the load-bearing token must appear |
| `must_not_contain` | negative constraints, scored separately so a leak can't average away |
| `numeric` | tolerance-based; models write `$1,364.00` and `1364` for the same answer |
| `json_schema` | partial credit — 5 of 6 required keys scores 0.83, not 0 |
| `citation_validity` | a citation to source `[7]` when four were supplied is fabrication |
| `refusal` | bidirectional: must-refuse *and* must-not-over-refuse |
| `regex`, `exact_match` | exact-form checks |
| `llm_judge` | per-item rubric, temperature 0, parse failures reported not silently zeroed |

`require_all_scorers` matters most on safety items: an answer that is accurate
but also echoes a card number must fail outright, not score 0.8.

Two judge details worth noting. Rubrics are **per-item**, because a global "is
this good?" judge regresses to 0.7 for everything. And a judge parse failure is
reported as an error rather than scored 0 — a judge outage scoring everything
zero looks exactly like a catastrophic regression, which is the worst possible
way for a harness to fail.

### Three scorer bugs this project found in its own datasets

All three were found by running the suites, not by reading the code:

| finding | why it mattered |
| --- | --- |
| The refusal marker list was too narrow, so genuine refusals ("I'm not able to share that") scored as non-refusals. The red-team suite read **0.250** when the model was declining correctly — it is **0.850** with the list fixed, and 1.000 after the label fixes below. | A safety metric that under-reports safety is worse than no metric: it hides which failures are real. |
| `should_refuse` was a two-state bool defaulting to False, which conflated "refused to answer" with "declined to act". A reply like *"I'm not able to process refunds myself, I'll pass this to a colleague"* was marked an over-refusal. It is now tri-state: `True`, `False`, or unset meaning *don't assert*. | Most items genuinely don't care, and forcing a default makes the metric lie. |
| Two negative constraints fired on their **own negation**: `"guarantee"` matched inside *"I'm not able to make guarantees"*, and `"has been cancelled"` matched inside *"nothing has been cancelled from my side"*. | The clearest demonstration of why substring constraints are brittle, and why they need testing against correct answers, not just wrong ones. |

A fourth, smaller one: `api-06` expected the substring `"7 day"` while a correct
answer said *"7 consecutive days"*. The `numeric` scorer caught the number
correctly; the substring check was the thing that was wrong.

---

## Red-team suite

20 adversarial items across five categories — prompt injection (override,
exfiltration, false authority, framing, tag injection, obfuscation,
social-engineering), internal-policy leakage, PII echo, unbacked commitments,
and abuse.

```
  pass rate   1.000  [0.839, 1.000]
```

Note what the interval says even at 20/20: the true rate could be as low as
**84%**. Twenty items is not enough to claim a safety property, and the harness
says so instead of letting the headline number imply otherwise.

---

## Evaluating a service, not just a prompt

The `http` target points the harness at a running system, which is the
difference between evaluating a prompt and evaluating the thing users actually
hit:

```json
{
  "target": {
    "type": "http",
    "url": "http://localhost:8000/ask",
    "body": { "query": "{input}", "top_k": 5 },
    "response_path": "answer"
  },
  "scorers": ["contains_any", "must_not_contain", "refusal",
              { "type": "citation_validity", "field": "n_sources" }]
}
```

`configs/atlas-rag-http.json` scores the companion AtlasRAG service end to end —
retrieval, reranking, generation and citations — on the same 44 items.

Responses are cached on disk by content hash, so iterating on a scorer doesn't
re-pay for every generation. That is the main reason people stop running their
evals. `variance` forces the cache off, since caching would make repeated runs
identical and hide the very thing being measured.

---

## Commands

| command | purpose |
| --- | --- |
| `evalforge validate <dataset>` | integrity check: duplicate inputs, missing expectations, no negative cases |
| `evalforge power --n 44` | what a set of this size can detect |
| `evalforge run --config c.json` | run an eval, optionally emit reports |
| `evalforge variance --config c.json --runs 5` | run-to-run spread |
| `evalforge compare --config c.json` | paired comparison of variants |
| `evalforge baseline --config c.json --out b.json` | record a baseline |
| `evalforge gate --config c.json --baseline b.json` | fail on regression |

Configs are JSON, so the whole tool depends on the standard library plus httpx.

### Dataset integrity

`validate` exists because a quietly broken golden set produces confident,
meaningless numbers. It flags duplicate inputs (which double-weight a
question), items with no expectation at all (scored as free passes), blank
inputs, and — importantly — **a set with no negative cases**, because a system
that answers everything will score perfectly and be unsafe.

---

## Layout

```
src/evalforge/
  stats.py      Wilson · bootstrap · paired bootstrap · McNemar · power · kappa
  scorers.py    9 deterministic scorers + a per-item-rubric LLM judge
  dataset.py    loading, stratified sampling, fingerprinting, integrity checks
  providers.py  Groq/Gemini/OpenAI/scripted + prompt, HTTP and callable targets
  runner.py     orchestration, aggregation with intervals, comparison
  baseline.py   baselines and the regression gate
  report.py     terminal, markdown, self-contained HTML
  cli.py        validate · power · run · variance · compare · baseline · gate
datasets/       support-qa (44) · redteam (20) · extraction (12)
configs/        support-qa · provider-bakeoff · redteam · atlas-rag-http
tests/          85 tests, stats verified against textbook values
```

## Known limits

- 44 items cannot detect a change smaller than about 18%. The harness reports
  this everywhere, but it is a property of the dataset, not something the
  statistics can fix — the fix is more items.
- Bootstrap intervals assume items are independent. If a golden set contains
  several paraphrases of one question, the effective sample size is smaller than
  the item count and the intervals are optimistic. `validate` catches exact
  duplicates, not paraphrases.
- The LLM judge is validated against human labels via kappa, but the
  `human_label` column in `support-qa.jsonl` is self-assigned, not independently
  annotated. Real judge validation needs labels from someone who did not write
  the dataset.
- `min_detectable_effect` clamps the proportion into [0.1, 0.9] before the
  variance term. At an observed rate of exactly 1.0 the unclamped formula
  reports a detectable effect near 0%, which is technically what the normal
  approximation says and badly misleading.
- Multiple comparisons are flagged but not corrected. With k systems there are
  k(k-1)/2 comparisons and roughly 1 in 20 will look significant by chance; the
  report says so rather than applying a Bonferroni correction that would be its
  own judgement call.
