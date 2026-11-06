"""Reports: terminal, markdown and a self-contained HTML page.

Every reported figure carries its interval. That is the whole editorial stance
of this project: a bare "0.87" in a report invites a decision the data does not
support, so the format makes the uncertainty impossible to skip.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .runner import RunResult
from .stats import Comparison, Variance


def print_run(run: RunResult, confidence: float = 0.95) -> None:
    s = run.summary(confidence)
    pr_low, pr_high = s["pass_rate_ci"]

    print("\n" + "=" * 72)
    print(f"  {run.name}  ·  {s['dataset']}  ·  target={s['target']}")
    print("=" * 72)
    print(f"  items              {s['n']}")
    print(f"  pass rate          {s['pass_rate']:.3f}  [{pr_low:.3f}, {pr_high:.3f}]  "
          f"(±{s['pass_rate_margin']:.3f})")
    ms_low, ms_high = s["mean_score_ci"]
    print(f"  mean score         {s['mean_score']:.3f}  [{ms_low:.3f}, {ms_high:.3f}]")
    print(f"  detectable effect  {s['min_detectable_effect']:.1%}  "
          "<- smaller changes are noise on this set")
    print(f"  errors             {s['errors']}")
    print(f"  cost               ${s['total_cost_usd']:.4f} total, "
          f"${s['cost_per_item_usd']:.6f}/item")
    lat = s["latency_ms"]
    print(f"  latency            p50 {lat['p50']:.0f}ms · p95 {lat['p95']:.0f}ms · "
          f"p99 {lat['p99']:.0f}ms")
    print(f"  cached             {s['cached_items']}/{s['n']}")
    print(f"  wall               {s['wall_seconds']}s")

    print("\n  by category")
    print("  " + "-" * 68)
    for cat, vals in s["by_category"].items():
        print(
            f"  {cat:<18} n={vals['n']:<4} pass={vals['pass_rate']:.3f} "
            f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}]"
        )

    failures = run.failures()
    if failures:
        print(f"\n  failures ({len([r for r in run.results if not r.passed])})")
        print("  " + "-" * 68)
        for f in failures[:12]:
            print(f"  [{f['id']}] {f['input']}")
            print(f"       -> {f['detail']}")


def print_comparison(comparisons: Sequence[Comparison]) -> None:
    if not comparisons:
        return
    print("\n" + "=" * 72)
    print("  PAIRED COMPARISONS")
    print("=" * 72)
    for c in comparisons:
        print(f"  {c.summary()}")
        print(f"      verdict: {c.verdict}")
        if c.a_only or c.b_only:
            print(
                f"      discordant: {c.name_a}-only {c.a_only}, "
                f"{c.name_b}-only {c.b_only} (both right {c.both}, both wrong {c.neither})"
            )
    if len(comparisons) > 1:
        print(
            f"\n  note: {len(comparisons)} comparisons at 95% confidence — expect "
            "roughly 1 in 20 to look significant by chance. Treat a single "
            "surprising pair with suspicion."
        )


def print_variance(variance: Variance) -> None:
    print("\n" + "=" * 72)
    print("  RUN-TO-RUN VARIANCE")
    print("=" * 72)
    print(f"  {variance.summary()}")
    print(f"  scores: {[round(s, 3) for s in variance.scores]}")
    if not variance.stable:
        print(
            f"\n  This eval varies by {variance.spread:.1%} across identical runs. "
            "Any improvement smaller than that is not measurable here — reduce "
            "temperature, fix seeds, or grow the dataset before comparing prompts."
        )


def markdown_report(
    runs: Sequence[RunResult],
    comparisons: Sequence[Comparison] = (),
    variance: Variance | None = None,
    confidence: float = 0.95,
) -> str:
    lines = ["# Evaluation report", ""]

    lines += ["## Runs", "", "| run | n | pass rate (95% CI) | mean score | $/item | p95 ms |", "| --- | --- | --- | --- | --- | --- |"]
    for run in runs:
        s = run.summary(confidence)
        low, high = s["pass_rate_ci"]
        lines.append(
            f"| {s['name']} | {s['n']} | **{s['pass_rate']:.3f}** [{low:.3f}, {high:.3f}] | "
            f"{s['mean_score']:.3f} | ${s['cost_per_item_usd']:.6f} | "
            f"{s['latency_ms']['p95']:.0f} |"
        )

    if runs:
        mde = runs[0].summary(confidence)["min_detectable_effect"]
        lines += [
            "",
            f"> This dataset ({runs[0].n} items) can detect a difference of about "
            f"**{mde:.0%}** or larger. Differences smaller than that are "
            "indistinguishable from noise, however suggestive the point estimates look.",
        ]

    if comparisons:
        lines += ["", "## Paired comparisons", "", "| A | B | delta (95% CI) | verdict |", "| --- | --- | --- | --- |"]
        for c in comparisons:
            lines.append(
                f"| {c.name_a} ({c.mean_a:.3f}) | {c.name_b} ({c.mean_b:.3f}) | "
                f"{c.delta:+.3f} [{c.delta_interval.low:+.3f}, {c.delta_interval.high:+.3f}] | "
                f"{c.verdict} |"
            )

    if variance:
        lines += [
            "",
            "## Run-to-run variance",
            "",
            f"- runs: {variance.runs}",
            f"- mean: {variance.mean:.3f}",
            f"- spread: {variance.spread:.3f} ({'stable' if variance.stable else '**unstable**'})",
        ]

    for run in runs:
        s = run.summary(confidence)
        lines += ["", f"## {s['name']} — by category", "", "| category | n | pass rate (95% CI) |", "| --- | --- | --- |"]
        for cat, vals in s["by_category"].items():
            lines.append(
                f"| {cat} | {vals['n']} | {vals['pass_rate']:.3f} "
                f"[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}] |"
            )

        failures = run.failures(limit=15)
        if failures:
            lines += ["", f"### {s['name']} — failures", ""]
            for f in failures:
                lines.append(f"- **{f['id']}** ({f['category']}): {f['detail']}")

    return "\n".join(lines) + "\n"


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1a1a1a; --bg:#fbfbfa; --muted:#6b6b6b;
           --line:#e3e3e0; --ok:#1a7f4b; --bad:#b4342a; --accent:#2f5d8f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e8e8e6; --bg:#17181a; --muted:#9a9a97; --line:#2e3034;
             --ok:#4ec380; --bad:#ec7b70; --accent:#79a8dd; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:2.5rem 1.5rem; background:var(--bg); color:var(--fg);
          font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  main {{ max-width:960px; margin:0 auto; }}
  h1 {{ font-size:1.6rem; margin:0 0 .25rem; letter-spacing:-.01em; }}
  h2 {{ font-size:1.15rem; margin:2.5rem 0 .75rem; padding-bottom:.4rem;
        border-bottom:1px solid var(--line); }}
  .sub {{ color:var(--muted); margin:0 0 2rem; font-size:.9rem; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.75rem; }}
  .card {{ border:1px solid var(--line); border-radius:10px; padding:.9rem 1rem; }}
  .card .label {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
                  color:var(--muted); }}
  .card .value {{ font-size:1.5rem; font-weight:600; margin:.3rem 0 .1rem;
                  font-variant-numeric:tabular-nums; }}
  .card .ci {{ font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums; }}
  table {{ width:100%; border-collapse:collapse; font-size:.9rem; }}
  th,td {{ text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line);
           font-variant-numeric:tabular-nums; }}
  th {{ font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }}
  .bar {{ position:relative; height:7px; background:var(--line); border-radius:4px;
          overflow:hidden; min-width:90px; }}
  .bar > span {{ position:absolute; inset:0 auto 0 0; background:var(--accent); }}
  .ok {{ color:var(--ok); }} .bad {{ color:var(--bad); }}
  .note {{ border-left:3px solid var(--accent); background:color-mix(in srgb,var(--accent) 7%,transparent);
           padding:.75rem 1rem; border-radius:0 8px 8px 0; margin:1rem 0; font-size:.9rem; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em; }}
  details {{ margin:.4rem 0; }} summary {{ cursor:pointer; }}
</style>
</head>
<body><main>
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
{body}
</main></body>
</html>
"""


def html_report(
    runs: Sequence[RunResult],
    comparisons: Sequence[Comparison] = (),
    variance: Variance | None = None,
    confidence: float = 0.95,
    title: str = "Evaluation report",
) -> str:
    def esc(text: Any) -> str:
        return html.escape(str(text))

    parts: list[str] = []

    if runs:
        s = runs[0].summary(confidence)
        low, high = s["pass_rate_ci"]
        ms_low, ms_high = s["mean_score_ci"]
        parts.append(
            '<div class="cards">'
            f'<div class="card"><div class="label">Pass rate</div>'
            f'<div class="value">{s["pass_rate"]:.1%}</div>'
            f'<div class="ci">95% CI {low:.3f} – {high:.3f}</div></div>'
            f'<div class="card"><div class="label">Mean score</div>'
            f'<div class="value">{s["mean_score"]:.3f}</div>'
            f'<div class="ci">95% CI {ms_low:.3f} – {ms_high:.3f}</div></div>'
            f'<div class="card"><div class="label">Items</div>'
            f'<div class="value">{s["n"]}</div>'
            f'<div class="ci">{s["errors"]} target error(s)</div></div>'
            f'<div class="card"><div class="label">Cost / item</div>'
            f'<div class="value">${s["cost_per_item_usd"]:.5f}</div>'
            f'<div class="ci">${s["total_cost_usd"]:.4f} total</div></div>'
            f'<div class="card"><div class="label">p95 latency</div>'
            f'<div class="value">{s["latency_ms"]["p95"]:.0f}<span class="ci"> ms</span></div>'
            f'<div class="ci">p50 {s["latency_ms"]["p50"]:.0f} ms</div></div>'
            "</div>"
        )
        parts.append(
            '<div class="note">This dataset of '
            f'{s["n"]} items can detect a difference of about '
            f'<strong>{s["min_detectable_effect"]:.0%}</strong> or larger. '
            "Anything smaller is indistinguishable from run-to-run noise, "
            "regardless of how the point estimates compare.</div>"
        )

    if len(runs) > 1:
        parts.append("<h2>Runs</h2><table><tr><th>Run</th><th>Pass rate</th><th></th>"
                     "<th>95% CI</th><th>$/item</th><th>p95 ms</th></tr>")
        for run in runs:
            s = run.summary(confidence)
            low, high = s["pass_rate_ci"]
            parts.append(
                f"<tr><td>{esc(s['name'])}</td>"
                f"<td>{s['pass_rate']:.3f}</td>"
                f'<td><div class="bar"><span style="width:{s["pass_rate"]*100:.1f}%"></span></div></td>'
                f"<td>{low:.3f} – {high:.3f}</td>"
                f"<td>${s['cost_per_item_usd']:.6f}</td>"
                f"<td>{s['latency_ms']['p95']:.0f}</td></tr>"
            )
        parts.append("</table>")

    if comparisons:
        parts.append("<h2>Paired comparisons</h2><table>"
                     "<tr><th>A</th><th>B</th><th>Delta</th><th>95% CI</th><th>Verdict</th></tr>")
        for c in comparisons:
            cls = "ok" if c.significant else ""
            parts.append(
                f"<tr><td>{esc(c.name_a)} ({c.mean_a:.3f})</td>"
                f"<td>{esc(c.name_b)} ({c.mean_b:.3f})</td>"
                f"<td>{c.delta:+.3f}</td>"
                f"<td>{c.delta_interval.low:+.3f} – {c.delta_interval.high:+.3f}</td>"
                f'<td class="{cls}">{esc(c.verdict)}</td></tr>'
            )
        parts.append("</table>")
        if len(comparisons) > 1:
            parts.append(
                f'<div class="note">{len(comparisons)} comparisons at 95% confidence: '
                "roughly 1 in 20 will look significant by chance. A single surprising "
                "pair is not a finding.</div>"
            )

    if variance:
        cls = "ok" if variance.stable else "bad"
        parts.append(
            f"<h2>Run-to-run variance</h2><p>{variance.runs} identical runs, "
            f"mean {variance.mean:.3f}, spread "
            f'<span class="{cls}">{variance.spread:.3f}</span> '
            f"({'stable' if variance.stable else 'unstable'}).</p>"
        )
        if not variance.stable:
            parts.append(
                f'<div class="note">Identical runs differ by {variance.spread:.1%}. '
                "No improvement smaller than that is measurable here.</div>"
            )

    for run in runs:
        s = run.summary(confidence)
        parts.append(f"<h2>{esc(s['name'])} — by category</h2><table>"
                     "<tr><th>Category</th><th>n</th><th>Pass rate</th><th></th><th>95% CI</th></tr>")
        for cat, vals in s["by_category"].items():
            parts.append(
                f"<tr><td>{esc(cat)}</td><td>{vals['n']}</td>"
                f"<td>{vals['pass_rate']:.3f}</td>"
                f'<td><div class="bar"><span style="width:{vals["pass_rate"]*100:.1f}%"></span></div></td>'
                f"<td>{vals['ci_low']:.3f} – {vals['ci_high']:.3f}</td></tr>"
            )
        parts.append("</table>")

        failures = run.failures(limit=25)
        if failures:
            parts.append(f"<h2>{esc(s['name'])} — failures ({len(failures)})</h2>")
            for f in failures:
                parts.append(
                    f"<details><summary><code>{esc(f['id'])}</code> "
                    f"{esc(f['input'])}</summary>"
                    f"<p><strong>Why:</strong> {esc(f['detail'])}<br>"
                    f"<strong>Output:</strong> {esc(f['output'])}</p></details>"
                )

    subtitle = (
        f"{runs[0].dataset} · dataset fingerprint {runs[0].dataset_fingerprint[:12]}"
        if runs
        else ""
    )
    return _HTML_TEMPLATE.format(title=esc(title), subtitle=esc(subtitle), body="\n".join(parts))


def write_reports(
    runs: Sequence[RunResult],
    directory: str | Path,
    comparisons: Sequence[Comparison] = (),
    variance: Variance | None = None,
    confidence: float = 0.95,
) -> dict[str, Path]:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "json": out / "report.json",
        "markdown": out / "report.md",
        "html": out / "report.html",
    }
    paths["json"].write_text(
        json.dumps(
            {
                "runs": [r.summary(confidence) for r in runs],
                "rows": [[x.as_row() for x in r.results] for r in runs],
                "comparisons": [
                    {
                        "a": c.name_a,
                        "b": c.name_b,
                        "delta": c.delta,
                        "ci": [c.delta_interval.low, c.delta_interval.high],
                        "significant": c.significant,
                        "p_value": c.p_value,
                    }
                    for c in comparisons
                ],
                "variance": {
                    "runs": variance.runs,
                    "mean": variance.mean,
                    "spread": variance.spread,
                    "stable": variance.stable,
                }
                if variance
                else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        markdown_report(runs, comparisons, variance, confidence), encoding="utf-8"
    )
    paths["html"].write_text(
        html_report(runs, comparisons, variance, confidence), encoding="utf-8"
    )
    return paths
