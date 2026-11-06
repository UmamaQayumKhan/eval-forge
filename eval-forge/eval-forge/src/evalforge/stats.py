"""Statistics for LLM evaluation.

This module is the reason the project exists. The standard practice is to run an
eval once, get 0.84, change a prompt, get 0.87, and ship it. That comparison is
usually meaningless:

* On a 50-item set, the 95% confidence interval on 0.84 is roughly ±0.10. A move
  from 0.84 to 0.87 is inside the noise.
* LLM outputs vary between identical runs. Some of that 0.03 is sampling, not
  improvement.
* To detect a real 3-point difference on a binary metric at 80% power you need
  on the order of a thousand items, which nobody has.

So every number this harness reports carries a confidence interval, comparisons
between two systems are **paired** on the same items (which removes
item-difficulty variance and is far more sensitive than comparing two
independent means), and the regression gate is explicitly told how small a
change it is capable of detecting.

Everything is implemented with the standard library: bootstrapping needs a RNG
and arithmetic, and depending on scipy for this would be the only heavy
dependency in the project.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def margin(self) -> float:
        return self.width / 2

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def overlaps(self, other: "Interval") -> bool:
        return not (self.high < other.low or other.high < self.low)

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile. Used for latency and for bootstrap quantiles."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]


# --------------------------------------------------------------------------
# confidence intervals
# --------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because eval metrics live near 0 and
    1, where the normal interval famously returns bounds outside [0, 1] and has
    poor coverage. At 19/20 the normal interval runs past 1.0; Wilson does not.
    """
    if n == 0:
        return Interval(0.0, 0.0, 1.0, confidence)
    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Interval(p, max(0.0, centre - spread), min(1.0, centre + spread), confidence)


def bootstrap_interval(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> Interval:
    """Percentile bootstrap.

    Works for any statistic, not just a proportion, so the same function gives
    an interval for mean score, for a judge rating out of 5, and for p95
    latency. The seed is fixed so a report is reproducible: an eval whose
    confidence intervals move every run is not auditable.
    """
    if not values:
        return Interval(0.0, 0.0, 0.0, confidence)
    if len(values) == 1:
        v = float(values[0])
        return Interval(v, v, v, confidence)

    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(iterations):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic(resample))

    alpha = (1 - confidence) / 2
    return Interval(
        point=statistic(values),
        low=percentile(samples, alpha * 100),
        high=percentile(samples, (1 - alpha) * 100),
        confidence=confidence,
    )


def _z_for(confidence: float) -> float:
    table = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    if confidence in table:
        return table[confidence]
    # Acklam-style rational approximation to the inverse normal CDF, accurate
    # to ~1e-4 which is far beyond what matters here.
    p = 1 - (1 - confidence) / 2
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# --------------------------------------------------------------------------
# comparing two systems
# --------------------------------------------------------------------------


@dataclass
class Comparison:
    """A paired comparison of two systems on the same items."""

    name_a: str
    name_b: str
    n: int
    mean_a: float
    mean_b: float
    delta: float
    delta_interval: Interval
    # McNemar counts: items where exactly one system was right.
    a_only: int = 0
    b_only: int = 0
    both: int = 0
    neither: int = 0
    p_value: float | None = None

    @property
    def significant(self) -> bool:
        """A difference is called real only when the interval on the difference
        excludes zero. Comparing two point estimates is not evidence."""
        return not self.delta_interval.contains(0.0)

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "no detectable difference"
        better = self.name_b if self.delta > 0 else self.name_a
        return f"{better} is better"

    def summary(self) -> str:
        sig = "significant" if self.significant else "NOT significant"
        p = f", p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"{self.name_a} {self.mean_a:.3f} vs {self.name_b} {self.mean_b:.3f} "
            f"| delta {self.delta:+.3f} {self.delta_interval} | {sig}{p}"
        )


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    name_a: str = "A",
    name_b: str = "B",
    iterations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> Comparison:
    """Bootstrap the *difference*, resampling items not scores.

    Pairing is the whole point. Comparing two independent means carries the
    variance of item difficulty twice; resampling item indices and taking the
    difference within each resample cancels it, which typically shrinks the
    interval enough to detect a few points of difference on a 50-item set rather
    than needing hundreds.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths, got {len(a)} and {len(b)}")
    if not a:
        return Comparison(name_a, name_b, 0, 0.0, 0.0, 0.0, Interval(0, 0, 0, confidence))

    rng = random.Random(seed)
    n = len(a)
    deltas: list[float] = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(mean([b[i] for i in idx]) - mean([a[i] for i in idx]))

    alpha = (1 - confidence) / 2
    observed = mean(b) - mean(a)
    interval = Interval(
        point=observed,
        low=percentile(deltas, alpha * 100),
        high=percentile(deltas, (1 - alpha) * 100),
        confidence=confidence,
    )

    # McNemar counts, meaningful when the scores are binary.
    binary = all(v in (0.0, 1.0) for v in list(a) + list(b))
    a_only = b_only = both = neither = 0
    p_value = None
    if binary:
        for x, y in zip(a, b):
            if x and y:
                both += 1
            elif x and not y:
                a_only += 1
            elif y and not x:
                b_only += 1
            else:
                neither += 1
        p_value = mcnemar_exact(a_only, b_only)

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        n=n,
        mean_a=mean(a),
        mean_b=mean(b),
        delta=observed,
        delta_interval=interval,
        a_only=a_only,
        b_only=b_only,
        both=both,
        neither=neither,
        p_value=p_value,
    )


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar test via the binomial distribution.

    Only the discordant pairs carry information: items both systems got right,
    or both got wrong, say nothing about which is better. The exact binomial is
    used rather than the chi-squared approximation because discordant counts in
    an LLM eval are routinely under 10, where the approximation is unreliable.
    """
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


# --------------------------------------------------------------------------
# power: what can this eval set actually detect?
# --------------------------------------------------------------------------


def min_detectable_effect(
    n: int, baseline: float = 0.8, *, confidence: float = 0.95, power: float = 0.80
) -> float:
    """Smallest difference in a proportion this many items can detect.

    Reported alongside every gate so a passing run is honest about its limits.
    Without it, "no regression detected" gets read as "no regression", when on a
    40-item set it means "no regression larger than about 20 points".
    """
    if n <= 1:
        return 1.0
    z_a = _z_for(confidence)
    z_b = _z_for(1 - 2 * (1 - power))  # one-sided z for the desired power
    # The proportion is clamped away from the extremes before it enters the
    # variance term. At an observed rate of exactly 1.0, p*(1-p) is zero and the
    # formula reports a detectable effect near 0% -- technically what the normal
    # approximation says, and badly misleading: it told a 3-item eval it could
    # detect a 0.2% change. Clamping keeps the estimate conservative, which is
    # the correct direction to err for a number whose whole job is honesty
    # about what the eval cannot see.
    p = min(max(baseline, 0.1), 0.9)
    return min(1.0, (z_a + z_b) * math.sqrt(2 * p * (1 - p) / n))


def required_n(effect: float, baseline: float = 0.8, *, confidence: float = 0.95, power: float = 0.80) -> int:
    """Items needed to detect `effect`. The number that explains why most
    published eval comparisons are not evidence."""
    if effect <= 0:
        return 10**9
    z_a = _z_for(confidence)
    z_b = _z_for(1 - 2 * (1 - power))
    p = min(max(baseline, 0.1), 0.9)  # see min_detectable_effect
    return math.ceil(2 * p * (1 - p) * ((z_a + z_b) / effect) ** 2)


# --------------------------------------------------------------------------
# run-to-run variance
# --------------------------------------------------------------------------


@dataclass
class Variance:
    runs: int
    scores: list[float]
    mean: float
    stdev: float
    spread: float
    interval: Interval

    @property
    def stable(self) -> bool:
        """Under 2 points of run-to-run spread, a single run is usable."""
        return self.spread < 0.02

    def summary(self) -> str:
        return (
            f"{self.runs} runs: mean {self.mean:.3f} sd {self.stdev:.4f} "
            f"spread {self.spread:.3f} {'(stable)' if self.stable else '(UNSTABLE)'}"
        )


def run_variance(scores: Sequence[float], confidence: float = 0.95) -> Variance:
    """Characterise variance across repeated identical runs.

    Run this before trusting any single-run number. If the spread across
    identical runs is 4 points, a 3-point "improvement" from a prompt change is
    not an improvement.
    """
    vals = [float(s) for s in scores]
    return Variance(
        runs=len(vals),
        scores=vals,
        mean=mean(vals),
        stdev=stdev(vals),
        spread=(max(vals) - min(vals)) if vals else 0.0,
        interval=bootstrap_interval(vals, confidence=confidence) if len(vals) > 1 else Interval(
            mean(vals), mean(vals), mean(vals), confidence
        ),
    )


def agreement(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Raw agreement and Cohen's kappa between two binary scorers.

    Used to validate an LLM judge against human labels. Raw agreement alone is
    misleading: when 90% of items pass, a judge that says "pass" every time
    scores 0.90 agreement and has learned nothing. Kappa corrects for chance and
    exposes that as ~0.
    """
    if len(a) != len(b) or not a:
        return {"agreement": 0.0, "kappa": 0.0, "n": 0}
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1, pb1 = mean(a), mean(b)
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1 - expected)
    return {"agreement": round(observed, 4), "kappa": round(kappa, 4), "n": n}
