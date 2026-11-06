"""Tests for the statistics.

This is the module the project's claims rest on, so it is tested against known
closed-form values rather than against its own output. If these are wrong, every
number the harness reports is wrong in a way nobody would notice.
"""

from __future__ import annotations

import pytest

from evalforge.stats import (
    Interval,
    agreement,
    bootstrap_interval,
    mcnemar_exact,
    mean,
    min_detectable_effect,
    paired_bootstrap,
    percentile,
    required_n,
    run_variance,
    stdev,
    wilson_interval,
)


# --- Wilson --------------------------------------------------------------


def test_wilson_stays_inside_zero_one_at_the_boundary():
    """The whole reason for using Wilson over the normal approximation: at
    19/20 the normal interval runs past 1.0."""
    interval = wilson_interval(20, 20)
    assert interval.high <= 1.0
    assert interval.low > 0.5
    interval = wilson_interval(0, 20)
    assert interval.low >= 0.0


@pytest.mark.parametrize(
    "successes,n,low,high",
    [
        # Textbook Wilson score intervals (no continuity correction), 95%.
        # Note these are NOT the Clopper-Pearson values, which are wider --
        # the first version of this test asserted 0.751 for 19/20, which is
        # Clopper-Pearson's lower bound, and failed against a correct Wilson.
        (19, 20, 0.7639, 0.9911),
        (8, 10, 0.4902, 0.9433),
        (50, 100, 0.4038, 0.5962),
    ],
)
def test_wilson_matches_textbook_values(successes, n, low, high):
    interval = wilson_interval(successes, n)
    assert interval.point == pytest.approx(successes / n)
    assert interval.low == pytest.approx(low, abs=0.001)
    assert interval.high == pytest.approx(high, abs=0.001)


def test_wilson_interval_narrows_with_more_data():
    small = wilson_interval(8, 10)
    large = wilson_interval(800, 1000)
    assert large.width < small.width / 5


def test_wilson_handles_zero_items():
    interval = wilson_interval(0, 0)
    assert interval.low == 0.0 and interval.high == 1.0


# --- bootstrap -----------------------------------------------------------


def test_bootstrap_is_deterministic_given_a_seed():
    """A report whose confidence intervals move every run is not auditable."""
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    a = bootstrap_interval(values, iterations=800)
    b = bootstrap_interval(values, iterations=800)
    assert (a.low, a.high) == (b.low, b.high)


def test_bootstrap_brackets_the_point_estimate():
    values = [0.9, 0.8, 1.0, 0.7, 0.85, 0.95, 0.6, 0.75]
    interval = bootstrap_interval(values, iterations=1500)
    assert interval.low <= interval.point <= interval.high


def test_bootstrap_on_identical_values_has_zero_width():
    interval = bootstrap_interval([0.5] * 10, iterations=300)
    assert interval.width == pytest.approx(0.0)


def test_bootstrap_single_value_is_degenerate_not_an_error():
    interval = bootstrap_interval([0.7])
    assert interval.point == interval.low == interval.high == 0.7


def test_bootstrap_empty_is_safe():
    assert bootstrap_interval([]).point == 0.0


# --- interval helpers ----------------------------------------------------


def test_interval_overlap_and_containment():
    a = Interval(0.80, 0.70, 0.90)
    b = Interval(0.85, 0.78, 0.92)
    c = Interval(0.50, 0.40, 0.60)
    assert a.overlaps(b) and not a.overlaps(c)
    assert a.contains(0.75) and not a.contains(0.95)
    assert a.margin == pytest.approx(0.10)


# --- paired comparison ---------------------------------------------------


def test_paired_comparison_detects_a_large_real_difference():
    a = [0.0] * 20 + [1.0] * 5
    b = [1.0] * 25
    comparison = paired_bootstrap(a, b, iterations=2000)
    assert comparison.delta > 0.5
    assert comparison.significant
    assert "B is better" in comparison.verdict


def test_paired_comparison_calls_noise_not_significant():
    """Two systems differing on a single item out of 40 must not be reported as
    a difference. This is the specific false conclusion the project exists to
    prevent."""
    a = [1.0] * 32 + [0.0] * 8
    b = [1.0] * 33 + [0.0] * 7
    comparison = paired_bootstrap(a, b, iterations=2000)
    assert not comparison.significant
    assert comparison.verdict == "no detectable difference"


def test_paired_comparison_is_tighter_than_unpaired():
    """Pairing removes item-difficulty variance. The paired interval on the
    difference must be narrower than the sum of the two individual intervals."""
    a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0] * 3
    b = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0] * 3
    paired = paired_bootstrap(a, b, iterations=2000)
    independent_width = bootstrap_interval(a).width + bootstrap_interval(b).width
    assert paired.delta_interval.width < independent_width


def test_paired_comparison_counts_discordant_pairs():
    a = [1.0, 1.0, 0.0, 0.0]
    b = [1.0, 0.0, 1.0, 0.0]
    comparison = paired_bootstrap(a, b, iterations=400)
    assert comparison.both == 1
    assert comparison.a_only == 1
    assert comparison.b_only == 1
    assert comparison.neither == 1


def test_paired_comparison_requires_equal_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        paired_bootstrap([1.0, 0.0], [1.0])


# --- McNemar -------------------------------------------------------------


def test_mcnemar_with_no_disagreement_is_one():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_is_symmetric():
    assert mcnemar_exact(2, 8) == pytest.approx(mcnemar_exact(8, 2))


def test_mcnemar_known_values():
    # all 10 discordant pairs favour one side: 2 * (1/2)^10
    assert mcnemar_exact(0, 10) == pytest.approx(2 * 0.5**10, abs=1e-6)
    # an even split is maximally unsurprising
    assert mcnemar_exact(5, 5) == pytest.approx(1.0, abs=1e-6)


def test_mcnemar_ignores_concordant_pairs():
    """Items both systems got right carry no information about which is better.
    This is why a 2-vs-8 split can be significant on 10 discordant pairs while
    a 2-point aggregate difference on 1000 items is not."""
    assert mcnemar_exact(1, 9) < 0.05


# --- power ---------------------------------------------------------------


def test_min_detectable_effect_shrinks_with_sample_size():
    assert min_detectable_effect(20) > min_detectable_effect(100) > min_detectable_effect(1000)


def test_small_eval_sets_cannot_detect_small_changes():
    """The headline fact: a 40-item eval cannot see a 5-point change, so
    comparing 0.84 to 0.87 on 40 items is not evidence."""
    assert min_detectable_effect(40) > 0.15


def test_required_n_round_trips_with_min_detectable_effect():
    n = required_n(0.10, baseline=0.8)
    assert min_detectable_effect(n, baseline=0.8) == pytest.approx(0.10, abs=0.015)


def test_detecting_three_points_needs_over_a_thousand_items():
    assert required_n(0.03, baseline=0.8) > 1000


def test_required_n_handles_zero_effect():
    assert required_n(0.0) > 10**6


# --- variance ------------------------------------------------------------


def test_identical_runs_are_stable():
    variance = run_variance([0.84, 0.84, 0.84])
    assert variance.spread == 0.0
    assert variance.stable


def test_noisy_runs_are_flagged_unstable():
    variance = run_variance([0.80, 0.86, 0.83, 0.89])
    assert not variance.stable
    assert "UNSTABLE" in variance.summary()


def test_variance_reports_sd_and_spread():
    variance = run_variance([0.8, 0.9])
    assert variance.spread == pytest.approx(0.1)
    assert variance.stdev > 0


# --- judge validation ----------------------------------------------------


def test_kappa_exposes_a_judge_that_always_says_pass():
    """Raw agreement is misleading when the labels are imbalanced. A judge that
    passes everything scores 0.9 agreement on a 90%-pass set and has learned
    nothing; kappa shows that as ~0."""
    human = [1.0] * 18 + [0.0] * 2
    lazy_judge = [1.0] * 20
    result = agreement(human, lazy_judge)
    assert result["agreement"] == pytest.approx(0.9)
    assert result["kappa"] < 0.05


def test_kappa_rewards_a_real_judge():
    human = [1.0] * 10 + [0.0] * 10
    good_judge = [1.0] * 9 + [0.0] + [0.0] * 9 + [1.0]
    result = agreement(human, good_judge)
    assert result["kappa"] > 0.7


def test_agreement_handles_mismatched_input():
    assert agreement([1.0], [1.0, 0.0])["n"] == 0


# --- helpers -------------------------------------------------------------


def test_percentile_edges():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5
    assert percentile(values, 100) == 10
    assert percentile([], 50) == 0.0


def test_mean_and_stdev():
    assert mean([1, 2, 3]) == pytest.approx(2.0)
    assert stdev([2, 2, 2]) == 0.0
    assert stdev([1]) == 0.0
