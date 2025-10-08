"""
Complete APL provided evaluation setup - https://github.com/darpa-scify-apl/scify-evaluation
"""

import functools
import warnings
from typing import List, Literal

import numpy as np
from krippendorff import alpha as kripp_alpha
from rich import print as rprint
from rich.table import Table
from scipy.stats import bootstrap, pearsonr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
)

from eval_types import AnnotatedSystemResult, Score, ScoreStats, problem_from_system_result,assessment_from_system_result


def compute_score_bootstrap(gold_scores, sys_scores, metric, bootstrap_samples, bootstrap_ci) -> ScoreStats:
    """
    Given the scores and a metric, compute stats about the metric using bootstrap sampling.

    *metric* should be a callable that takes in paired lists of predictions and returns a float.
    """
    try:
        observed = metric(gold_scores, sys_scores)
    except ValueError as e:
        rprint(f"[red]Failed to compute {getattr(metric, '__name__', str(metric))}: {e}.[/red]")
        return ScoreStats(
            observed=np.nan,
            bootstrap_samples=None,
            bootstrap_ci=None,
            mean=np.nan,  # Changed from None to np.nan for consistency
            stderr=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
        )

    if not bootstrap_samples:  # for partial evals
        return ScoreStats(
            observed=observed,
            bootstrap_samples=None,
            bootstrap_ci=None,
            mean=None,
            stderr=None,
            ci_low=None,
            ci_high=None,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bootstrap_result = bootstrap(
                (gold_scores, sys_scores),
                metric,
                n_resamples=bootstrap_samples,
                paired=True,
                method="percentile",
                confidence_level=bootstrap_ci,
            )
        except ValueError as e:
            rprint(
                f"[yellow]Failed to compute bootstrap stats for {getattr(metric, '__name__', str(metric))}:"
                f" {e}.[/yellow]"
            )
            return ScoreStats(
                observed=observed,
                bootstrap_samples=bootstrap_samples,
                bootstrap_ci=bootstrap_ci,
                mean=np.nan,
                stderr=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
            )

    return ScoreStats(
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
        observed=observed,
        mean=bootstrap_result.bootstrap_distribution.mean(),
        stderr=bootstrap_result.standard_error,
        ci_low=bootstrap_result.confidence_interval.low,
        ci_high=bootstrap_result.confidence_interval.high,
    )


def safe_quadratic_cohen_kappa(y_true, y_pred, weights="quadratic"):
    """
    Compute quadratic-weighted Cohen's kappa with boundary condition handling.
    """
    # Handle boundary condition: if only one unique class in both arrays
    if len(set(y_true)) == 1 and len(set(y_pred)) == 1:
        if y_true[0] == y_pred[0]:
            return 1.0  # Perfect agreement boundary case

    # Define all possible Likert scale labels for consistent calculation
    all_labels = [-2, -1, 0, 1, 2]  # Full Likert range

    return cohen_kappa_score(y_true, y_pred, weights=weights, labels=all_labels)


def safe_krippendorff_alpha(y1, y2, level_of_measurement="interval"):
    """
    Compute Krippendorff's alpha with boundary condition handling.
    """
    # Handle boundary condition: if only one unique value in both arrays
    if len(set(y1)) == 1 and len(set(y2)) == 1:
        if y1[0] == y2[0]:
            return 1.0  # Perfect agreement boundary case

    try:
        return kripp_alpha(np.vstack([y1, y2]), level_of_measurement=level_of_measurement)
    except ValueError as e:
        rprint(f"[yellow]Error computing krippendorff alpha: {e}[/yellow]")
        return np.nan


def safe_pearson_correlation(y1, y2):
    """
    Compute Pearson correlation with boundary condition handling.
    """
    # Handle boundary condition: if all values are identical
    if len(set(y1)) == 1 and len(set(y2)) == 1:
        if y1[0] == y2[0]:
            return 1.0  # Perfect correlation for identical values
        else:
            return -1.0  # This case shouldn't occur in practice for this evaluation

    # Handle case where one array has no variance
    # example case 2: One array has no variance (this condition)
    # y1 = [-2, -2]  # gold scores (no variance)
    # y2 = [-2, 1]   # system scores (has variance)
    if len(set(y1)) == 1 or len(set(y2)) == 1:
        return 0.0  # No correlation when one variable has no variance

    return pearsonr(y1, y2)[0]


def krippendorff_wrapped(y1, y2):
    try:
        return safe_krippendorff_alpha(y1, y2, level_of_measurement="interval")
    except ValueError as e:
        rprint(f"[yellow]Error computing krippendorff alpha: {e}[/yellow]")
        return np.nan


def evaluate_feasibility(gold_scores, sys_scores, *, bootstrap_samples=10000, bootstrap_ci=0.95) -> Score:
    """
    Compute agreement metrics between two numeric arrays.
    Returns dict of metric_name -> score.

    :param gold_scores: numeric array of gold scores
    :param sys_scores: numeric array of system assessments
    :param bootstrap_samples: The number of bootstrap samples to take to calculate the confidence intervals
    :param bootstrap_ci: The confidence interval to report [0..1].
    :return: dict: metric_name -> numeric score
    """
    # Quadratic-weighted Cohen's kappa
    quadratic_cohen_kappa = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=functools.partial(safe_quadratic_cohen_kappa, weights="quadratic"),
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Krippendorff's alpha (interval)
    krippendorff_alpha = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=krippendorff_wrapped,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Pearson correlation coefficient
    pearson = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=lambda y1, y2: safe_pearson_correlation(y1, y2),
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Mean Absolute Error
    mae = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=mean_absolute_error,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Accuracy
    accuracy = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=accuracy_score,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Balanced Accuracy
    balanced_accuracy = compute_score_bootstrap(
        gold_scores,
        sys_scores,
        metric=balanced_accuracy_score,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    # Accuracy if only looking at the sign
    sign_accuracy = compute_score_bootstrap(
        np.sign(gold_scores),
        np.sign(sys_scores),
        metric=accuracy_score,
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
    )

    return Score(
        bootstrap_samples=bootstrap_samples,
        bootstrap_ci=bootstrap_ci,
        quadratic_cohen_kappa=quadratic_cohen_kappa,
        krippendorff_alpha=krippendorff_alpha,
        pearson=pearson,
        mean_absolute_error=mae,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        sign_accuracy=sign_accuracy,
    )


def evaluate_timing(system_results: List[AnnotatedSystemResult]):
    times = []
    no_times = []
    for r in system_results:
        if "duration" in r.keys():
            if r["duration"]:
                times.append(r["duration"])
            else:
                no_times.append(assessment_from_system_result(r).problem_id)
        elif "wall_clock_time" in r.keys():
            if r["wall_clock_time"]:
                times.append(r["wall_clock_time"])
            else:
                no_times.append(assessment_from_system_result(r).problem_id)
    if not times:
        rprint("[bold yellow]No timings found for submission[/bold yellow]")
        return {
            "min": 0,  # convert to minutes
            "mean": 0,  # convert to minutes
            "max": 0,  # convert to minutes
            "std_dev": 0,  # population std dev
        }
    if no_times:
        rprint(f"[bold yellow]{len(no_times)} submissions are missing timings:[/bold yellow]\n{no_times=}")
    
    return {
        "min": float(np.min(times) / 60),  # convert to minutes
        "mean": float(np.mean(times) / 60),  # convert to minutes
        "max": float(np.max(times) / 60),  # convert to minutes
        "std_dev": float(np.std(times) / 60),  # population std dev
    }


def print_evaluation_summary(metrics: Score, time_metrics: dict, *, fmt: Literal["simple", "table"] = "simple"):
    """Print a formatted summary of evaluation results"""
    rprint("[bold blue]Evaluation Summary:[/bold blue]")

    # list of name, attr to display
    metrics_to_print = [
        ("Quadratic Cohen Kappa", "quadratic_cohen_kappa"),
        ("Krippendorff Alpha", "krippendorff_alpha"),
        ("Pearson Correlation", "pearson"),
        ("Mean Absolute Error", "mean_absolute_error"),
        ("Accuracy", "accuracy"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("Sign Accuracy", "sign_accuracy"),
    ]

    # TABLE
    if fmt == "table":
        table = Table()
        table.add_column("Metric", style="bold cyan")
        table.add_column("Observed")
        table.add_column("BS Mean")
        table.add_column("Std. Err")
        table.add_column(f"{metrics.bootstrap_ci:.0%} CI")

        for name, attr in metrics_to_print:
            score_stat = getattr(metrics, attr)
            table.add_row(
                name,
                f"{score_stat.observed:.4f}",
                f"{score_stat.mean:.4f}",
                f"±{score_stat.stderr:.4f}",
                f"[{score_stat.ci_low:.4f}, {score_stat.ci_high:.4f}]",
            )
        rprint(table)
    # SIMPLE
    else:
        for name, attr in metrics_to_print:
            score_stat = getattr(metrics, attr)
            if not score_stat.bootstrap_samples:
                out = f"{score_stat.observed:.4f}"
            else:
                out = (
                    f"{score_stat.observed:.4f}\n"
                    f"    BS Mean: {score_stat.mean:.4f} +/- {score_stat.stderr:.4f}\n"
                    f"    {score_stat.bootstrap_ci:.0%} CI: [{score_stat.ci_low:.4f}, {score_stat.ci_high:.4f}]"
                )
            rprint(f"{name}: {out}")
    rprint(
        f"Mean Wall Clock Time: {time_metrics.get('mean', float('nan')):.2f} ±"
        f" {time_metrics.get('std_dev', float('nan')):.2f}mins"
    )
