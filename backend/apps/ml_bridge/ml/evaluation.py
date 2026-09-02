"""Operational evaluation protocol for landslide early warning system.

This module implements evaluation metrics and protocols appropriate for
an operational early warning system, not just academic ML metrics.

Key principles:
1. Lead-time awareness: Metrics computed per forecast horizon
2. Actionable thresholds: Precision/recall at decision thresholds
3. Cost-sensitive: False alarms vs missed events have different costs
4. Temporal consistency: Metrics stable across time periods
5. Spatial relevance: Metrics per region/zone

Metrics:
- Standard: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Operational: Precision@k alerts, Recall@lead_time, Alert rate
- Cost-sensitive: Expected cost per warning, Cost-weighted F1
- Temporal: Rolling window metrics, Seasonal breakdown
- Spatial: Per-zone metrics, Hotspot detection rate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for operational evaluation."""

    # Lead times to evaluate (days)
    lead_times: list[int] = field(default_factory=lambda: [1, 3, 7])

    # Probability thresholds for alerting
    alert_thresholds: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # Cost matrix (relative costs)
    # cost_fn: cost of false negative (missed landslide)
    # cost_fp: cost of false positive (false alarm)
    cost_fn: float = 100.0  # Missing a landslide is very costly
    cost_fp: float = 1.0  # False alarm has operational cost

    # Minimum samples for reliable metrics
    min_samples_per_zone: int = 30

    # Rolling window for temporal stability
    rolling_window_days: int = 90

    # Seasons (for seasonal breakdown)
    seasons: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "pre_monsoon": (3, 5),  # Mar-May
            "monsoon": (6, 9),  # Jun-Sep
            "post_monsoon": (10, 11),  # Oct-Nov
            "winter": (12, 2),  # Dec-Feb
        }
    )


@dataclass
class EvaluationResult:
    """Results from operational evaluation."""

    overall_metrics: dict
    per_lead_time: dict
    per_threshold: dict
    per_zone: dict
    per_season: dict
    temporal_stability: dict
    cost_analysis: dict
    recommendations: list[str]


def compute_cost_weighted_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """Compute cost-weighted metrics for a given threshold.

    Expected cost = cost_fn * P(FN) + cost_fp * P(FP)
    """
    if len(y_prob) == 0:
        return {}

    # Find optimal threshold minimizing expected cost
    thresholds = np.linspace(0, 1, 101)
    best_cost = float("inf")
    best_threshold = 0.5
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        expected_cost = cost_fn * fn_rate + cost_fp * fp_rate

        if expected_cost < best_cost:
            best_cost = expected_cost
            best_threshold = t
            best_metrics = {
                "threshold": t,
                "expected_cost": expected_cost,
                "fn_rate": fn_rate,
                "fp_rate": fp_rate,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }

    best_metrics["optimal_threshold"] = best_threshold
    best_metrics["min_expected_cost"] = best_cost
    return best_metrics


def evaluate_per_lead_time(
    df: pd.DataFrame,
    lead_times: list[int],
    alert_thresholds: list[float],
) -> dict:
    """Evaluate metrics per forecast lead time.

    The lead time is encoded in the sample definition:
    - lead_time = event_date - sample_date
    """
    results = {}

    for lead in lead_times:
        lead_df = df[df["event_date"].notna()].copy()
        lead_df["lead_time"] = (
            pd.to_datetime(lead_df["event_date"])
            - pd.to_datetime(lead_df["sample_date"])
        ).dt.days
        lead_df = lead_df[lead_df["lead_time"] == lead]

        if len(lead_df) == 0:
            results[f"lead_{lead}d"] = {"status": "no_data"}
            continue

        y_true = lead_df["label"].values
        y_prob = (
            lead_df["probability"].values if "probability" in lead_df.columns else None
        )

        if y_prob is None:
            results[f"lead_{lead}d"] = {"status": "no_probabilities"}
            continue

        lead_metrics = {
            "n_samples": len(lead_df),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
        }

        # Standard metrics at 0.5 threshold
        if y_prob is not None:
            y_pred_50 = (y_prob >= 0.5).astype(int)
            lead_metrics["precision_0.5"] = precision_score(
                y_true, y_pred_50, zero_division=0
            )
            lead_metrics["recall_0.5"] = recall_score(
                y_true, y_pred_50, zero_division=0
            )
            lead_metrics["f1_0.5"] = f1_score(y_true, y_pred_50, zero_division=0)

            try:
                lead_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                lead_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
            except ValueError:
                lead_metrics["roc_auc"] = None
                lead_metrics["pr_auc"] = None

        # Metrics at each threshold
        if y_prob is not None:
            threshold_metrics = {}
            for t in alert_thresholds:
                y_pred = (y_prob >= t).astype(int)
                threshold_metrics[f"t_{t:.1f}"] = {
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0),
                    "n_alerts": int(y_pred.sum()),
                    "alert_rate": float(y_pred.mean()),
                }
            lead_metrics["thresholds"] = threshold_metrics

        results[f"lead_{lead}d"] = lead_metrics

    return results


def evaluate_per_zone(
    df: pd.DataFrame,
    zone_column: str = "zone_id",
    min_samples: int = 30,
) -> dict:
    """Evaluate metrics per spatial zone."""
    if zone_column not in df.columns:
        return {"status": "no_zone_column"}

    results = {}
    for zone_id, zone_df in df.groupby(zone_column):
        if len(zone_df) < min_samples:
            results[str(zone_id)] = {
                "status": "insufficient_samples",
                "n": len(zone_df),
            }
            continue

        y_true = zone_df["label"].values
        y_prob = (
            zone_df["probability"].values if "probability" in zone_df.columns else None
        )

        zone_metrics = {
            "n_samples": len(zone_df),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
        }

        if y_prob is not None:
            y_pred = (y_prob >= 0.5).astype(int)
            zone_metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            zone_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            zone_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

            try:
                zone_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                zone_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
            except ValueError:
                zone_metrics["roc_auc"] = None
                zone_metrics["pr_auc"] = None

        results[str(zone_id)] = zone_metrics

    return results


def evaluate_per_season(
    df: pd.DataFrame,
    date_column: str = "sample_date",
    seasons: dict | None = None,
) -> dict:
    """Evaluate metrics per season."""
    if seasons is None:
        seasons = {
            "pre_monsoon": (3, 5),
            "monsoon": (6, 9),
            "post_monsoon": (10, 11),
            "winter": (12, 2),
        }

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df["month"] = df[date_column].dt.month

    results = {}
    for season_name, (start_m, end_m) in seasons.items():
        if start_m <= end_m:
            mask = (df["month"] >= start_m) & (df["month"] <= end_m)
        else:  # wraps around year (winter)
            mask = (df["month"] >= start_m) | (df["month"] <= end_m)

        season_df = df[mask]
        if len(season_df) == 0:
            results[season_name] = {"status": "no_data"}
            continue

        y_true = season_df["label"].values
        y_prob = (
            season_df["probability"].values
            if "probability" in season_df.columns
            else None
        )

        season_metrics = {
            "n_samples": len(season_df),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
        }

        if y_prob is not None:
            y_pred = (y_prob >= 0.5).astype(int)
            season_metrics["precision"] = precision_score(
                y_true, y_pred, zero_division=0
            )
            season_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
            season_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

            try:
                season_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                season_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
            except ValueError:
                season_metrics["roc_auc"] = None
                season_metrics["pr_auc"] = None

        results[season_name] = season_metrics

    return results


def evaluate_temporal_stability(
    df: pd.DataFrame,
    date_column: str = "sample_date",
    window_days: int = 90,
    step_days: int = 30,
) -> dict:
    """Evaluate metric stability over rolling time windows."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    min_date = df[date_column].min()
    max_date = df[date_column].max()

    windows = []
    current = min_date
    while current + timedelta(days=window_days) <= max_date:
        window_end = current + timedelta(days=window_days)
        mask = (df[date_column] >= current) & (df[date_column] < window_end)
        window_df = df[mask]

        if len(window_df) >= 20:
            y_true = window_df["label"].values
            y_prob = (
                window_df["probability"].values
                if "probability" in window_df.columns
                else None
            )

            window_metrics = {
                "window_start": current.strftime("%Y-%m-%d"),
                "window_end": window_end.strftime("%Y-%m-%d"),
                "n_samples": len(window_df),
                "n_positive": int(y_true.sum()),
            }

            if y_prob is not None and len(np.unique(y_true)) > 1:
                y_pred = (y_prob >= 0.5).astype(int)
                window_metrics["precision"] = precision_score(
                    y_true, y_pred, zero_division=0
                )
                window_metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
                window_metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

                try:
                    window_metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
                    window_metrics["pr_auc"] = average_precision_score(y_true, y_prob)
                except ValueError:
                    window_metrics["roc_auc"] = None
                    window_metrics["pr_auc"] = None

            windows.append(window_metrics)

        current += timedelta(days=step_days)

    # Compute stability (coefficient of variation)
    stability = {}
    for metric in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        values = [w.get(metric) for w in windows if w.get(metric) is not None]
        if len(values) >= 3:
            stability[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "cv": float(np.std(values) / np.mean(values))
                if np.mean(values) > 0
                else None,
            }

    return {
        "windows": windows,
        "stability": stability,
        "n_windows": len(windows),
    }


def run_operational_evaluation(
    predictions_df: pd.DataFrame,
    config: EvaluationConfig | None = None,
) -> EvaluationResult:
    """Run full operational evaluation on prediction results.

    Args:
        predictions_df: DataFrame with columns:
            - label: true label (0/1)
            - probability: predicted probability
            - sample_date: date of prediction
            - event_date: date of event (for lead time)
            - zone_id: optional spatial zone
        config: Evaluation configuration.

    Returns:
        EvaluationResult with all metrics.
    """
    if config is None:
        config = EvaluationConfig()

    logger.info(f"Running operational evaluation on {len(predictions_df)} samples")

    # Overall metrics
    y_true = predictions_df["label"].values
    y_prob = (
        predictions_df["probability"].values
        if "probability" in predictions_df.columns
        else None
    )

    overall = {"n_samples": len(predictions_df), "n_positive": int(y_true.sum())}

    if y_prob is not None:
        y_pred = (y_prob >= 0.5).astype(int)
        overall["precision"] = precision_score(y_true, y_pred, zero_division=0)
        overall["recall"] = recall_score(y_true, y_pred, zero_division=0)
        overall["f1"] = f1_score(y_true, y_pred, zero_division=0)

        try:
            overall["roc_auc"] = roc_auc_score(y_true, y_prob)
            overall["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            overall["roc_auc"] = None
            overall["pr_auc"] = None

        # Cost-weighted optimal threshold
        cost_metrics = compute_cost_weighted_metrics(
            y_true, y_prob, config.cost_fn, config.cost_fp
        )
        overall["cost_analysis"] = cost_metrics

    # Per lead time
    per_lead = evaluate_per_lead_time(
        predictions_df, config.lead_times, config.alert_thresholds
    )

    # Per zone
    per_zone = evaluate_per_zone(
        predictions_df, min_samples=config.min_samples_per_zone
    )

    # Per season
    per_season = evaluate_per_season(predictions_df)

    # Temporal stability
    temporal = evaluate_temporal_stability(
        predictions_df, window_days=config.rolling_window_days
    )

    # Generate recommendations
    recommendations = generate_recommendations(overall, per_lead, per_zone, temporal)

    return EvaluationResult(
        overall_metrics=overall,
        per_lead_time=per_lead,
        per_threshold={},  # Included in per_lead_time
        per_zone=per_zone,
        per_season=per_season,
        temporal_stability=temporal,
        cost_analysis=overall.get("cost_analysis", {}),
        recommendations=recommendations,
    )


def generate_recommendations(
    overall: dict,
    per_lead: dict,
    per_zone: dict,
    temporal: dict,
) -> list[str]:
    """Generate actionable recommendations from evaluation."""
    recs = []

    # Overall performance
    if overall.get("roc_auc", 0) < 0.7:
        recs.append(
            "Overall ROC-AUC < 0.7: Consider adding more features or trying different model architecture"
        )

    if overall.get("precision", 0) < 0.3:
        recs.append(
            "Low precision: High false alarm rate. Consider raising alert threshold or improving negative sampling"
        )

    if overall.get("recall", 0) < 0.5:
        recs.append(
            "Low recall: Many events missed. Consider lowering alert threshold or improving positive sampling"
        )

    # Lead time degradation
    lead_aucs = {
        k: v.get("roc_auc") for k, v in per_lead.items() if v.get("roc_auc") is not None
    }
    if len(lead_aucs) >= 2:
        auc_1d = lead_aucs.get("lead_1d", 0)
        auc_7d = lead_aucs.get("lead_7d", 0)
        if auc_7d < auc_1d - 0.1:
            recs.append(
                "Significant performance drop at longer lead times: Features may not be predictive beyond short horizons"
            )

    # Zone variation
    zone_aucs = [
        v.get("roc_auc") for v in per_zone.values() if v.get("roc_auc") is not None
    ]
    if len(zone_aucs) >= 3 and (max(zone_aucs) - min(zone_aucs)) > 0.2:
        recs.append(
            "High spatial variation in performance: Consider zone-specific models or additional spatial features"
        )

    # Temporal stability
    for metric, stats in temporal.get("stability", {}).items():
        if stats.get("cv", 0) > 0.3:
            recs.append(
                f"High temporal instability in {metric} (CV={stats['cv']:.2f}): Consider time-aware features or seasonal models"
            )

    # Cost analysis
    cost = overall.get("cost_analysis", {})
    if cost.get("optimal_threshold", 0.5) > 0.5:
        recs.append(
            f"Cost-optimal threshold is {cost['optimal_threshold']:.2f} (>0.5): Current 0.5 threshold may be too aggressive"
        )

    if not recs:
        recs.append(
            "Model performance appears adequate for operational use. Continue monitoring."
        )

    return recs


def save_evaluation_report(
    result: EvaluationResult,
    output_path: str | Path,
) -> Path:
    """Save evaluation report to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    report = {
        "overall_metrics": result.overall_metrics,
        "per_lead_time": result.per_lead_time,
        "per_threshold": result.per_threshold,
        "per_zone": result.per_zone,
        "per_season": result.per_season,
        "temporal_stability": result.temporal_stability,
        "cost_analysis": result.cost_analysis,
        "recommendations": result.recommendations,
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Evaluation report saved to {path}")
    return path
