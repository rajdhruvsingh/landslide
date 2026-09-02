"""Landslide risk classifier with walk-forward temporal cross-validation.

This module implements the ML-4 model training pipeline:
1. Load dataset from ML-2/ML-3 pipeline
2. Feature engineering / selection
2. Walk-forward temporal cross-validation
3. Model training (XGBoost / RandomForest / Logistic Regression)
4. Calibration (Platt / Isotonic)
5. Evaluation with operational metrics (precision@lead_time, recall@lead_time)
6. Model serialization with metadata

Dependencies:
    - xgboost: pip install xgboost
    - scikit-learn: pip install scikit-learn
    - joblib: pip install joblib

Model Output:
    - risk_level: Low / Moderate / High / Severe
    - probability: P(landslide in next lead_time_days)
    - confidence: model confidence score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

logger = logging.getLogger(__name__)

# Feature columns (from ML-3 feature engineering)
FEATURE_COLUMNS = [
    # Rainfall
    "rainfall_current_mm",
    "rainfall_3d_mm",
    "rainfall_7d_mm",
    "rainfall_15d_mm",
    "rainfall_30d_mm",
    # Terrain
    "slope_angle_deg",
    "slope_aspect_deg",
    "elevation_m",
    # Proximity
    "distance_nearest_landslide_km",
    "n_landslides_within_5km",
    # Land cover
    "lulc_category",
    # Roads
    "road_distance_km",
]

# Categorical features that need encoding
CATEGORICAL_FEATURES = ["lulc_category"]

# Target column
TARGET_COLUMN = "label"


@dataclass
class ModelConfig:
    """Configuration for model training."""

    # Model type
    model_type: str = "xgboost"  # "xgboost", "random_forest", "logistic"

    # XGBoost params
    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "auc",
        }
    )

    # RandomForest params
    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    # Logistic Regression params
    lr_params: dict = field(
        default_factory=lambda: {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )

    # Walk-forward CV
    n_splits: int = 5
    gap_days: int = 30
    min_train_size: int = 100

    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # "isotonic" or "sigmoid"

    # Feature selection
    feature_selection: bool = False
    n_features_to_select: int | None = None

    # Output
    output_dir: str = "models"
    model_name: str = "landslide_risk_classifier"


@dataclass
class TrainingResult:
    """Results from model training."""

    model_path: str
    metrics: dict
    cv_metrics: list[dict]
    feature_importance: dict | None
    calibration_curve: dict | None
    class_weights: dict
    config: ModelConfig


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load training dataset from CSV."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def prepare_features(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare feature matrix and target vector.

    Args:
        df: DataFrame with features and target.
        feature_columns: List of feature column names. If None, uses FEATURE_COLUMNS.

    Returns:
        Tuple of (X, y) arrays.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    # Ensure all feature columns exist
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].values

    # Handle missing values
    X = X.fillna(0)

    # Convert categorical
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    return X.values, y


def create_model(config: ModelConfig):
    """Create model instance based on config."""
    if config.model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, falling back to RandomForest")
            return RandomForestClassifier(**config.rf_params)
        return xgb.XGBClassifier(**config.xgb_params)

    elif config.model_type == "random_forest":
        return RandomForestClassifier(**config.rf_params)

    elif config.model_type == "logistic":
        return LogisticRegression(**config.lr_params)

    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def get_feature_importance(model, feature_names: list[str]) -> dict:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        return dict(zip(feature_names, model.feature_importances_.tolist()))
    elif hasattr(model, "coef_"):
        # Linear model
        coef = np.abs(model.coef_).flatten()
        return dict(zip(feature_names, coef.tolist()))
    return {}


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    prefix: str = "",
) -> dict:
    """Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for AUC, PR-AUC).
        prefix: Prefix for metric names.

    Returns:
        Dict of metrics.
    """
    p = prefix + "_" if prefix else ""

    metrics = {
        f"{p}accuracy": accuracy_score(y_true, y_pred),
        f"{p}precision": precision_score(y_true, y_pred, zero_division=0),
        f"{p}recall": recall_score(y_true, y_pred, zero_division=0),
        f"{p}f1": f1_score(y_true, y_pred, zero_division=0),
        f"{p}support_pos": int(y_true.sum()),
        f"{p}support_neg": int((y_true == 0).sum()),
    }

    if y_prob is not None:
        try:
            metrics[f"{p}roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics[f"{p}roc_auc"] = None

        try:
            metrics[f"{p}pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics[f"{p}pr_auc"] = None

        # Precision@Recall thresholds
        if len(np.unique(y_true)) > 1:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            # Find threshold for 90% recall
            idx = np.where(recall >= 0.9)[0]
            if len(idx) > 0:
                metrics[f"{p}precision_at_90_recall"] = float(precision[idx[0]])
            else:
                metrics[f"{p}precision_at_90_recall"] = 0.0

    return metrics


def evaluate_operational(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lead_time_days: int,
    thresholds: list[float] = None,
) -> dict:
    """Evaluate model for operational early warning.

    Computes precision/recall at different probability thresholds,
    focusing on actionable warning levels.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        lead_time_days: Forecast lead time (for naming).
        thresholds: Probability thresholds to evaluate.

    Returns:
        Dict with operational metrics.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    metrics = {"lead_time_days": lead_time_days}

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        p = recall_score(y_true, y_pred, zero_division=0)
        r = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        n_alerts = int(y_pred.sum())

        metrics[f"threshold_{t:.1f}_precision"] = r
        metrics[f"threshold_{t:.1f}_recall"] = p
        metrics[f"threshold_{t:.1f}_f1"] = f1
        metrics[f"threshold_{t:.1f}_n_alerts"] = n_alerts

    return metrics


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    config: ModelConfig,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Perform walk-forward temporal cross-validation.

    Args:
        X: Feature matrix.
        y: Target vector.
        dates: Sample dates (datetime64 or date objects).
        config: Model configuration.
        feature_names: Feature names for importance.

    Returns:
        List of fold metrics.
    """
    # Convert dates to ordinal for sorting
    if hasattr(dates[0], "toordinal"):
        date_ordinals = np.array([d.toordinal() for d in dates])
    else:
        date_ordinals = np.array(
            [date.fromisoformat(str(d)).toordinal() for d in dates]
        )

    # Sort by date
    sort_idx = np.argsort(date_ordinals)
    X = X[sort_idx]
    y = y[sort_idx]
    date_ordinals = date_ordinals[sort_idx]

    # TimeSeriesSplit with gap
    tscv = TimeSeriesSplit(n_splits=config.n_splits, gap=config.gap_days)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < config.min_train_size:
            logger.warning(
                f"Fold {fold}: train size {len(train_idx)} < min {config.min_train_size}"
            )
            continue

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create and train model
        model = create_model(config)
        model.fit(X_train, y_train)

        # Calibrate if requested
        if config.calibrate:
            calibrator = CalibratedClassifierCV(
                model, method=config.calibration_method, cv=3
            )
            calibrator.fit(X_train, y_train)
            model = calibrator

        # Predict
        y_pred = model.predict(X_test)
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        # Evaluate
        fold_metrics = {
            "fold": fold,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_date_range": f"{date.fromordinal(int(date_ordinals[train_idx[0]]))} to {date.fromordinal(int(date_ordinals[train_idx[-1]]))}",
            "test_date_range": f"{date.fromordinal(int(date_ordinals[test_idx[0]]))} to {date.fromordinal(int(date_ordinals[test_idx[-1]]))}",
        }

        # Standard metrics
        fold_metrics.update(evaluate_model(y_test, y_pred, y_prob, prefix="test"))

        # Operational metrics
        if y_prob is not None:
            fold_metrics.update(evaluate_operational(y_test, y_prob, config=config))

        # Feature importance (last fold only)
        if fold == config.n_splits - 1 and feature_names:
            if hasattr(model, "named_steps"):
                # Pipeline
                final_estimator = model.named_steps.get("classifier", model)
            else:
                final_estimator = model
            fold_metrics["feature_importance"] = get_feature_importance(
                final_estimator, feature_names
            )

        fold_metrics = fold_metrics
        fold_metrics = fold_metrics
        fold_metrics = fold_metrics
        fold_metrics = fold_metrics
        fold_metrics_list = fold_metrics
        fold_metrics_list.append(fold_metrics)

    return fold_metrics_list


def train_model(
    dataset_path: str | Path,
    config: ModelConfig | None = None,
    cv_splits_path: str | Path | None = None,
) -> TrainingResult:
    """Train the landslide risk classifier with walk-forward CV.

    Args:
        dataset_path: Path to training CSV.
        config: Model configuration.
        cv_splits_path: Optional path to pre-computed CV splits.

    Returns:
        TrainingResult with model path and metrics.
    """
    if config is None:
        config = ModelConfig()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_dataset(dataset_path)

    # Ensure dates are parsed
    df["sample_date"] = pd.to_datetime(df["sample_date"])

    # Prepare features
    X, y = prepare_features(df)
    dates = df["sample_date"].values
    feature_names = [c for c in FEATURE_COLUMNS if c in df.columns]

    logger.info(
        f"Training {config.model_type} on {X.shape[0]} samples, {X.shape[1]} features"
    )
    logger.info(f"Class distribution: pos={y.sum()}, neg={len(y)-y.sum()}")

    # Walk-forward CV
    logger.info(
        f"Running walk-forward CV with {config.n_splits} splits, gap={config.gap_days} days"
    )
    cv_metrics = walk_forward_cv(X, y, dates, config, feature_names)

    # Aggregate CV metrics
    cv_summary = aggregate_cv_metrics(cv_metrics)
    logger.info(f"CV Summary: {cv_summary}")

    # Train final model on all data
    logger.info("Training final model on full dataset...")
    final_model = create_model(config)
    final_model.fit(X, y)

    # Calibrate
    if config.calibrate:
        calibrator = CalibratedClassifierCV(
            final_model, method=config.calibration_method, cv=3
        )
        calibrator.fit(X, y)
        final_model = calibrator

    # Feature importance
    feature_importance = get_feature_importance(final_model, feature_names)

    # Final evaluation on all data (for reference)
    y_pred = final_model.predict(X)
    y_prob = (
        final_model.predict_proba(X)[:, 1]
        if hasattr(final_model, "predict_proba")
        else None
    )
    final_metrics = evaluate_model(y, y_pred, y_prob, prefix="final")
    final_metrics.update(
        evaluate_operational(
            y, y_prob, config.lead_time_days if hasattr(config, "lead_time_days") else 1
        )
    )

    # Save model
    model_path = output_dir / f"{config.model_name}.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "config": config,
            "cv_metrics": cv_metrics,
            "feature_importance": feature_importance,
            "final_metrics": final_metrics,
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        model_path,
    )

    logger.info(f"Model saved to {model_path}")

    return TrainingResult(
        model_path=str(model_path),
        metrics=final_metrics,
        cv_metrics=cv_metrics,
        feature_importance=feature_importance,
        calibration_curve=None,  # TODO: add calibration curve
        class_weights={0: 1.0, 1: 1.0},  # TODO: compute
        config=config,
    )


def aggregate_cv_metrics(cv_metrics: list[dict]) -> dict:
    """Aggregate metrics across CV folds."""
    if not cv_metrics:
        return {}

    keys = [
        k for k in cv_metrics[0].keys() if isinstance(cv_metrics[0][k], (int, float))
    ]
    agg = {}
    for k in keys:
        values = [m[k] for m in cv_metrics if k in m and m[k] is not None]
        if values:
            agg[f"{k}_mean"] = np.mean(values)
            agg[f"{k}_std"] = np.std(values)
            agg[f"{k}_min"] = np.min(values)
            agg[f"{k}_max"] = np.max(values)
    return agg


def load_model(model_path: str | Path) -> dict:
    """Load trained model with metadata."""
    return joblib.load(model_path)


def predict_risk(
    model_path: str | Path,
    features: dict | pd.DataFrame,
) -> dict:
    """Predict landslide risk for new data.

    Args:
        model_path: Path to saved model.
        features: Dict or DataFrame with feature columns.

    Returns:
        Dict with risk_level, probability, confidence.
    """
    model_data = load_model(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features.copy()

    # Ensure all features present
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    X = df[feature_names].fillna(0).values

    # Predict
    prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    pred = model.predict(X)

    # Map to risk levels
    risk_map = {0: "Low", 1: "High"}  # Binary for now

    return {
        "risk_level": risk_map[int(pred[0])],
        "probability": float(prob[0]) if prob is not None else None,
        "confidence": float(prob[0]) if prob is not None else None,
        "model_version": model_data.get("trained_at", "unknown"),
    }


RISK_LEVELS = ["Low", "Moderate", "High", "Severe"]
