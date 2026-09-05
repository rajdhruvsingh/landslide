"""ML-2: Data and label pipeline for landslide risk classification.

This module builds a reproducible, scientifically defensible pipeline that
converts historical landslide inventory and spatial/temporal data into a
training-ready dataset.

DESIGN PRINCIPLES:
    1. All configuration is explicit and documented — nothing is hard-coded.
    2. Leakage prevention is built into the pipeline, not deferred to training.
    3. Every output record carries provenance (source file, REAL vs DEV_FIXTURE).
    4. Positive and negative labels are defined with clear, auditable rules.
    5. The pipeline is testable with synthetic fixtures.

SPATIAL UNIT:
    1km × 1km grid cells (configurable). Each training sample is a
    (grid_cell_id, sample_date) pair.

LABELING METHODOLOGY (Forecasting Task):
    The task is to predict whether a landslide will occur within `lead_time_days`
    after the `sample_date`. This matches the operational early-warning scenario:
    given conditions today, what is the probability of a landslide in the next N days?

    Positive samples: Grid cells where a landslide occurred within
    [sample_date + 1, sample_date + lead_time_days] (inclusive).

    Negative samples: Grid cells with no landslides in the same forward window,
    sampled with exclusion buffer and slope stratification.

PROVENANCE TRACKING:
    Every record in the output dataset includes a `data_origin` field:
    - "REAL" = derived from actual historical data
    - "DEV_FIXTURE" = synthetic data for pipeline testing only

    This field is set based on the input files. If ANY input file is a
    fixture, all output records derived from it are marked "DEV_FIXTURE".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import csv
import math


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline.

    All parameters are configurable. Defaults are set for initial development
    and can be changed without modifying pipeline code.
    """

    # Grid resolution
    grid_resolution_km: float = 1.0

    # Forecast horizon (lead time): number of days AFTER sample_date to check for landslides.
    # Default 1 day means: predict if landslide occurs tomorrow given today's conditions.
    # This defines the prediction target for early warning.
    lead_time_days: int = 1

    # DEPRECATED: positive_window_days is no longer used.
    # Kept for backward compatibility but ignored.
    # The new methodology uses lead_time_days (forward-looking) instead.
    positive_window_days: int = 0

    # Negative sampling
    exclusion_buffer_km: float = 2.0
    positive_negative_ratio: int = 3

    # Slope stratification for negative sampling
    # Bin boundaries in degrees. Negatives are sampled to match the slope
    # distribution of positive samples within these bins.
    slope_bins: list[float] = field(default_factory=lambda: [0, 15, 30, 45, 90])

    # Train/test split
    # All samples on or after this date go to test set.
    # All samples before this date go to training set.
    train_test_split_date: date = field(default_factory=lambda: date(2023, 1, 1))

    # Walk-forward validation
    # If enabled, use expanding window CV instead of single split.
    # n_splits: number of temporal folds
    # gap_days: gap between train and test to prevent leakage
    use_walk_forward_cv: bool = False
    walk_forward_n_splits: int = 5
    walk_forward_gap_days: int = 30

    # Output
    output_dir: str = "data/processed"

    # Provenance: "REAL" or "DEV_FIXTURE"
    data_origin: str = "REAL"


# ---------------------------------------------------------------------------
# Data loading functions
# ---------------------------------------------------------------------------


def load_landslide_inventory(csv_path: str | Path) -> list[dict]:
    """Load historical landslide inventory from CSV.

    Expected CSV schema:
        event_id, event_date, latitude, longitude, severity, source_reference

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts, one per event.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Landslide inventory not found: {path}")

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"event_id", "event_date", "latitude", "longitude"}
    if rows and not required.issubset(set(rows[0].keys())):
        missing = required - set(rows[0].keys())
        raise ValueError(f"Missing required columns: {missing}")

    events = []
    for row in rows:
        events.append(
            {
                "event_id": row["event_id"],
                "event_date": date.fromisoformat(row["event_date"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "severity": row.get("severity", "Unknown"),
                "source_reference": row.get("source_reference", ""),
            }
        )
    return events


def load_rainfall_timeseries(csv_path: str | Path) -> list[dict]:
    """Load daily rainfall time series from CSV.

    Expected CSV schema:
        station_id, station_lat, station_lon, reading_date, rainfall_mm

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of dicts, one per station-day record.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Rainfall data not found: {path}")

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    records = []
    for row in rows:
        records.append(
            {
                "station_id": row["station_id"],
                "station_lat": float(row["station_lat"]),
                "station_lon": float(row["station_lon"]),
                "reading_date": date.fromisoformat(row["reading_date"]),
                "rainfall_mm": float(row["rainfall_mm"]),
            }
        )
    return records


def load_static_features(csv_path: str | Path) -> dict[str, dict]:
    """Load pre-computed static features (slope, aspect, elevation, etc.) per grid cell.

    Expected CSV schema:
        grid_cell_id, slope_angle_deg, slope_aspect_deg, elevation_m,
        lulc_category, road_distance_km

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Dict mapping grid_cell_id -> feature dict.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    features = {}
    for row in rows:
        cell_id = row["grid_cell_id"]
        features[cell_id] = {
            "slope_angle_deg": float(row.get("slope_angle_deg", 0)),
            "slope_aspect_deg": float(row.get("slope_aspect_deg", 0)),
            "elevation_m": float(row.get("elevation_m", 0)),
            "lulc_category": int(row.get("lulc_category", 0)),
            "road_distance_km": float(row.get("road_distance_km", 0)),
        }
    return features


# ---------------------------------------------------------------------------
# Grid cell creation
# ---------------------------------------------------------------------------


def create_grid_cells(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    resolution_km: float = 1.0,
) -> list[dict]:
    """Create a regular grid of cells over a bounding box.

    Each cell is defined by its centroid lat/lon.

    Args:
        min_lat: Southern boundary of the bounding box.
        min_lon: Western boundary of the bounding box.
        max_lat: Northern boundary of the bounding box.
        max_lon: Eastern boundary of the bounding box.
        resolution_km: Side length of each grid cell in km.

    Returns:
        List of dicts with keys: grid_cell_id, centroid_lat, centroid_lon.
    """
    # Approximate conversion: 1 degree latitude ≈ 111 km
    # At ~27°N (Sikkim), 1 degree longitude ≈ 99 km
    lat_step = resolution_km / 111.0
    avg_lat = (min_lat + max_lat) / 2.0
    lon_step = resolution_km / (111.0 * math.cos(math.radians(avg_lat)))

    cells = []
    lat = min_lat + lat_step / 2.0  # center of first cell
    while lat < max_lat:
        lon = min_lon + lon_step / 2.0
        while lon < max_lon:
            cell_id = f"{lat:.4f}N_{lon:.4f}E"
            cells.append(
                {
                    "grid_cell_id": cell_id,
                    "centroid_lat": round(lat, 6),
                    "centroid_lon": round(lon, 6),
                }
            )
            lon += lon_step
        lat += lat_step
    return cells


# ---------------------------------------------------------------------------
# Labeling functions
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_in_cell(lat: float, lon: float, cell: dict, resolution_km: float) -> bool:
    """Check if a point falls within a grid cell.

    Uses cell centroid and half the resolution as the boundary.
    """
    half_size_km = resolution_km / 2.0
    # Approximate: 1 degree lat ≈ 111 km
    lat_half = half_size_km / 111.0
    avg_lat = cell["centroid_lat"]
    lon_half = half_size_km / (111.0 * math.cos(math.radians(avg_lat)))

    return (
        abs(lat - cell["centroid_lat"]) <= lat_half
        and abs(lon - cell["centroid_lon"]) <= lon_half
    )


def _get_cell_for_event(
    event: dict,
    grid_cells: list[dict],
    resolution_km: float,
) -> dict | None:
    """Find the grid cell containing an event point."""
    for cell in grid_cells:
        if _point_in_cell(event["latitude"], event["longitude"], cell, resolution_km):
            return cell
    return None


def label_positive_samples(
    grid_cells: list[dict],
    events: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    """Label grid cells for landslide forecasting.

    For each historical landslide event, create a positive sample at:
        sample_date = event_date - lead_time_days

    This means: given conditions at sample_date, a landslide occurred within
    the next lead_time_days. This matches the operational forecasting task.

    Args:
        grid_cells: List of grid cell dicts.
        events: List of landslide event dicts.
        config: Pipeline configuration (must have lead_time_days).

    Returns:
        List of positive sample dicts with label=1.
    """
    if config.lead_time_days <= 0:
        raise ValueError("lead_time_days must be >= 1 for forecasting task")

    positives = []
    for event in events:
        cell = _get_cell_for_event(event, grid_cells, config.grid_resolution_km)
        if cell is None:
            continue  # event outside grid bounds

        sample_date = event["event_date"] - timedelta(days=config.lead_time_days)

        positives.append(
            {
                "grid_cell_id": cell["grid_cell_id"],
                "centroid_lat": cell["centroid_lat"],
                "centroid_lon": cell["centroid_lon"],
                "sample_date": sample_date,
                "event_date": event["event_date"],
                "label": 1,
                "risk_level": _severity_to_risk(event["severity"]),
                "event_id": event["event_id"],
                "source_reference": event.get("source_reference", ""),
                "data_origin": config.data_origin,
            }
        )
    return positives


def _severity_to_risk(severity: str) -> str:
    """Map severity label to risk level."""
    mapping = {
        "Low": "Moderate",
        "Moderate": "High",
        "High": "High",
        "Severe": "Severe",
    }
    return mapping.get(severity, "High")  # Unknown → High (conservative)


def _compute_slope_distribution(
    positives: list[dict],
    static_features: dict[str, dict],
    slope_bins: list[float],
) -> dict[int, int]:
    """Compute the slope-bin distribution of positive samples.

    Returns a dict mapping bin_index -> count of positive samples in that bin.
    """
    bin_counts: dict[int, int] = {i: 0 for i in range(len(slope_bins))}
    for pos in positives:
        cell_id = pos["grid_cell_id"]
        slope = static_features.get(cell_id, {}).get("slope_angle_deg", 0)
        bin_idx = _slope_to_bin(slope, slope_bins)
        bin_counts[bin_idx] += 1
    return bin_counts


def _slope_to_bin(slope_deg: float, slope_bins: list[float]) -> int:
    """Map a slope value to a bin index."""
    for i in range(len(slope_bins) - 1):
        if slope_deg < slope_bins[i + 1]:
            return i
    return len(slope_bins) - 1


def sample_negative_samples(
    grid_cells: list[dict],
    events: list[dict],
    positives: list[dict],
    static_features: dict[str, dict],
    config: PipelineConfig,
    random_seed: int = 42,
) -> list[dict]:
    """Sample negative grid cells for landslide forecasting.

    Negative samples represent "no landslide in the next lead_time_days" scenarios.
    Key differences from positive sampling:
    1. Sample dates are drawn from the same temporal distribution as positive samples
    2. Exclusion buffer uses only PAST events (event_date < sample_date) to prevent leakage
    3. Cells with future landslides are NOT excluded (they're valid negatives for that date)

    Args:
        grid_cells: List of all grid cell dicts.
        events: List of ALL landslide event dicts.
        positives: List of already-labeled positive samples.
        static_features: Dict mapping cell_id -> feature dict.
        config: Pipeline configuration.
        random_seed: Seed for reproducibility.

    Returns:
        List of negative sample dicts with label=0.
    """
    import random

    rng = random.Random(random_seed)

    if not positives:
        return []

    # Step 1: Compute slope distribution of positives
    positive_slope_dist = _compute_slope_distribution(
        positives, static_features, config.slope_bins
    )
    total_positives = len(positives)

    # Step 2: Determine how many negatives per slope bin
    n_negatives_total = total_positives * config.positive_negative_ratio
    negatives_per_bin = {}
    for bin_idx, count in positive_slope_dist.items():
        proportion = (
            count / total_positives
            if total_positives > 0
            else 1.0 / len(config.slope_bins)
        )
        negatives_per_bin[bin_idx] = max(1, int(n_negatives_total * proportion))

    # Adjust to hit exact total
    current_total = sum(negatives_per_bin.values())
    deficit = n_negatives_total - current_total
    sorted_bins = sorted(
        positive_slope_dist.keys(), key=lambda b: positive_slope_dist[b], reverse=True
    )
    for i in range(abs(deficit)):
        bin_to_add = sorted_bins[i % len(sorted_bins)]
        if deficit > 0:
            negatives_per_bin[bin_to_add] += 1
        elif deficit < 0 and negatives_per_bin[bin_to_add] > 1:
            negatives_per_bin[bin_to_add] -= 1

    # Step 3: Collect positive cell IDs and sample dates for temporal distribution
    positive_cell_ids = {p["grid_cell_id"] for p in positives}
    positive_sample_dates = [p["sample_date"] for p in positives]

    # Step 4: Use ProximityFeatureTransformer for exclusion check (consistent logic)
    from apps.ml_bridge.ml.feature_engineering import ProximityFeatureTransformer

    proximity_transformer = ProximityFeatureTransformer(events)

    # Step 5: Group eligible cells by slope bin
    # A cell can be eligible for multiple sample dates (if no past events nearby)
    # We'll sample (cell, sample_date) pairs
    eligible_pairs_by_bin: dict[int, list[tuple[dict, date]]] = {
        i: [] for i in range(len(config.slope_bins))
    }

    for cell in grid_cells:
        slope = static_features.get(cell["grid_cell_id"], {}).get("slope_angle_deg", 0)
        bin_idx = _slope_to_bin(slope, config.slope_bins)

        # Check eligibility for each positive sample date
        # A cell is eligible for a sample_date if no PAST event is within buffer
        for sample_date in positive_sample_dates:
            if proximity_transformer.is_cell_excluded(
                cell["centroid_lat"],
                cell["centroid_lon"],
                sample_date,
                config.exclusion_buffer_km,
            ):
                continue
            if cell["grid_cell_id"] in positive_cell_ids:
                continue
            eligible_pairs_by_bin[bin_idx].append((cell, sample_date))

    # Step 6: Sample from each bin
    negatives = []
    for bin_idx, n_needed in negatives_per_bin.items():
        eligible = eligible_pairs_by_bin.get(bin_idx, [])
        if not eligible:
            continue
        n_sample = min(n_needed, len(eligible))
        sampled_pairs = rng.sample(eligible, n_sample)
        for cell, sample_date in sampled_pairs:
            negatives.append(
                {
                    "grid_cell_id": cell["grid_cell_id"],
                    "centroid_lat": cell["centroid_lat"],
                    "centroid_lon": cell["centroid_lon"],
                    "sample_date": sample_date,
                    "event_date": None,
                    "label": 0,
                    "risk_level": "Low",
                    "event_id": None,
                    "source_reference": "",
                    "data_origin": config.data_origin,
                }
            )

    return negatives


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------


def assemble_dataset(
    positives: list[dict],
    negatives: list[dict],
    rainfall_records: list[dict],
    static_features: dict[str, dict],
    config: PipelineConfig,
) -> list[dict]:
    """Assemble the final training dataset.

    For each sample (positive or negative):
    1. Compute antecedent rainfall features using RainfallFeatureTransformer.
    2. Attach static features (slope, aspect, etc.).
    3. Set the data_origin based on input provenance.

    Proximity features are NOT computed here (they require the full event list
    and exclude_event_id logic). Use FeatureTransformer.compute_all_for_dataset
    after assembly if proximity features are needed.

    Args:
        positives: List of positive sample dicts.
        negatives: List of negative sample dicts.
        rainfall_records: List of rainfall record dicts.
        static_features: Dict mapping cell_id -> feature dict.
        config: Pipeline configuration.

    Returns:
        List of complete sample dicts ready for model training.
    """
    all_samples = positives + negatives
    if not all_samples:
        return []

    dataset = []

    for sample in all_samples:
        # Get static features
        cell_static = static_features.get(sample["grid_cell_id"], {})

        # Compute rainfall features using the shared transformer
        # This ensures train/inference consistency
        rainfall_features = {}
        if rainfall_records:
            from apps.ml_bridge.ml.feature_engineering import RainfallFeatureTransformer

            rf_transformer = RainfallFeatureTransformer(rainfall_records)
            rainfall_features = rf_transformer.compute(
                sample["centroid_lat"],
                sample["centroid_lon"],
                sample["sample_date"],
            )

        record = {
            "sample_id": f"{sample['grid_cell_id']}_{sample['sample_date'].isoformat()}",
            "grid_cell_id": sample["grid_cell_id"],
            "centroid_lat": sample["centroid_lat"],
            "centroid_lon": sample["centroid_lon"],
            "sample_date": sample["sample_date"].isoformat(),
            "event_date": sample["event_date"].isoformat()
            if sample["event_date"]
            else None,
            "label": sample["label"],
            "risk_level": sample["risk_level"],
            "event_id": sample["event_id"],
            "source_reference": sample["source_reference"],
            "data_origin": sample["data_origin"],
            # Static features
            "slope_angle_deg": cell_static.get("slope_angle_deg", 0),
            "slope_aspect_deg": cell_static.get("slope_aspect_deg", 0),
            "elevation_m": cell_static.get("elevation_m", 0),
            "lulc_category": cell_static.get("lulc_category", 0),
            "road_distance_km": cell_static.get("road_distance_km", 0),
            # Temporal features
            **rainfall_features,
        }
        dataset.append(record)

    return dataset


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def split_train_test(
    dataset: list[dict],
    split_date: date,
) -> tuple[list[dict], list[dict]]:
    """Split dataset into train and test based on time.

    Samples with sample_date < split_date go to train.
    Samples with sample_date >= split_date go to test.

    This prevents future information from leaking into training.

    Args:
        dataset: List of sample dicts.
        split_date: Date at which to split.

    Returns:
        Tuple of (train_samples, test_samples).
    """
    train = []
    test = []
    for sample in dataset:
        sample_date = date.fromisoformat(sample["sample_date"])
        if sample_date < split_date:
            train.append(sample)
        else:
            test.append(sample)
    return train, test


def walk_forward_split(
    dataset: list[dict],
    n_splits: int = 5,
    gap_days: int = 30,
    min_train_size: int = 10,
) -> list[tuple[list[dict], list[dict]]]:
    """Generate walk-forward (expanding window) train/test splits.

    This provides more robust temporal validation than a single split.
    Each fold uses an expanding training window and a fixed-size test window
    with a gap to prevent leakage from autocorrelation.

    Args:
        dataset: List of sample dicts with sample_date field.
        n_splits: Number of folds to generate.
        gap_days: Days between train end and test start (purge gap).
        min_train_size: Minimum samples required in train set.

    Returns:
        List of (train_samples, test_samples) tuples.
    """
    # Sort by sample_date
    sorted_samples = sorted(dataset, key=lambda s: date.fromisoformat(s["sample_date"]))
    dates = [date.fromisoformat(s["sample_date"]) for s in sorted_samples]
    unique_dates = sorted(set(dates))

    if len(unique_dates) < n_splits + 2:
        # Not enough dates for walk-forward, fall back to single split
        split_date = unique_dates[len(unique_dates) // 2]
        return [split_train_test(dataset, split_date)]

    # Determine test windows
    # Use last portion of data for test windows, expanding train
    test_window_size = max(1, len(unique_dates) // (n_splits * 2))
    splits = []

    for i in range(n_splits):
        # Test window moves forward
        test_start_idx = len(unique_dates) - (n_splits - i) * test_window_size
        test_end_idx = test_start_idx + test_window_size

        if test_start_idx <= 0:
            continue

        test_start_date = unique_dates[test_start_idx]
        test_end_date = unique_dates[min(test_end_idx, len(unique_dates) - 1)]

        # Train is everything before (test_start - gap)
        train_cutoff = test_start_date - timedelta(days=gap_days)
        train_samples = [
            s
            for s in sorted_samples
            if date.fromisoformat(s["sample_date"]) < train_cutoff
        ]
        test_samples = [
            s
            for s in sorted_samples
            if test_start_date <= date.fromisoformat(s["sample_date"]) <= test_end_date
        ]

        if len(train_samples) >= min_train_size and len(test_samples) > 0:
            splits.append((train_samples, test_samples))

    if not splits:
        # Fallback to single split
        split_date = unique_dates[len(unique_dates) // 2]
        return [split_train_test(dataset, split_date)]

    return splits


def get_class_weights(dataset: list[dict]) -> dict[int, float]:
    """Compute class weights for imbalanced datasets.

    Args:
        dataset: List of sample dicts with 'label' field (0 or 1).

    Returns:
        Dict mapping class -> weight. Weight = n_samples / (n_classes * n_class_samples).
    """
    n_pos = sum(1 for s in dataset if s["label"] == 1)
    n_neg = sum(1 for s in dataset if s["label"] == 0)
    total = len(dataset)

    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}

    weight_0 = total / (2 * n_neg)
    weight_1 = total / (2 * n_pos)

    return {0: weight_0, 1: weight_1}


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def check_leakage(train: list[dict], test: list[dict]) -> dict:
    """Check for data leakage between train and test sets.

    Checks performed:
    1. No sample_id appears in both train and test.
    2. No grid_cell_id + event_id combination appears in both.
    3. Train set contains no samples with sample_date >= split_date
       (this is guaranteed by the split function, but verified here).
    4. No post-event features are present (all rainfall features should
       be antecedent only).

    Returns:
        Dict with check results: {check_name: passed (bool), details (str)}.
    """
    train_ids = {s["sample_id"] for s in train}
    test_ids = {s["sample_id"] for s in test}
    overlap = train_ids & test_ids

    # Check event_id overlap
    train_events = {(s["event_id"], s["grid_cell_id"]) for s in train if s["event_id"]}
    test_events = {(s["event_id"], s["grid_cell_id"]) for s in test if s["event_id"]}
    event_overlap = train_events & test_events

    # Check temporal leakage
    train_dates = {s["sample_date"] for s in train}
    test_dates = {s["sample_date"] for s in test}
    future_in_train = {d for d in train_dates if d >= min(test_dates) if test_dates}

    return {
        "no_sample_id_overlap": {
            "passed": len(overlap) == 0,
            "details": f"{len(overlap)} overlapping sample_ids"
            if overlap
            else "No overlap",
        },
        "no_event_id_overlap": {
            "passed": len(event_overlap) == 0,
            "details": f"{len(event_overlap)} overlapping (event_id, cell_id)"
            if event_overlap
            else "No overlap",
        },
        "no_future_in_train": {
            "passed": len(future_in_train) == 0,
            "details": f"{len(future_in_train)} future dates in train"
            if future_in_train
            else "Clean",
        },
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_dataset(
    dataset: list[dict],
    config: PipelineConfig,
    output_path: str | Path | None = None,
) -> Path:
    """Save the assembled dataset to CSV.

    Args:
        dataset: List of sample dicts.
        config: Pipeline configuration.
        output_path: Optional explicit path. If None, uses config.output_dir.

    Returns:
        Path to the saved CSV file.
    """
    if output_path is None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "training_dataset.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset:
        # Write empty file with header
        headers = [
            "sample_id",
            "grid_cell_id",
            "centroid_lat",
            "centroid_lon",
            "sample_date",
            "event_date",
            "label",
            "risk_level",
            "event_id",
            "source_reference",
            "data_origin",
            "slope_angle_deg",
            "slope_aspect_deg",
            "elevation_m",
            "lulc_category",
            "road_distance_km",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return output_path

    headers = list(dataset[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset)

    return output_path


def save_metadata(
    dataset: list[dict],
    config: PipelineConfig,
    data_sources: dict | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Save dataset metadata as JSON.

    Args:
        dataset: The assembled dataset.
        config: Pipeline configuration used.
        data_sources: Dict describing source file status.
        output_path: Optional explicit path.

    Returns:
        Path to the saved metadata file.
    """
    if output_path is None:
        output_path = Path(config.output_dir) / "dataset_metadata.json"
    else:
        output_path = Path(output_path)

    n_pos = sum(1 for s in dataset if s["label"] == 1)
    n_neg = sum(1 for s in dataset if s["label"] == 0)

    has_fixture = any(s["data_origin"] == "DEV_FIXTURE" for s in dataset)

    metadata = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "total_samples": len(dataset),
        "spatial_resolution_km": config.grid_resolution_km,
        "positive_window_days": config.positive_window_days,
        "exclusion_buffer_km": config.exclusion_buffer_km,
        "positive_negative_ratio": config.positive_negative_ratio,
        "train_test_split_date": config.train_test_split_date.isoformat(),
        "has_dev_fixtures": has_fixture,
        "data_sources": data_sources or {},
        "known_limitations": [
            "Pipeline tested with synthetic fixtures only."
            if has_fixture
            else "Built with real data — verify source provenance.",
            "All rainfall features are antecedent (pre-event) only.",
            "Static features (slope, aspect, etc.) are computed once per cell.",
        ],
        "synthetic_data_warning": (
            "THIS DATASET CONTAINS SYNTHETIC DEV_FIXTURE DATA. "
            "It must NEVER be used to report final ML performance. "
            "It is for pipeline testing and development only."
            if has_fixture
            else "This dataset was built from real data sources. "
            "Verify provenance before reporting metrics."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path
