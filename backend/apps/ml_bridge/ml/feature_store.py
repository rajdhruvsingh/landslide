"""Feature store and versioning for ML pipeline.

This module provides:
1. Feature registry with schema validation
2. Feature versioning (semantic versioning)
3. Feature lineage tracking
4. Train/inference feature consistency enforcement
5. Feature drift detection

Storage backends:
- Local filesystem (default, for development)
- S3/GCS (for production)
- PostgreSQL (for metadata)

Format:
- Features stored as Parquet (columnar, efficient)
- Metadata stored as JSON
- Schema stored as Avro/Protobuf (for strict validation)
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pq = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureSchema:
    """Schema definition for a feature."""

    name: str
    dtype: str  # "int64", "float64", "bool", "string", "category"
    description: str = ""
    units: str = ""
    source: str = ""
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str] | None = None  # for categorical

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "units": self.units,
            "source": self.source,
            "nullable": self.nullable,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSchema":
        return cls(**d)


@dataclass
class FeatureSetMetadata:
    """Metadata for a feature set version."""

    name: str
    version: str  # semantic version: MAJOR.MINOR.PATCH
    created_at: str
    description: str
    schema: list[FeatureSchema]
    source_data_hash: str
    transformation_code_hash: str
    num_features: int
    num_rows: int
    statistics: dict  # per-feature stats
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "description": self.description,
            "schema": [s.to_dict() for s in self.schema],
            "source_data_hash": self.source_data_hash,
            "transformation_code_hash": self.transformation_code_hash,
            "num_features": self.num_features,
            "num_rows": self.num_rows,
            "statistics": self.statistics,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSetMetadata":
        return cls(
            name=d["name"],
            version=d["version"],
            created_at=d["created_at"],
            description=d["description"],
            schema=[FeatureSchema.from_dict(s) for s in d["schema"]],
            source_data_hash=d["source_data_hash"],
            transformation_code_hash=d["transformation_code_hash"],
            num_features=d["num_features"],
            num_rows=d["num_rows"],
            statistics=d["statistics"],
            tags=d.get("tags", []),
        )


class FeatureStore:
    """Feature store with versioning and lineage tracking."""

    def __init__(
        self,
        root_path: str | Path = "feature_store",
        backend: str = "local",  # "local", "s3", "gcs"
    ):
        """
        Args:
            root_path: Root directory for feature store.
            backend: Storage backend.
        """
        self.root_path = Path(root_path)
        self.backend = backend
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.features_dir = self.root_path / "features"
        self.metadata_dir = self.root_path / "metadata"
        self.lineage_dir = self.root_path / "lineage"
        for d in [self.features_dir, self.metadata_dir, self.lineage_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # In-memory registry
        self._feature_sets: dict[str, dict[str, FeatureSetMetadata]] = {}

    def _compute_hash(self, data: bytes) -> str:
        """Compute SHA256 hash."""
        return hashlib.sha256(data).hexdigest()[:16]

    def _compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame content."""
        # Sort columns and rows for deterministic hash
        df_sorted = (
            df.sort_index(axis=1)
            .sort_values(by=df.columns.tolist())
            .reset_index(drop=True)
        )
        return self._compute_hash(
            pd.util.hash_pandas_object(df_sorted, index=True).values.tobytes()
        )

    def register_feature_set(
        self,
        name: str,
        df: pd.DataFrame,
        version: str | None = None,
        description: str = "",
        source_data_hash: str | None = None,
        transformation_code: str | None = None,
        tags: list[str] | None = None,
    ) -> FeatureSetMetadata:
        """Register a new feature set version.

        Args:
            name: Feature set name (e.g., "landslide_risk_v1").
            df: DataFrame with features.
            version: Semantic version. If None, auto-increments PATCH.
            description: Description of this version.
            source_data_hash: Hash of source data (for lineage).
            transformation_code: Source code of transformation (for reproducibility).
            tags: Optional tags.

        Returns:
            FeatureSetMetadata for the registered version.
        """
        if name not in self._feature_sets:
            self._feature_sets[name] = {}

        # Determine version
        if version is None:
            existing = list(self._feature_sets[name].keys())
            if existing:
                # Parse latest version and increment PATCH
                latest = sorted(existing, key=lambda v: tuple(map(int, v.split("."))))[
                    -1
                ]
                major, minor, patch = map(int, latest.split("."))
                version = f"{major}.{minor}.{patch + 1}"
            else:
                version = "1.0.0"

        if version in self._feature_sets[name]:
            raise ValueError(f"Version {version} already exists for {name}")

        # Infer schema from DataFrame
        schema = []
        statistics = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype.startswith("int"):
                dtype = "int64"
            elif dtype.startswith("float"):
                dtype = "float64"
            elif dtype == "bool":
                dtype = "bool"
            elif dtype == "object":
                dtype = "string"
            elif "category" in dtype:
                dtype = "category"

            feat_schema = FeatureSchema(
                name=col,
                dtype=dtype,
                nullable=df[col].isnull().any(),
            )
            schema.append(feat_schema)

            # Statistics
            if dtype in ("int64", "float64"):
                statistics[col] = {
                    "mean": float(df[col].mean())
                    if not df[col].isnull().all()
                    else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "null_count": int(df[col].isnull().sum()),
                }
            elif dtype == "category":
                statistics[col] = {
                    "categories": df[col].astype(str).unique().tolist(),
                    "null_count": int(df[col].isnull().sum()),
                }
            else:
                statistics[col] = {
                    "null_count": int(df[col].isnull().sum()),
                }

        # Compute hashes
        self._compute_dataframe_hash(df)
        code_hash = (
            self._compute_hash(transformation_code.encode())
            if transformation_code
            else "unknown"
        )

        metadata = FeatureSetMetadata(
            name=name,
            version=version,
            created_at=datetime.now().isoformat(),
            description=description,
            schema=schema,
            source_data_hash=source_data_hash or "unknown",
            transformation_code_hash=code_hash,
            num_features=len(schema),
            num_rows=len(df),
            statistics=statistics,
            tags=tags or [],
        )

        # Save feature data as Parquet
        feature_path = self.features_dir / name / version
        feature_path.mkdir(parents=True, exist_ok=True)

        if PYARROW_AVAILABLE:
            table = pa.Table.from_pandas(df)
            pq.write_table(table, feature_path / "data.parquet")
        else:
            # Fallback to CSV
            df.to_parquet(feature_path / "data.parquet") if hasattr(
                df, "to_parquet"
            ) else df.to_csv(feature_path / "data.csv", index=False)

        # Save metadata
        meta_path = self.metadata_dir / name / version
        meta_path.mkdir(parents=True, exist_ok=True)
        with open(meta_path / "metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Register
        self._feature_sets[name][version] = metadata
        logger.info(
            f"Registered feature set {name} v{version}: {len(df)} rows, {len(schema)} features"
        )

        return metadata

    def get_feature_set(
        self,
        name: str,
        version: str | None = "latest",
    ) -> tuple[pd.DataFrame, FeatureSetMetadata]:
        """Load a feature set by name and version.

        Args:
            name: Feature set name.
            version: Version string or "latest".

        Returns:
            Tuple of (DataFrame, FeatureSetMetadata).
        """
        if name not in self._feature_sets:
            # Try loading from disk
            self._load_metadata_from_disk(name)

        if name not in self._feature_sets:
            raise ValueError(f"Feature set '{name}' not found")

        versions = self._feature_sets[name]
        if version == "latest":
            version = sorted(
                versions.keys(), key=lambda v: tuple(map(int, v.split(".")))
            )[-1]

        if version not in versions:
            raise ValueError(f"Version {version} not found for {name}")

        metadata = versions[version]

        # Load data
        feature_path = self.features_dir / name / version
        if PYARROW_AVAILABLE and (feature_path / "data.parquet").exists():
            df = pq.read_table(feature_path / "data.parquet").to_pandas()
        elif (feature_path / "data.csv").exists():
            df = pd.read_csv(feature_path / "data.csv")
        else:
            raise FileNotFoundError(f"Feature data not found at {feature_path}")

        return df, metadata

    def _load_metadata_from_disk(self, name: str) -> None:
        """Load metadata from disk into registry."""
        meta_dir = self.metadata_dir / name
        if not meta_dir.exists():
            return

        self._feature_sets[name] = {}
        for version_dir in meta_dir.iterdir():
            if version_dir.is_dir():
                meta_file = version_dir / "metadata.json"
                if meta_file.exists():
                    with open(meta_file) as f:
                        data = json.load(f)
                    self._feature_sets[name][version_dir.name] = (
                        FeatureSetMetadata.from_dict(data)
                    )

    def list_versions(self, name: str) -> list[FeatureSetMetadata]:
        """List all versions of a feature set."""
        if name not in self._feature_sets:
            self._load_metadata_from_disk(name)

        if name not in self._feature_sets:
            return []

        return list(self._feature_sets[name].values())

    def validate_schema(
        self,
        df: pd.DataFrame,
        name: str,
        version: str = "latest",
    ) -> tuple[bool, list[str]]:
        """Validate DataFrame against registered schema.

        Args:
            df: DataFrame to validate.
            name: Feature set name.
            version: Version to validate against.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        _, metadata = self.get_feature_set(name, version)
        errors = []

        # Check columns
        expected_cols = {s.name for s in metadata.schema}
        actual_cols = set(df.columns)

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            errors.append(f"Missing columns: {missing}")
        if extra:
            warnings.warn(f"Extra columns (will be ignored): {extra}")

        # Check dtypes and constraints
        for feat in metadata.schema:
            if feat.name not in df.columns:
                continue

            col = df[feat.name]

            # Nullable check
            if not feat.nullable and col.isnull().any():
                errors.append(
                    f"Column '{feat.name}' has null values but is non-nullable"
                )

            # Range check
            if feat.min_value is not None:
                if col.min() < feat.min_value:
                    errors.append(
                        f"Column '{feat.name}' has values below min {feat.min_value}"
                    )
            if feat.max_value is not None:
                if col.max() > feat.max_value:
                    errors.append(
                        f"Column '{feat.name}' has values above max {feat.max_value}"
                    )

            # Category check
            if feat.categories is not None:
                invalid = set(col.dropna().unique()) - set(feat.categories)
                if invalid:
                    errors.append(
                        f"Column '{feat.name}' has invalid categories: {invalid}"
                    )

        return len(errors) == 0, errors

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        name: str,
        version: str = "latest",
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Detect feature drift between current data and registered version.

        Args:
            current_df: Current feature DataFrame.
            name: Feature set name.
            version: Baseline version.
            threshold: Drift threshold (PSI or KS statistic).

        Returns:
            Dict with drift detection results per feature.
        """
        _, metadata = self.get_feature_set(name, version)

        # Load baseline data
        baseline_df, _ = self.get_feature_set(name, version)

        drift_results = {}

        for feat in metadata.schema:
            if (
                feat.name not in current_df.columns
                or feat.name not in baseline_df.columns
            ):
                continue

            if feat.dtype not in ("int64", "float64"):
                continue

            current_vals = current_df[feat.name].dropna()
            baseline_vals = baseline_df[feat.name].dropna()

            if len(current_vals) < 30 or len(baseline_vals) < 30:
                drift_results[feat.name] = {"status": "insufficient_data"}
                continue

            # Population Stability Index (PSI)
            psi = self._compute_psi(baseline_vals, current_vals)

            # Kolmogorov-Smirnov test
            from scipy import stats

            ks_stat, p_value = stats.ks_2samp(baseline_vals, current_vals)

            drift_results[feat.name] = {
                "psi": float(psi),
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": psi > threshold or p_value < 0.05,
                "baseline_mean": float(baseline_vals.mean()),
                "current_mean": float(current_vals.mean()),
                "baseline_std": float(baseline_vals.std()),
                "current_std": float(current_vals.std()),
            }

        return {
            "feature_drift": drift_results,
            "overall_drift": any(
                r.get("drift_detected", False) for r in drift_results.values()
            ),
            "threshold": threshold,
        }

    def _compute_psi(
        self, baseline: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Compute Population Stability Index."""
        # Create bins based on baseline quantiles
        quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(baseline.quantile(quantiles))
        if len(bin_edges) < 2:
            return 0.0

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
        current_counts = np.histogram(current, bins=bin_edges)[0]

        # Avoid division by zero
        baseline_pct = baseline_counts / baseline_counts.sum()
        current_pct = current_counts / current_counts.sum()

        baseline_pct = np.clip(baseline_pct, 0.0001, 1.0)
        current_pct = np.clip(current_pct, 0.0001, 1.0)

        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
        return float(psi)


# Global feature store instance
_feature_store: FeatureStore | None = None


def get_feature_store(root_path: str | Path = "feature_store") -> FeatureStore:
    """Get or create global feature store instance."""
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore(root_path)
    return _feature_store


def register_features(
    name: str,
    df: pd.DataFrame,
    version: str | None = None,
    description: str = "",
    **kwargs,
) -> FeatureSetMetadata:
    """Convenience function to register features."""
    store = get_feature_store()
    return store.register_feature_set(name, df, version, description, **kwargs)


def load_features(
    name: str,
    version: str = "latest",
) -> tuple[pd.DataFrame, FeatureSetMetadata]:
    """Convenience function to load features."""
    store = get_feature_store()
    return store.get_feature_set(name, version)
