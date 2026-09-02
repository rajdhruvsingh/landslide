"""ML-3: Feature engineering pipeline for landslide risk classification.

This module computes all features required for the ML classifier, ensuring
strict temporal separation — no future information enters any feature vector.

DESIGN PRINCIPLES:
    1. Temporal integrity: All features use only data available BEFORE the
       sample date. No exceptions.
    2. Extensibility: Features are organized into composable transformers.
       Missing data sources are clearly marked, not fabricated.
    3. Provenance: Every feature value carries metadata about its source,
       availability, and whether it uses real or synthetic data.
    4. Testability: All transformers accept plain dicts, no database or
       API dependencies.
    5. Single source of truth: All feature computation logic lives here.
       Data pipeline uses these transformers for train/inference consistency.

FEATURE FAMILIES:
    - Rainfall: current + antecedent (3d, 7d, 15d, 30d)
    - Terrain: slope angle, slope aspect (from DEM)
    - Proximity: distance to nearest historical landslide (leakage-safe)
    - Land cover: extensible stub (no real data available)
    - Road density: extensible stub (no real data available)

LABELING METHODOLOGY:
    The task is landslide forecasting: given conditions at sample_date,
    predict whether a landslide occurs within lead_time_days.
    Positive samples have sample_date = event_date - lead_time_days.
    Features at sample_date must only use data available AT OR BEFORE sample_date.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Feature definitions (registry)
# ---------------------------------------------------------------------------


@dataclass
class FeatureDefinition:
    """Metadata for a single feature."""

    name: str
    description: str
    units: str
    source: str
    calculation_method: str
    spatial_resolution: str
    temporal_resolution: str
    missing_value_handler: str
    leakage_notes: str
    data_availability: str  # "AVAILABLE", "DEV_FIXTURE_ONLY", "UNAVAILABLE"


FEATURE_REGISTRY: dict[str, FeatureDefinition] = {}


def _register_feature(defn: FeatureDefinition) -> None:
    FEATURE_REGISTRY[defn.name] = defn


# --- Rainfall features ---

_register_feature(
    FeatureDefinition(
        name="rainfall_current_mm",
        description="Rainfall on the day immediately before the sample date (T-1)",
        units="mm",
        source="IMD daily rainfall",
        calculation_method="Nearest-station daily rainfall on day T-1",
        spatial_resolution="Station-level (point data, not gridded)",
        temporal_resolution="Daily",
        missing_value_handler="0.0 if no station within range or no data available",
        leakage_notes="Excludes sample date; uses only T-1 data",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

for window in [3, 7, 15, 30]:
    _register_feature(
        FeatureDefinition(
            name=f"rainfall_{window}d_mm",
            description=f"Sum of daily rainfall from T-{window} to T-1",
            units="mm",
            source="IMD daily rainfall",
            calculation_method=f"Sum of nearest-station daily rainfall over [{window} days ending T-1]",
            spatial_resolution="Station-level (point data, not gridded)",
            temporal_resolution="Daily",
            missing_value_handler="0.0 if no station within range or no data available",
            leakage_notes=f"Excludes sample date; uses only data from T-{window} to T-1",
            data_availability="DEV_FIXTURE_ONLY",
        )
    )

# --- Terrain features ---

_register_feature(
    FeatureDefinition(
        name="slope_angle_deg",
        description="Terrain slope angle derived from DEM",
        units="degrees",
        source="DEM (Bhuvan/SRTM)",
        calculation_method="3x3 window slope from DEM ( Horn's method or equivalent)",
        spatial_resolution="Same as DEM (30m for SRTM, configurable resampled to grid)",
        temporal_resolution="Static (time-invariant)",
        missing_value_handler="0.0 if DEM data unavailable",
        leakage_notes="Static feature — no temporal component",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

_register_feature(
    FeatureDefinition(
        name="slope_aspect_deg",
        description="Terrain slope aspect derived from DEM",
        units="degrees (0=N, 90=E, 180=S, 270=W)",
        source="DEM (Bhuvan/SRTM)",
        calculation_method="3x3 window aspect from DEM",
        spatial_resolution="Same as DEM",
        temporal_resolution="Static (time-invariant)",
        missing_value_handler="0.0 if DEM data unavailable",
        leakage_notes="Static feature — no temporal component",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

_register_feature(
    FeatureDefinition(
        name="elevation_m",
        description="Terrain elevation at grid cell centroid",
        units="meters",
        source="DEM (Bhuvan/SRTM) — currently DEV_FIXTURE",
        calculation_method="Bilinear interpolation at cell centroid",
        spatial_resolution="Same as DEM (30m, resampled to grid)",
        temporal_resolution="Static (time-invariant)",
        missing_value_handler="0.0 if DEM data unavailable",
        leakage_notes="Static feature — no temporal component",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

# --- Proximity features ---

_register_feature(
    FeatureDefinition(
        name="distance_nearest_landslide_km",
        description="Distance from grid cell centroid to nearest historical landslide",
        units="km",
        source="Landslide inventory (GSI/curated papers)",
        calculation_method="Haversine distance to nearest event with event_date < sample_date",
        spatial_resolution="1km grid cell centroid",
        temporal_resolution="Computed per sample_date (uses only past events)",
        missing_value_handler="999.0 if no historical landslides exist before sample_date",
        leakage_notes="CRITICAL: Only uses events with event_date < sample_date. For a positive sample, the target event itself is excluded.",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

_register_feature(
    FeatureDefinition(
        name="n_landslides_within_5km",
        description="Count of historical landslides within 5km of grid cell",
        units="count",
        source="Landslide inventory",
        calculation_method="Count of events with event_date < sample_date and distance < 5km",
        spatial_resolution="1km grid cell centroid",
        temporal_resolution="Computed per sample_date (uses only past events)",
        missing_value_handler="0 if no historical landslides exist before sample_date",
        leakage_notes="CRITICAL: Only uses events with event_date < sample_date",
        data_availability="DEV_FIXTURE_ONLY",
    )
)

# --- Land cover features (extensible stubs) ---

_register_feature(
    FeatureDefinition(
        name="lulc_category",
        description="Land use / land cover category",
        units="categorical integer",
        source="Land cover dataset (To be determined)",
        calculation_method="Majority class within grid cell from land cover raster",
        spatial_resolution="Raster resolution (typically 30m, resampled to grid)",
        temporal_resolution="Static (updated periodically)",
        missing_value_handler="0 (unknown) — NOT fabricated",
        leakage_notes="Static feature — no temporal component",
        data_availability="UNAVAILABLE",
    )
)

_register_feature(
    FeatureDefinition(
        name="road_distance_km",
        description="Distance to nearest road",
        units="km",
        source="OpenStreetMap road network",
        calculation_method="Nearest road segment distance from cell centroid",
        spatial_resolution="1km grid cell centroid",
        temporal_resolution="Static (road network snapshot)",
        missing_value_handler="999.0 (unknown) — NOT fabricated",
        leakage_notes="Static feature — no temporal component",
        data_availability="UNAVAILABLE",
    )
)


# ---------------------------------------------------------------------------
# Haversine distance (internal utility)
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Rainfall transformer
# ---------------------------------------------------------------------------


class RainfallFeatureTransformer:
    """Computes rainfall features for a single grid cell and date.

    Uses only rainfall data available BEFORE the sample date to prevent
    information leakage.
    """

    def __init__(
        self,
        rainfall_records: list[dict],
        max_station_distance_km: float = 50.0,
        windows: list[int] | None = None,
    ):
        """
        Args:
            rainfall_records: List of rainfall dicts with keys:
                station_id, station_lat, station_lon, reading_date, rainfall_mm
            max_station_distance_km: Max distance to consider a station relevant.
            windows: List of antecedent window sizes in days.
        """
        self.rainfall_records = rainfall_records
        self.max_station_distance_km = max_station_distance_km
        self.windows = windows or [3, 7, 15, 30]

    def _find_nearby_stations(
        self, cell_lat: float, cell_lon: float
    ) -> dict[str, float]:
        """Find nearby stations and return {station_id: distance_km}."""
        nearby = {}
        for record in self.rainfall_records:
            dist = _haversine_km(
                cell_lat,
                cell_lon,
                record["station_lat"],
                record["station_lon"],
            )
            if dist <= self.max_station_distance_km:
                sid = record["station_id"]
                if sid not in nearby or dist < nearby[sid]:
                    nearby[sid] = dist
        return nearby

    def compute(
        self,
        cell_lat: float,
        cell_lon: float,
        sample_date: date,
    ) -> dict[str, float]:
        """Compute all rainfall features for a cell at a given date.

        Uses only data from [sample_date - max_window, sample_date - 1].
        Rainfall on the sample date itself is NEVER included.

        Args:
            cell_lat: Grid cell centroid latitude.
            cell_lon: Grid cell centroid longitude.
            sample_date: The date for which features are computed.

        Returns:
            Dict mapping feature_name -> value.
        """
        nearby = self._find_nearby_stations(cell_lat, cell_lon)
        features: dict[str, float] = {}

        if not nearby:
            features["rainfall_current_mm"] = 0.0
            for w in self.windows:
                features[f"rainfall_{w}d_mm"] = 0.0
            return features

        # Filter records to nearby stations only
        relevant = [r for r in self.rainfall_records if r["station_id"] in nearby]

        # Current rainfall: T-1 only
        yesterday = sample_date - timedelta(days=1)
        current_total = sum(
            r["rainfall_mm"] for r in relevant if r["reading_date"] == yesterday
        )
        features["rainfall_current_mm"] = round(current_total, 2)

        # Antecedent windows: [sample_date - window, sample_date - 1]
        for window in self.windows:
            start = sample_date - timedelta(days=window)
            end = sample_date - timedelta(days=1)  # exclude sample date

            total = sum(
                r["rainfall_mm"] for r in relevant if start <= r["reading_date"] <= end
            )
            features[f"rainfall_{window}d_mm"] = round(total, 2)

        return features


# ---------------------------------------------------------------------------
# Terrain transformer
# ---------------------------------------------------------------------------


class TerrainFeatureTransformer:
    """Provides terrain features from a pre-computed static features dict.

    If real DEM data is unavailable, returns DEV_FIXTURE values clearly
    marked as such.
    """

    def __init__(self, static_features: dict[str, dict]):
        """
        Args:
            static_features: Dict mapping grid_cell_id -> feature dict.
                Expected keys: slope_angle_deg, slope_aspect_deg, elevation_m
        """
        self.static_features = static_features

    def compute(self, cell_id: str) -> dict[str, float | int]:
        """Return terrain features for a grid cell.

        Returns:
            Dict with slope_angle_deg, slope_aspect_deg, elevation_m.
            Values default to 0.0 if not available.
        """
        cell = self.static_features.get(cell_id, {})
        return {
            "slope_angle_deg": float(cell.get("slope_angle_deg", 0.0)),
            "slope_aspect_deg": float(cell.get("slope_aspect_deg", 0.0)),
            "elevation_m": float(cell.get("elevation_m", 0.0)),
        }


# ---------------------------------------------------------------------------
# Proximity transformer (LEAKAGE-SAFE)
# ---------------------------------------------------------------------------


class ProximityFeatureTransformer:
    """Computes distance to nearest historical landslide.

    LEAKAGE PREVENTION:
        For a sample at date T, only events with event_date < T are used.
        The target event itself is excluded from proximity computation.

        This means:
        - For a positive sample with event_date T, the nearest-landslide
          distance uses ALL OTHER historical events, not the target.
        - For a negative sample, all historical events before sample_date
          are used.
    """

    def __init__(self, landslide_events: list[dict]):
        """
        Args:
            landslide_events: List of event dicts with keys:
                event_id, event_date, latitude, longitude
        """
        self.events = landslide_events
        self.events.sort(key=lambda e: e["event_date"])

    def _get_past_events(
        self,
        sample_date: date,
        exclude_event_id: str | None = None,
    ) -> list[dict]:
        """Get events strictly before sample_date, optionally excluding one."""
        past_events = [e for e in self.events if e["event_date"] < sample_date]
        if exclude_event_id:
            past_events = [e for e in past_events if e["event_id"] != exclude_event_id]
        return past_events

    def is_cell_excluded(
        self,
        cell_lat: float,
        cell_lon: float,
        sample_date: date,
        exclusion_buffer_km: float,
        exclude_event_id: str | None = None,
    ) -> bool:
        """Check if a cell is within exclusion buffer of any past landslide.

        This is used for negative sampling to ensure negatives are not
        near any historical landslide (using only past events).

        Args:
            cell_lat: Grid cell centroid latitude.
            cell_lon: Grid cell centroid longitude.
            sample_date: The sample date. Only events before this are checked.
            exclusion_buffer_km: Buffer distance in km.
            exclude_event_id: Optional event_id to exclude.

        Returns:
            True if cell is within buffer of any past landslide.
        """
        past_events = self._get_past_events(sample_date, exclude_event_id)
        for event in past_events:
            dist = _haversine_km(
                cell_lat,
                cell_lon,
                event["latitude"],
                event["longitude"],
            )
            if dist <= exclusion_buffer_km:
                return True
        return False

    def compute(
        self,
        cell_lat: float,
        cell_lon: float,
        sample_date: date,
        exclude_event_id: str | None = None,
    ) -> dict[str, float | int]:
        """Compute proximity features using only past events.

        Args:
            cell_lat: Grid cell centroid latitude.
            cell_lon: Grid cell centroid longitude.
            sample_date: The sample date. Only events before this are used.
            exclude_event_id: Optional event_id to exclude (for leakage-safe
                computation on positive samples). If provided, the event with
                this ID is never used regardless of date.

        Returns:
            Dict with distance_nearest_landslide_km and n_landslides_within_5km.
        """
        past_events = self._get_past_events(sample_date, exclude_event_id)

        if not past_events:
            return {
                "distance_nearest_landslide_km": 999.0,
                "n_landslides_within_5km": 0,
            }

        min_dist = float("inf")
        n_within_5km = 0

        for event in past_events:
            dist = _haversine_km(
                cell_lat,
                cell_lon,
                event["latitude"],
                event["longitude"],
            )
            if dist < min_dist:
                min_dist = dist
            if dist <= 5.0:
                n_within_5km += 1

        return {
            "distance_nearest_landslide_km": round(min_dist, 4),
            "n_landslides_within_5km": n_within_5km,
        }

        min_dist = float("inf")
        n_within_5km = 0

        for event in past_events:
            dist = _haversine_km(
                cell_lat,
                cell_lon,
                event["latitude"],
                event["longitude"],
            )
            if dist < min_dist:
                min_dist = dist
            if dist <= 5.0:
                n_within_5km += 1

        return {
            "distance_nearest_landslide_km": round(min_dist, 4),
            "n_landslides_within_5km": n_within_5km,
        }


# ---------------------------------------------------------------------------
# Land cover transformer (extensible stub)
# ---------------------------------------------------------------------------


class LandCoverFeatureTransformer:
    """Land cover features. Currently unavailable — returns placeholder values.

    When real land cover data becomes available, replace this with actual
    raster-based lookups.
    """

    def compute(self, cell_id: str) -> dict[str, int]:
        """Return land cover features.

        Currently always returns 0 (unknown).
        """
        return {"lulc_category": 0}


# ---------------------------------------------------------------------------
# Road distance transformer (extensible stub)
# ---------------------------------------------------------------------------


class RoadFeatureTransformer:
    """Road distance features. Currently unavailable — returns placeholder values.

    When real road network data becomes available, replace with nearest-road
    distance computation.
    """

    def compute(self, cell_id: str) -> dict[str, float]:
        """Return road distance features.

        Currently always returns 999.0 (unknown).
        """
        return {"road_distance_km": 999.0}


# ---------------------------------------------------------------------------
# Composite feature transformer
# ---------------------------------------------------------------------------


class FeatureTransformer:
    """Composes all feature transformers into a single interface.

    Usage:
        transformer = FeatureTransformer(
            rainfall_records=records,
            static_features=static,
            landslide_events=events,
        )
        features = transformer.compute_all(
            cell_id="...",
            cell_lat=27.3,
            cell_lon=88.6,
            sample_date=date(2022, 7, 15),
            exclude_event_id="E1",  # for positive samples
        )
    """

    def __init__(
        self,
        rainfall_records: list[dict] | None = None,
        static_features: dict[str, dict] | None = None,
        landslide_events: list[dict] | None = None,
        max_station_distance_km: float = 50.0,
        rainfall_windows: list[int] | None = None,
    ):
        self.rainfall = RainfallFeatureTransformer(
            rainfall_records or [],
            max_station_distance_km=max_station_distance_km,
            windows=rainfall_windows or [3, 7, 15, 30],
        )
        self.terrain = TerrainFeatureTransformer(static_features or {})
        self.proximity = ProximityFeatureTransformer(landslide_events or [])
        self.landcover = LandCoverFeatureTransformer()
        self.road = RoadFeatureTransformer()

    def compute_all(
        self,
        cell_id: str,
        cell_lat: float,
        cell_lon: float,
        sample_date: date,
        exclude_event_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute all features for a single sample.

        Args:
            cell_id: Grid cell identifier.
            cell_lat: Cell centroid latitude.
            cell_lon: Cell centroid longitude.
            sample_date: Date of the sample.
            exclude_event_id: For positive samples, the event_id to exclude
                from proximity computation (prevents target-event leakage).

        Returns:
            Dict of all feature values.
        """
        features: dict[str, Any] = {}

        # Rainfall
        features.update(self.rainfall.compute(cell_lat, cell_lon, sample_date))

        # Terrain
        features.update(self.terrain.compute(cell_id))

        # Proximity (leakage-safe)
        features.update(
            self.proximity.compute(
                cell_lat,
                cell_lon,
                sample_date,
                exclude_event_id,
            )
        )

        # Land cover (stub)
        features.update(self.landcover.compute(cell_id))

        # Road (stub)
        features.update(self.road.compute(cell_id))

        return features

    def compute_all_for_dataset(
        self,
        dataset: list[dict],
    ) -> list[dict]:
        """Compute features for an entire ML-2 dataset.

        Each sample must have: grid_cell_id, centroid_lat, centroid_lon,
        sample_date, event_id, label.

        For positive samples (label=1), the event_id is excluded from
        proximity computation to prevent target-event leakage.

        Args:
            dataset: List of sample dicts from the ML-2 pipeline.

        Returns:
            List of dicts with original fields + all computed features.
        """
        enriched = []
        for sample in dataset:
            sample_date = (
                date.fromisoformat(sample["sample_date"])
                if isinstance(sample["sample_date"], str)
                else sample["sample_date"]
            )

            exclude_id = None
            if sample.get("label") == 1 and sample.get("event_id"):
                exclude_id = sample["event_id"]

            features = self.compute_all(
                cell_id=sample["grid_cell_id"],
                cell_lat=sample["centroid_lat"],
                cell_lon=sample["centroid_lon"],
                sample_date=sample_date,
                exclude_event_id=exclude_id,
            )

            enriched_sample = {**sample, **features}
            enriched.append(enriched_sample)

        return enriched


# ---------------------------------------------------------------------------
# Feature dictionary export
# ---------------------------------------------------------------------------


def get_feature_dictionary() -> list[dict]:
    """Return the complete feature dictionary as a list of dicts.

    Each dict describes one feature with its definition, units, source,
    calculation method, and leakage considerations.
    """
    return [
        {
            "name": defn.name,
            "description": defn.description,
            "units": defn.units,
            "source": defn.source,
            "calculation_method": defn.calculation_method,
            "spatial_resolution": defn.spatial_resolution,
            "temporal_resolution": defn.temporal_resolution,
            "missing_value_handler": defn.missing_value_handler,
            "leakage_notes": defn.leakage_notes,
            "data_availability": defn.data_availability,
        }
        for defn in FEATURE_REGISTRY.values()
    ]


def get_feature_names() -> list[str]:
    """Return ordered list of all feature names."""
    return list(FEATURE_REGISTRY.keys())
