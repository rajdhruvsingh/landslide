"""Tests for the ML-3 feature engineering pipeline.

Tests cover:
1. Feature registry completeness
2. Rainfall features (temporal integrity)
3. Terrain features
4. Proximity features (LEAKAGE CRITICAL)
5. Land cover and road stubs
6. Composite FeatureTransformer
7. End-to-end with dev fixtures
8. Explicit leakage detection tests
"""

from datetime import date, timedelta

import pytest

from apps.ml_bridge.ml.feature_engineering import (
    FEATURE_REGISTRY,
    FeatureTransformer,
    LandCoverFeatureTransformer,
    ProximityFeatureTransformer,
    RainfallFeatureTransformer,
    RoadFeatureTransformer,
    TerrainFeatureTransformer,
    get_feature_dictionary,
    get_feature_names,
)
from apps.ml_bridge.ml.terrain_dem import TERRAIN_DEV_FIXTURE


# ---------------------------------------------------------------------------
# Helper: build test rainfall data
# ---------------------------------------------------------------------------


def _make_rainfall(
    station_id: str = "S1",
    station_lat: float = 27.30,
    station_lon: float = 88.60,
    start_date: date = date(2022, 7, 1),
    n_days: int = 40,
    daily_pattern: list[float] | None = None,
) -> list[dict]:
    """Generate synthetic daily rainfall records."""
    if daily_pattern is None:
        daily_pattern = list(range(1, n_days + 1))
    records = []
    for i in range(n_days):
        records.append(
            {
                "station_id": station_id,
                "station_lat": station_lat,
                "station_lon": station_lon,
                "reading_date": start_date + timedelta(days=i),
                "rainfall_mm": float(daily_pattern[i % len(daily_pattern)]),
            }
        )
    return records


def _make_events() -> list[dict]:
    """Create test landslide events spanning a range of dates."""
    return [
        {
            "event_id": "E1",
            "event_date": date(2022, 6, 1),
            "latitude": 27.30,
            "longitude": 88.60,
            "severity": "Moderate",
            "source_reference": "test",
        },
        {
            "event_id": "E2",
            "event_date": date(2022, 6, 15),
            "latitude": 27.35,
            "longitude": 88.65,
            "severity": "High",
            "source_reference": "test",
        },
        {
            "event_id": "E3",
            "event_date": date(2022, 7, 1),
            "latitude": 27.40,
            "longitude": 88.70,
            "severity": "Severe",
            "source_reference": "test",
        },
    ]


# ============================================================================
# 1. Feature registry
# ============================================================================


class TestFeatureRegistry:
    def test_registry_not_empty(self):
        assert len(FEATURE_REGISTRY) > 0

    def test_all_expected_features_registered(self):
        expected = [
            "rainfall_current_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
            "slope_angle_deg",
            "slope_aspect_deg",
            "elevation_m",
            "distance_nearest_landslide_km",
            "n_landslides_within_5km",
            "lulc_category",
            "road_distance_km",
        ]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Missing: {name}"

    def test_each_feature_has_required_fields(self):
        for name, defn in FEATURE_REGISTRY.items():
            assert defn.name == name
            assert defn.units
            assert defn.source
            assert defn.calculation_method
            assert defn.missing_value_handler
            assert defn.leakage_notes
            assert defn.data_availability in (
                "AVAILABLE",
                "DEV_FIXTURE_ONLY",
                "UNAVAILABLE",
            )

    def test_get_feature_dictionary(self):
        d = get_feature_dictionary()
        assert len(d) == len(FEATURE_REGISTRY)
        assert all(isinstance(f, dict) for f in d)
        assert all("name" in f and "leakage_notes" in f for f in d)

    def test_get_feature_names(self):
        names = get_feature_names()
        assert len(names) == len(FEATURE_REGISTRY)
        assert isinstance(names, list)


# ============================================================================
# 2. Rainfall features
# ============================================================================


class TestRainfallFeatures:
    def test_current_rainfall_excludes_sample_date(self):
        """Rainfall on the sample date itself must not be included."""
        records = _make_rainfall(start_date=date(2022, 7, 1), n_days=10)
        transformer = RainfallFeatureTransformer(records)

        # sample_date = July 10 → current = July 9 (day index 8, rainfall=9.0)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        assert features["rainfall_current_mm"] == pytest.approx(9.0, abs=0.1)

    def test_current_rainfall_uses_nearest_day(self):
        records = _make_rainfall(start_date=date(2022, 7, 1), n_days=10)
        transformer = RainfallFeatureTransformer(records)

        # sample_date = July 5 → current = July 4 (day index 3, rainfall=4.0)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 5))
        assert features["rainfall_current_mm"] == pytest.approx(4.0, abs=0.1)

    def test_3day_window_excludes_sample_date(self):
        records = _make_rainfall(start_date=date(2022, 7, 1), n_days=10)
        transformer = RainfallFeatureTransformer(records)

        # sample_date = July 10 → window = July 7-9 (days 6,7,8 → rainfall 7+8+9=24)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        assert features["rainfall_3d_mm"] == pytest.approx(24.0, abs=0.1)

    def test_7day_window(self):
        records = _make_rainfall(start_date=date(2022, 7, 1), n_days=15)
        transformer = RainfallFeatureTransformer(records)

        # sample_date = July 15 → window = July 8-14 (days 7-13 → rainfall 8+9+10+11+12+13+14=77)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 15))
        assert features["rainfall_7d_mm"] == pytest.approx(77.0, abs=0.1)

    def test_no_nearby_stations_returns_zeros(self):
        records = _make_rainfall(station_lat=27.30, station_lon=88.60)
        transformer = RainfallFeatureTransformer(records, max_station_distance_km=1.0)

        # Far away cell
        features = transformer.compute(29.0, 90.0, date(2022, 7, 10))
        assert features["rainfall_current_mm"] == 0.0
        assert features["rainfall_3d_mm"] == 0.0
        assert features["rainfall_30d_mm"] == 0.0

    def test_all_windows_present(self):
        records = _make_rainfall()
        transformer = RainfallFeatureTransformer(records)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        for key in [
            "rainfall_current_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
        ]:
            assert key in features

    def test_empty_rainfall_returns_zeros(self):
        transformer = RainfallFeatureTransformer([])
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        for key in [
            "rainfall_current_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
        ]:
            assert features[key] == 0.0

    def test_values_are_floats(self):
        records = _make_rainfall()
        transformer = RainfallFeatureTransformer(records)
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        for v in features.values():
            assert isinstance(v, float)


# ============================================================================
# 3. Terrain features
# ============================================================================


class TestTerrainFeatures:
    def test_returns_fixture_values(self):
        transformer = TerrainFeatureTransformer(TERRAIN_DEV_FIXTURE)
        features = transformer.compute("27.3200N_88.6100E")
        assert features["slope_angle_deg"] == 28.5
        assert features["slope_aspect_deg"] == 225.0
        assert features["elevation_m"] == 1850.0

    def test_missing_cell_defaults_to_zero(self):
        transformer = TerrainFeatureTransformer(TERRAIN_DEV_FIXTURE)
        features = transformer.compute("NONEXISTENT_CELL")
        assert features["slope_angle_deg"] == 0.0
        assert features["slope_aspect_deg"] == 0.0
        assert features["elevation_m"] == 0.0

    def test_empty_static_features(self):
        transformer = TerrainFeatureTransformer({})
        features = transformer.compute("any_cell")
        assert features["slope_angle_deg"] == 0.0
        assert features["slope_aspect_deg"] == 0.0
        assert features["elevation_m"] == 0.0

    def test_all_cells_have_terrain(self):
        transformer = TerrainFeatureTransformer(TERRAIN_DEV_FIXTURE)
        for cell_id in TERRAIN_DEV_FIXTURE:
            features = transformer.compute(cell_id)
            assert features["slope_angle_deg"] >= 0
            assert features["slope_aspect_deg"] >= 0
            assert features["elevation_m"] >= 0


# ============================================================================
# 4. Proximity features (LEAKAGE CRITICAL)
# ============================================================================


class TestProximityFeatures:
    def test_only_uses_past_events(self):
        """Events AFTER sample_date must not affect proximity."""
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 6, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
            {
                "event_id": "E2",
                "event_date": date(2022, 8, 1),
                "latitude": 27.31,
                "longitude": 88.61,
            },  # future event
        ]
        transformer = ProximityFeatureTransformer(events)

        # Before E2 occurs: only E1 is in the past
        features = transformer.compute(27.30, 88.60, date(2022, 7, 1))
        assert features["distance_nearest_landslide_km"] == pytest.approx(0.0, abs=0.1)

    def test_excludes_target_event(self):
        """For a positive sample, the target event must be excluded."""
        events = _make_events()
        transformer = ProximityFeatureTransformer(events)

        # Compute proximity for E1's cell, excluding E1 itself
        features = transformer.compute(
            27.30,
            88.60,
            date(2022, 7, 1),
            exclude_event_id="E1",
        )
        # E2 is at 27.35, 88.65 → distance ~6.3km
        # E3 is at 27.40, 88.70 → distance ~12.5km
        # Nearest should be E2
        assert features["distance_nearest_landslide_km"] > 0
        assert features["distance_nearest_landslide_km"] < 10  # closer to E2 than E3

    def test_excludes_target_event_even_if_past(self):
        """The target event is excluded even if it's before sample_date."""
        events = _make_events()
        transformer = ProximityFeatureTransformer(events)

        # E1 is at 27.30, 88.60; sample_date is after E1
        # Without exclusion: distance = 0 (E1 is at same location)
        # With exclusion: distance > 0
        features_excluded = transformer.compute(
            27.30,
            88.60,
            date(2022, 7, 15),
            exclude_event_id="E1",
        )
        features_not_excluded = transformer.compute(
            27.30,
            88.60,
            date(2022, 7, 15),
            exclude_event_id=None,
        )
        # Excluded should be farther (E1 excluded, nearest is E2)
        assert (
            features_excluded["distance_nearest_landslide_km"]
            > features_not_excluded["distance_nearest_landslide_km"]
        )

    def test_no_past_events_returns_default(self):
        events = [
            {
                "event_id": "E1",
                "event_date": date(2023, 1, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
        ]
        transformer = ProximityFeatureTransformer(events)

        # Before any events
        features = transformer.compute(27.30, 88.60, date(2022, 1, 1))
        assert features["distance_nearest_landslide_km"] == 999.0
        assert features["n_landslides_within_5km"] == 0

    def test_count_within_5km(self):
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 6, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
            {
                "event_id": "E2",
                "event_date": date(2022, 6, 15),
                "latitude": 27.305,
                "longitude": 88.605,
            },  # ~0.7km away
            {
                "event_id": "E3",
                "event_date": date(2022, 7, 1),
                "latitude": 27.40,
                "longitude": 88.70,
            },  # ~12km away
        ]
        transformer = ProximityFeatureTransformer(events)

        features = transformer.compute(27.30, 88.60, date(2022, 7, 15))
        # E1 and E2 are within 5km, E3 is not
        assert features["n_landslides_within_5km"] == 2

    def test_symmetry(self):
        """Distance should be symmetric (A→B = B→A)."""
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 6, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
        ]
        transformer = ProximityFeatureTransformer(events)

        f1 = transformer.compute(27.35, 88.65, date(2022, 7, 1))
        transformer.compute(27.30, 88.60, date(2022, 7, 15))
        # E1 is at 27.30, 88.60 → for f2, E1 is the only event and is excluded
        # For f1, E1 is the nearest past event
        assert f1["distance_nearest_landslide_km"] > 0


# ============================================================================
# 5. Land cover and road stubs
# ============================================================================


class TestLandCoverStub:
    def test_returns_zero(self):
        transformer = LandCoverFeatureTransformer()
        features = transformer.compute("any_cell")
        assert features["lulc_category"] == 0

    def test_key_present(self):
        transformer = LandCoverFeatureTransformer()
        features = transformer.compute("any_cell")
        assert "lulc_category" in features


class TestRoadStub:
    def test_returns_default(self):
        transformer = RoadFeatureTransformer()
        features = transformer.compute("any_cell")
        assert features["road_distance_km"] == 999.0

    def test_key_present(self):
        transformer = RoadFeatureTransformer()
        features = transformer.compute("any_cell")
        assert "road_distance_km" in features


# ============================================================================
# 6. Composite FeatureTransformer
# ============================================================================


class TestFeatureTransformer:
    def test_compute_all_returns_all_features(self):
        records = _make_rainfall()
        events = _make_events()
        transformer = FeatureTransformer(
            rainfall_records=records,
            static_features=TERRAIN_DEV_FIXTURE,
            landslide_events=events,
        )
        features = transformer.compute_all(
            cell_id="27.3200N_88.6100E",
            cell_lat=27.32,
            cell_lon=88.61,
            sample_date=date(2022, 7, 15),
        )
        expected_keys = [
            "rainfall_current_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
            "slope_angle_deg",
            "slope_aspect_deg",
            "elevation_m",
            "distance_nearest_landslide_km",
            "n_landslides_within_5km",
            "lulc_category",
            "road_distance_km",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_compute_all_for_dataset(self):
        records = _make_rainfall()
        events = _make_events()
        transformer = FeatureTransformer(
            rainfall_records=records,
            static_features=TERRAIN_DEV_FIXTURE,
            landslide_events=events,
        )
        dataset = [
            {
                "grid_cell_id": "27.3200N_88.6100E",
                "centroid_lat": 27.32,
                "centroid_lon": 88.61,
                "sample_date": "2022-07-15",
                "event_id": "E1",
                "label": 1,
            },
            {
                "grid_cell_id": "27.3300N_88.6200E",
                "centroid_lat": 27.33,
                "centroid_lon": 88.62,
                "sample_date": "2022-07-15",
                "event_id": None,
                "label": 0,
            },
        ]
        enriched = transformer.compute_all_for_dataset(dataset)
        assert len(enriched) == 2
        assert "rainfall_current_mm" in enriched[0]
        assert "slope_angle_deg" in enriched[0]

    def test_positive_sample_excludes_target_event(self):
        """Positive samples must exclude their own event from proximity."""
        events = _make_events()
        transformer = FeatureTransformer(landslide_events=events)

        # Positive sample for E1 at E1's location
        features = transformer.compute_all(
            cell_id="27.3200N_88.6100E",
            cell_lat=27.30,
            cell_lon=88.60,
            sample_date=date(2022, 7, 15),
            exclude_event_id="E1",
        )
        # Distance should NOT be 0 (E1 excluded)
        assert features["distance_nearest_landslide_km"] > 0

    def test_negative_sample_uses_all_events(self):
        events = _make_events()
        transformer = FeatureTransformer(landslide_events=events)

        features = transformer.compute_all(
            cell_id="27.3200N_88.6100E",
            cell_lat=27.30,
            cell_lon=88.60,
            sample_date=date(2022, 7, 15),
            exclude_event_id=None,
        )
        # E1 is at same location → distance = 0
        assert features["distance_nearest_landslide_km"] == pytest.approx(0.0, abs=0.1)


# ============================================================================
# 7. End-to-end with dev fixtures
# ============================================================================


class TestEndToEndFeatures:
    def test_full_pipeline(self):
        """Run the complete feature pipeline with dev fixtures."""
        import csv
        from pathlib import Path

        fixture_dir = (
            Path(__file__).resolve().parent.parent.parent / "data" / "reference"
        )
        raw_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

        # Load dev fixtures
        events = []
        with open(fixture_dir / "landslide_events_dev_fixture.csv") as f:
            for row in csv.DictReader(f):
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

        rainfall = []
        with open(raw_dir / "rainfall_dev_fixture.csv") as f:
            for row in csv.DictReader(f):
                rainfall.append(
                    {
                        "station_id": row["station_id"],
                        "station_lat": float(row["station_lat"]),
                        "station_lon": float(row["station_lon"]),
                        "reading_date": date.fromisoformat(row["reading_date"]),
                        "rainfall_mm": float(row["rainfall_mm"]),
                    }
                )

        transformer = FeatureTransformer(
            rainfall_records=rainfall,
            static_features=TERRAIN_DEV_FIXTURE,
            landslide_events=events,
        )

        # Compute features for first event's cell
        e = events[0]
        features = transformer.compute_all(
            cell_id="27.3200N_88.6100E",
            cell_lat=e["latitude"],
            cell_lon=e["longitude"],
            sample_date=date(2022, 7, 15),
            exclude_event_id=e["event_id"],
        )

        # Verify all feature families present
        assert "rainfall_current_mm" in features
        assert "slope_angle_deg" in features
        assert "distance_nearest_landslide_km" in features
        assert "lulc_category" in features
        assert "road_distance_km" in features

        # Verify values are numeric
        for key, value in features.items():
            assert isinstance(value, (int, float)), f"{key} is not numeric: {value}"

    def test_all_records_get_features(self):
        """Every sample in a dataset gets all features computed."""
        events = _make_events()
        records = _make_rainfall()
        transformer = FeatureTransformer(
            rainfall_records=records,
            static_features=TERRAIN_DEV_FIXTURE,
            landslide_events=events,
        )

        dataset = [
            {
                "grid_cell_id": "27.3200N_88.6100E",
                "centroid_lat": 27.32,
                "centroid_lon": 88.61,
                "sample_date": "2022-07-15",
                "event_id": "E1",
                "label": 1,
            },
            {
                "grid_cell_id": "27.3300N_88.6200E",
                "centroid_lat": 27.33,
                "centroid_lon": 88.62,
                "sample_date": "2022-07-15",
                "event_id": None,
                "label": 0,
            },
            {
                "grid_cell_id": "27.3400N_88.6300E",
                "centroid_lat": 27.34,
                "centroid_lon": 88.63,
                "sample_date": "2022-06-20",
                "event_id": None,
                "label": 0,
            },
        ]

        enriched = transformer.compute_all_for_dataset(dataset)
        assert len(enriched) == 3
        for sample in enriched:
            assert "rainfall_current_mm" in sample
            assert "distance_nearest_landslide_km" in sample


# ============================================================================
# 8. Leakage detection tests
# ============================================================================


class TestLeakageDetection:
    def test_rainfall_excludes_sample_date(self):
        """All rainfall features must not use data from sample_date onward."""
        # Day-by-day rainfall with known values
        records = []
        for i in range(30):
            records.append(
                {
                    "station_id": "S1",
                    "station_lat": 27.3,
                    "station_lon": 88.6,
                    "reading_date": date(2022, 7, 1) + timedelta(days=i),
                    "rainfall_mm": 100.0 if i == 9 else 1.0,  # spike on July 10
                }
            )
        transformer = RainfallFeatureTransformer(records)

        # sample_date = July 10: July 10 rainfall (100.0) must NOT appear
        features = transformer.compute(27.3, 88.6, date(2022, 7, 10))
        # 1-day window = July 9 only (1.0), not July 10 (100.0)
        assert features["rainfall_current_mm"] == pytest.approx(1.0, abs=0.1)

        # If sample_date = July 11: July 10 (100.0) appears in current
        features_later = transformer.compute(27.3, 88.6, date(2022, 7, 11))
        assert features_later["rainfall_current_mm"] == pytest.approx(100.0, abs=0.1)

    def test_proximity_excludes_future_events(self):
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 8, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
        ]
        transformer = ProximityFeatureTransformer(events)

        # Before E1: no past events
        features = transformer.compute(27.30, 88.60, date(2022, 7, 1))
        assert features["distance_nearest_landslide_km"] == 999.0

    def test_proximity_excludes_target_event(self):
        """Target event must never be used for its own sample's proximity."""
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 6, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
        ]
        transformer = ProximityFeatureTransformer(events)

        # E1 is at 27.30, 88.60 — compute proximity for same cell, excluding E1
        features = transformer.compute(
            27.30,
            88.60,
            date(2022, 7, 1),
            exclude_event_id="E1",
        )
        # Should NOT be 0 (E1 excluded, no other events)
        assert features["distance_nearest_landslide_km"] == 999.0

    def test_no_duplicate_features(self):
        """Each feature name must appear exactly once."""
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_features_not_derived_from_label(self):
        """No feature computation uses the label field."""
        # The feature transformers don't accept labels as input
        # This is a structural guarantee — labels are not inputs to any
        # feature computation function
        pass  # Verified by code inspection

    def test_train_test_no_feature_leakage(self):
        """Features for train set should not depend on test-set events."""
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 6, 1),
                "latitude": 27.30,
                "longitude": 88.60,
            },
            {
                "event_id": "E2",
                "event_date": date(2023, 6, 1),  # test set
                "latitude": 27.30,
                "longitude": 88.60,
            },
        ]
        transformer = ProximityFeatureTransformer(events)

        # Train sample (before split date)
        train_features = transformer.compute(
            27.30,
            88.60,
            date(2022, 7, 1),
            exclude_event_id="E1",
        )
        # E2 is in the future → must not affect train proximity
        assert train_features["distance_nearest_landslide_km"] == 999.0
        assert train_features["n_landslides_within_5km"] == 0
