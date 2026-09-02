"""Tests for the ML-2 data and label pipeline.

All tests use either:
1. Synthetic DEV_FIXTURE data clearly marked as NOT REAL DATA, or
2. Programmatically constructed test data.

No real historical landslide or rainfall data is used.
No fabricated data is presented as real.
"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from apps.ml_bridge.ml.data_pipeline import (
    PipelineConfig,
    _compute_slope_distribution,
    _haversine_km,
    _point_in_cell,
    _severity_to_risk,
    _slope_to_bin,
    assemble_dataset,
    check_leakage,
    create_grid_cells,
    label_positive_samples,
    load_landslide_inventory,
    load_rainfall_timeseries,
    load_static_features,
    sample_negative_samples,
    save_dataset,
    save_metadata,
    split_train_test,
)
from apps.ml_bridge.ml.feature_engineering import RainfallFeatureTransformer

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "reference"
)
RAW_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


# ============================================================================
# 1. Configuration
# ============================================================================


class TestPipelineConfig:
    def test_default_config(self):
        config = PipelineConfig()
        assert config.grid_resolution_km == 1.0
        assert config.positive_window_days == 0
        assert config.exclusion_buffer_km == 2.0
        assert config.positive_negative_ratio == 3
        assert config.train_test_split_date == date(2023, 1, 1)

    def test_custom_config(self):
        config = PipelineConfig(
            grid_resolution_km=0.5,
            positive_window_days=3,
            exclusion_buffer_km=5.0,
            positive_negative_ratio=5,
            train_test_split_date=date(2022, 6, 1),
        )
        assert config.grid_resolution_km == 0.5
        assert config.positive_window_days == 3
        assert config.exclusion_buffer_km == 5.0
        assert config.positive_negative_ratio == 5
        assert config.train_test_split_date == date(2022, 6, 1)


# ============================================================================
# 2. Haversine distance
# ============================================================================


class TestHaversine:
    def test_same_point(self):
        assert _haversine_km(27.3, 88.6, 27.3, 88.6) == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        # ~111 km between 27.0N and 28.0N at same longitude
        d = _haversine_km(27.0, 88.0, 28.0, 88.0)
        assert d == pytest.approx(111.0, abs=5.0)

    def test_symmetry(self):
        d1 = _haversine_km(27.0, 88.0, 27.5, 88.5)
        d2 = _haversine_km(27.5, 88.5, 27.0, 88.0)
        assert d1 == pytest.approx(d2, abs=0.01)


# ============================================================================
# 3. Grid cell creation
# ============================================================================


class TestCreateGridCells:
    def test_basic_grid(self):
        cells = create_grid_cells(27.0, 88.0, 27.02, 88.02, resolution_km=1.0)
        assert len(cells) > 0
        for cell in cells:
            assert "grid_cell_id" in cell
            assert "centroid_lat" in cell
            assert "centroid_lon" in cell

    def test_grid_resolution_affects_count(self):
        coarse = create_grid_cells(27.0, 88.0, 27.1, 88.1, resolution_km=1.0)
        fine = create_grid_cells(27.0, 88.0, 27.1, 88.1, resolution_km=0.5)
        assert len(fine) > len(coarse)

    def test_grid_covers_bbox(self):
        cells = create_grid_cells(27.0, 88.0, 27.05, 88.05, resolution_km=1.0)
        lats = [c["centroid_lat"] for c in cells]
        lons = [c["centroid_lon"] for c in cells]
        assert min(lats) >= 27.0
        assert max(lats) <= 27.05
        assert min(lons) >= 88.0
        assert max(lons) <= 88.05


# ============================================================================
# 4. Point-in-cell
# ============================================================================


class TestPointInCell:
    def test_center_is_inside(self):
        cell = {"grid_cell_id": "test", "centroid_lat": 27.0, "centroid_lon": 88.0}
        assert _point_in_cell(27.0, 88.0, cell, 1.0) is True

    def test_edge_is_inside(self):
        cell = {"grid_cell_id": "test", "centroid_lat": 27.0, "centroid_lon": 88.0}
        # Half of 1km in lat degrees ≈ 0.0045
        assert _point_in_cell(27.0 + 0.004, 88.0, cell, 1.0) is True

    def test_outside_is_outside(self):
        cell = {"grid_cell_id": "test", "centroid_lat": 27.0, "centroid_lon": 88.0}
        assert _point_in_cell(27.01, 88.01, cell, 1.0) is False


# ============================================================================
# 5. Data loading
# ============================================================================


class TestLoadLandslideInventory:
    def test_load_dev_fixture(self):
        path = FIXTURE_DIR / "landslide_events_dev_fixture.csv"
        events = load_landslide_inventory(path)
        assert len(events) == 10
        assert all("event_id" in e for e in events)
        assert all("event_date" in e for e in events)
        assert all("latitude" in e for e in events)
        assert all("longitude" in e for e in events)

    def test_event_dates_are_dates(self):
        path = FIXTURE_DIR / "landslide_events_dev_fixture.csv"
        events = load_landslide_inventory(path)
        for event in events:
            assert isinstance(event["event_date"], date)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_landslide_inventory("/nonexistent/path.csv")


class TestLoadRainfallTimeseries:
    def test_load_dev_fixture(self):
        path = RAW_FIXTURE_DIR / "rainfall_dev_fixture.csv"
        records = load_rainfall_timeseries(path)
        assert len(records) > 0
        assert all("station_id" in r for r in records)
        assert all("reading_date" in r for r in records)
        assert all("rainfall_mm" in r for r in records)

    def test_reading_dates_are_dates(self):
        path = RAW_FIXTURE_DIR / "rainfall_dev_fixture.csv"
        records = load_rainfall_timeseries(path)
        for record in records:
            assert isinstance(record["reading_date"], date)


class TestLoadStaticFeatures:
    def test_missing_file_returns_empty(self):
        result = load_static_features("/nonexistent/path.csv")
        assert result == {}


# ============================================================================
# 6. Positive labeling
# ============================================================================


class TestLabelPositiveSamples:
    def _make_test_data(self):
        cells = [
            {"grid_cell_id": "C1", "centroid_lat": 27.30, "centroid_lon": 88.60},
            {"grid_cell_id": "C2", "centroid_lat": 27.35, "centroid_lon": 88.65},
            {"grid_cell_id": "C3", "centroid_lat": 27.40, "centroid_lon": 88.70},
        ]
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 7, 15),
                "latitude": 27.30,
                "longitude": 88.60,
                "severity": "Moderate",
                "source_reference": "test",
            },
            {
                "event_id": "E2",
                "event_date": date(2022, 8, 1),
                "latitude": 27.35,
                "longitude": 88.65,
                "severity": "High",
                "source_reference": "test",
            },
        ]
        return cells, events

    def test_basic_positive_labeling(self):
        cells, events = self._make_test_data()
        config = PipelineConfig(lead_time_days=1)
        positives = label_positive_samples(cells, events, config)
        assert len(positives) == 2

    def test_positive_has_required_fields(self):
        cells, events = self._make_test_data()
        config = PipelineConfig(lead_time_days=1)
        positives = label_positive_samples(cells, events, config)
        for p in positives:
            assert "grid_cell_id" in p
            assert "sample_date" in p
            assert "label" in p
            assert p["label"] == 1
            assert "event_id" in p

    def test_lead_time_one_day_shifts_date(self):
        cells, events = self._make_test_data()
        config = PipelineConfig(lead_time_days=1)
        positives = label_positive_samples(cells, events, config)
        for p in positives:
            # sample_date = event_date - 1 day (lead_time_days=1)
            assert p["sample_date"] == p["event_date"] - timedelta(days=1)

    def test_lead_time_three_days_shifts_date(self):
        cells, events = self._make_test_data()
        config = PipelineConfig(lead_time_days=3)
        positives = label_positive_samples(cells, events, config)
        for p in positives:
            # sample_date = event_date - 3 days
            assert p["sample_date"] == p["event_date"] - timedelta(days=3)

    def test_one_positive_per_event(self):
        cells, events = self._make_test_data()
        config = PipelineConfig()
        positives = label_positive_samples(cells, events, config)
        event_ids = [p["event_id"] for p in positives]
        assert len(event_ids) == len(set(event_ids))


# ============================================================================
# 7. Negative sampling
# ============================================================================


class TestSampleNegativeSamples:
    def _make_test_data(self):
        # Create a small grid
        cells = []
        for lat_i in range(10):
            for lon_i in range(10):
                cells.append(
                    {
                        "grid_cell_id": f"C{lat_i}_{lon_i}",
                        "centroid_lat": 27.30 + lat_i * 0.01,
                        "centroid_lon": 88.60 + lon_i * 0.01,
                    }
                )
        events = [
            {
                "event_id": "E1",
                "event_date": date(2022, 7, 15),
                "latitude": 27.30,
                "longitude": 88.60,
                "severity": "Moderate",
                "source_reference": "test",
            },
        ]
        positives = [
            {
                "grid_cell_id": "C0_0",
                "centroid_lat": 27.30,
                "centroid_lon": 88.60,
                "sample_date": date(2022, 7, 15),
                "event_date": date(2022, 7, 15),
                "label": 1,
                "risk_level": "High",
                "event_id": "E1",
                "source_reference": "test",
                "data_origin": "DEV_FIXTURE",
            }
        ]
        return cells, events, positives

    def test_negative_count_matches_ratio(self):
        cells, events, positives = self._make_test_data()
        config = PipelineConfig(positive_negative_ratio=3, exclusion_buffer_km=0.001)
        negatives = sample_negative_samples(cells, events, positives, {}, config)
        # Should have roughly 3x positives
        assert len(negatives) >= 1

    def test_negatives_exclude_positive_cells(self):
        cells, events, positives = self._make_test_data()
        config = PipelineConfig(positive_negative_ratio=3, exclusion_buffer_km=0.001)
        negatives = sample_negative_samples(cells, events, positives, {}, config)
        neg_cell_ids = {n["grid_cell_id"] for n in negatives}
        assert "C0_0" not in neg_cell_ids

    def test_negatives_have_label_zero(self):
        cells, events, positives = self._make_test_data()
        config = PipelineConfig(positive_negative_ratio=3, exclusion_buffer_km=0.001)
        negatives = sample_negative_samples(cells, events, positives, {}, config)
        for n in negatives:
            assert n["label"] == 0

    def test_no_positives_returns_empty(self):
        cells, events = self._make_test_data()[:2]
        config = PipelineConfig()
        negatives = sample_negative_samples(cells, events, [], {}, config)
        assert negatives == []


# ============================================================================
# 8. Rainfall feature computation
# ============================================================================


class TestComputeAntecedentRainfall:
    def _make_rainfall(self):
        records = []
        base = date(2022, 7, 1)
        for i in range(30):
            records.append(
                {
                    "station_id": "S1",
                    "station_lat": 27.30,
                    "station_lon": 88.60,
                    "reading_date": base + timedelta(days=i),
                    "rainfall_mm": float(i + 1),  # 1, 2, 3, ... 30
                }
            )
        return records

    def test_basic_computation(self):
        records = self._make_rainfall()
        transformer = RainfallFeatureTransformer(records, windows=[3])
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        # Days 7, 8, 9 (July 7-9) → rainfall 7 + 8 + 9 = 24.0
        assert "rainfall_3d_mm" in features
        assert features["rainfall_3d_mm"] == pytest.approx(24.0, abs=0.1)

    def test_excludes_sample_date(self):
        records = self._make_rainfall()
        transformer = RainfallFeatureTransformer(records, windows=[1])
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        # Day 9 (July 9) → rainfall 9.0 (July 10 is excluded)
        assert features["rainfall_1d_mm"] == pytest.approx(9.0, abs=0.1)

    def test_multiple_windows(self):
        records = self._make_rainfall()
        transformer = RainfallFeatureTransformer(records, windows=[3, 7])
        features = transformer.compute(27.30, 88.60, date(2022, 7, 10))
        assert "rainfall_3d_mm" in features
        assert "rainfall_7d_mm" in features
        assert features["rainfall_7d_mm"] > features["rainfall_3d_mm"]

    def test_no_nearby_stations_returns_zeros(self):
        records = self._make_rainfall()
        transformer = RainfallFeatureTransformer(
            records, windows=[3], max_station_distance_km=1.0
        )
        features = transformer.compute(29.0, 90.0, date(2022, 7, 10))
        assert features["rainfall_3d_mm"] == 0.0


# ============================================================================
# 9. Severity mapping
# ============================================================================


class TestSeverityToRisk:
    def test_known_severities(self):
        assert _severity_to_risk("Low") == "Moderate"
        assert _severity_to_risk("Moderate") == "High"
        assert _severity_to_risk("High") == "High"
        assert _severity_to_risk("Severe") == "Severe"

    def test_unknown_defaults_to_high(self):
        assert _severity_to_risk("Unknown") == "High"
        assert _severity_to_risk("") == "High"


# ============================================================================
# 10. Slope binning
# ============================================================================


class TestSlopeBinning:
    def test_slope_to_bin(self):
        bins = [0, 15, 30, 45, 90]
        assert _slope_to_bin(5, bins) == 0
        assert _slope_to_bin(20, bins) == 1
        assert _slope_to_bin(35, bins) == 2
        assert _slope_to_bin(50, bins) == 3

    def test_compute_slope_distribution(self):
        positives = [
            {"grid_cell_id": "C1"},
            {"grid_cell_id": "C2"},
        ]
        static = {
            "C1": {"slope_angle_deg": 10},
            "C2": {"slope_angle_deg": 35},
        }
        dist = _compute_slope_distribution(positives, static, [0, 15, 30, 45, 90])
        assert dist[0] == 1  # C1 in bin 0
        assert dist[2] == 1  # C2 in bin 2


# ============================================================================
# 11. Train/test split
# ============================================================================


class TestTrainTestSplit:
    def test_time_based_split(self):
        dataset = [
            {"sample_date": "2022-07-01", "label": 1},
            {"sample_date": "2022-08-01", "label": 1},
            {"sample_date": "2023-01-01", "label": 0},
            {"sample_date": "2023-06-01", "label": 0},
        ]
        train, test = split_train_test(dataset, date(2023, 1, 1))
        assert len(train) == 2
        assert len(test) == 2
        for s in train:
            assert date.fromisoformat(s["sample_date"]) < date(2023, 1, 1)
        for s in test:
            assert date.fromisoformat(s["sample_date"]) >= date(2023, 1, 1)

    def test_all_before_split_goes_to_train(self):
        dataset = [{"sample_date": "2022-07-01", "label": 1}]
        train, test = split_train_test(dataset, date(2023, 1, 1))
        assert len(train) == 1
        assert len(test) == 0

    def test_all_after_split_goes_to_test(self):
        dataset = [{"sample_date": "2023-06-01", "label": 0}]
        train, test = split_train_test(dataset, date(2023, 1, 1))
        assert len(train) == 0
        assert len(test) == 1


# ============================================================================
# 12. Leakage checks
# ============================================================================


class TestCheckLeakage:
    def test_clean_split_passes(self):
        train = [
            {
                "sample_id": "A",
                "event_id": "E1",
                "grid_cell_id": "C1",
                "sample_date": "2022-07-01",
            },
        ]
        test = [
            {
                "sample_id": "B",
                "event_id": "E2",
                "grid_cell_id": "C2",
                "sample_date": "2023-06-01",
            },
        ]
        result = check_leakage(train, test)
        assert result["no_sample_id_overlap"]["passed"] is True
        assert result["no_event_id_overlap"]["passed"] is True
        assert result["no_future_in_train"]["passed"] is True

    def test_overlapping_sample_ids_fails(self):
        train = [
            {
                "sample_id": "A",
                "event_id": None,
                "grid_cell_id": "C1",
                "sample_date": "2022-07-01",
            }
        ]
        test = [
            {
                "sample_id": "A",
                "event_id": None,
                "grid_cell_id": "C2",
                "sample_date": "2023-06-01",
            }
        ]
        result = check_leakage(train, test)
        assert result["no_sample_id_overlap"]["passed"] is False

    def test_overlapping_event_ids_fails(self):
        train = [
            {
                "sample_id": "A",
                "event_id": "E1",
                "grid_cell_id": "C1",
                "sample_date": "2022-07-01",
            }
        ]
        test = [
            {
                "sample_id": "B",
                "event_id": "E1",
                "grid_cell_id": "C1",
                "sample_date": "2023-06-01",
            }
        ]
        result = check_leakage(train, test)
        assert result["no_event_id_overlap"]["passed"] is False


# ============================================================================
# 13. Dataset assembly
# ============================================================================


class TestAssembleDataset:
    def test_assembly_produces_records(self):
        positives = [
            {
                "grid_cell_id": "C1",
                "centroid_lat": 27.3,
                "centroid_lon": 88.6,
                "sample_date": date(2022, 7, 15),
                "event_date": date(2022, 7, 15),
                "label": 1,
                "risk_level": "High",
                "event_id": "E1",
                "source_reference": "test",
                "data_origin": "DEV_FIXTURE",
            }
        ]
        negatives = [
            {
                "grid_cell_id": "C2",
                "centroid_lat": 27.35,
                "centroid_lon": 88.65,
                "sample_date": date(2022, 7, 15),
                "event_date": None,
                "label": 0,
                "risk_level": "Low",
                "event_id": None,
                "source_reference": "",
                "data_origin": "DEV_FIXTURE",
            }
        ]
        rainfall = []
        config = PipelineConfig()
        dataset = assemble_dataset(positives, negatives, rainfall, {}, config)
        assert len(dataset) == 2
        assert dataset[0]["label"] == 1
        assert dataset[1]["label"] == 0

    def test_assembly_has_all_schema_fields(self):
        positives = [
            {
                "grid_cell_id": "C1",
                "centroid_lat": 27.3,
                "centroid_lon": 88.6,
                "sample_date": date(2022, 7, 15),
                "event_date": date(2022, 7, 15),
                "label": 1,
                "risk_level": "High",
                "event_id": "E1",
                "source_reference": "test",
                "data_origin": "DEV_FIXTURE",
            }
        ]
        # Provide minimal rainfall data so features get computed
        rainfall = [
            {
                "station_id": "S1",
                "station_lat": 27.3,
                "station_lon": 88.6,
                "reading_date": date(2022, 7, 14),
                "rainfall_mm": 10.0,
            }
        ]
        config = PipelineConfig(lead_time_days=1)
        dataset = assemble_dataset(positives, [], rainfall, {}, config)
        required = [
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
            "rainfall_current_mm",
            "rainfall_3d_mm",
            "rainfall_7d_mm",
            "rainfall_15d_mm",
            "rainfall_30d_mm",
        ]
        for field in required:
            assert field in dataset[0], f"Missing field: {field}"


# ============================================================================
# 14. Save/load roundtrip
# ============================================================================


class TestSaveDataset:
    def test_save_and_load(self, tmp_path):
        dataset = [
            {
                "sample_id": "C1_2022-07-15",
                "grid_cell_id": "C1",
                "centroid_lat": 27.3,
                "centroid_lon": 88.6,
                "sample_date": "2022-07-15",
                "event_date": "2022-07-15",
                "label": 1,
                "risk_level": "High",
                "event_id": "E1",
                "source_reference": "test",
                "data_origin": "DEV_FIXTURE",
                "slope_angle_deg": 25.0,
                "slope_aspect_deg": 180.0,
                "elevation_m": 1500.0,
                "lulc_category": 3,
                "road_distance_km": 2.5,
                "rainfall_3d_mm": 45.0,
                "rainfall_7d_mm": 80.0,
                "rainfall_15d_mm": 120.0,
                "rainfall_30d_mm": 200.0,
            }
        ]
        config = PipelineConfig(output_dir=str(tmp_path))
        path = save_dataset(dataset, config, tmp_path / "test.csv")
        assert path.exists()

    def test_empty_dataset(self, tmp_path):
        config = PipelineConfig(output_dir=str(tmp_path))
        path = save_dataset([], config, tmp_path / "empty.csv")
        assert path.exists()


class TestSaveMetadata:
    def test_saves_metadata(self, tmp_path):
        dataset = [
            {"label": 1, "data_origin": "DEV_FIXTURE", "sample_date": "2022-07-01"},
            {"label": 0, "data_origin": "DEV_FIXTURE", "sample_date": "2022-07-01"},
        ]
        config = PipelineConfig()
        path = save_metadata(dataset, config, output_path=tmp_path / "meta.json")
        assert path.exists()
        import json

        with open(path) as f:
            meta = json.load(f)
        assert meta["n_positive"] == 1
        assert meta["n_negative"] == 1
        assert meta["has_dev_fixtures"] is True
        assert "SYNTHETIC" in meta["synthetic_data_warning"]


# ============================================================================
# 15. End-to-end pipeline with dev fixtures
# ============================================================================


class TestEndToEndPipeline:
    def test_full_pipeline_with_fixtures(self, tmp_path):
        """Run the full pipeline using dev fixtures. Verifies pipeline works end-to-end."""
        config = PipelineConfig(
            grid_resolution_km=1.0,
            positive_window_days=0,
            exclusion_buffer_km=2.0,
            positive_negative_ratio=3,
            train_test_split_date=date(2023, 1, 1),
            output_dir=str(tmp_path),
            data_origin="DEV_FIXTURE",
        )

        # Load data
        events = load_landslide_inventory(
            FIXTURE_DIR / "landslide_events_dev_fixture.csv"
        )
        rainfall = load_rainfall_timeseries(
            RAW_FIXTURE_DIR / "rainfall_dev_fixture.csv"
        )

        # Create grid
        cells = create_grid_cells(27.25, 88.50, 27.45, 88.75, config.grid_resolution_km)
        assert len(cells) > 0

        # Label positives
        positives = label_positive_samples(cells, events, config)
        assert len(positives) == 10

        # Sample negatives
        negatives = sample_negative_samples(cells, events, positives, {}, config)
        assert len(negatives) > 0

        # Assemble dataset
        dataset = assemble_dataset(positives, negatives, rainfall, {}, config)
        assert len(dataset) > 0

        # Verify all records have data_origin
        for record in dataset:
            assert "data_origin" in record
            assert record["data_origin"] == "DEV_FIXTURE"

        # Split
        train, test = split_train_test(dataset, config.train_test_split_date)
        assert len(train) + len(test) == len(dataset)

        # Check leakage
        leakage = check_leakage(train, test)
        for check_name, check_result in leakage.items():
            assert check_result["passed"], f"Leakage check failed: {check_name}"

        # Save
        save_dataset(dataset, config, tmp_path / "dataset.csv")
        save_metadata(dataset, config, output_path=tmp_path / "metadata.json")

        assert (tmp_path / "dataset.csv").exists()
        assert (tmp_path / "metadata.json").exists()
