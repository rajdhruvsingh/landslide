"""Unified data ingestion pipeline for ML feature generation.

This module orchestrates all data sources to produce the complete feature set
for the landslide risk classifier.

Pipeline stages:
1. Define spatial/temporal scope (bbox, date range)
2. Fetch rainfall (IMD/NASA POWER)
3. Fetch landslide inventory (GSI/COOLR)
4. Fetch terrain (SRTM DEM -> slope, aspect, elevation)
5. Fetch roads (OSM Overpass -> distance, density)
6. Fetch soil moisture (SMAP/NASA POWER)
7. Fetch land cover (ESA WorldCover)
8. Build grid cells
9. Label positives/negatives (with lead_time_days)
10. Compute all features per sample
11. Save dataset with metadata
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


from apps.ml_bridge.ml.data_pipeline import (
    PipelineConfig,
    assemble_dataset,
    create_grid_cells,
    label_positive_samples,
    sample_negative_samples,
    save_dataset,
    save_metadata,
    split_train_test,
    walk_forward_split,
    get_class_weights,
)
from apps.ml_bridge.ml.feature_engineering import FeatureTransformer
from apps.risk_zones.ingestion.dem_loader import SRTMClient
from apps.risk_zones.ingestion.gsi_client import GSIClient
from apps.risk_zones.ingestion.lulc_client import WorldCoverClient
from apps.risk_zones.ingestion.osm_road_client import OSMRoadClient
from apps.weather.ingestion.imd_client import IMDClient
from apps.weather.ingestion.smap_client import SoilMoistureClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for the full ingestion pipeline."""

    # Spatial bounds
    min_lat: float = 27.0
    min_lon: float = 88.0
    max_lat: float = 28.0
    max_lon: float = 89.0

    # Temporal bounds
    start_date: date = field(default_factory=lambda: date(2020, 1, 1))
    end_date: date = field(default_factory=lambda: date(2023, 12, 31))

    # Grid
    grid_resolution_km: float = 1.0

    # Forecasting
    lead_time_days: int = 1

    # Sampling
    exclusion_buffer_km: float = 2.0
    positive_negative_ratio: int = 3

    # Data sources (enable/disable)
    use_imd_rainfall: bool = True
    use_nasa_power_fallback: bool = True
    use_gsi_inventory: bool = True
    use_coolr_fallback: bool = True
    use_srtm_dem: bool = True
    use_osm_roads: bool = True
    use_soil_moisture: bool = True
    use_land_cover: bool = True

    # Output
    output_dir: str = "data/processed"
    cache_dir: str = "data/cache"

    # Advanced
    include_tracks_in_roads: bool = False
    lulc_buffer_m: int = 500


class IngestionPipeline:
    """Full data ingestion and feature generation pipeline."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize clients
        self.imd_client = IMDClient(
            use_fallback=config.use_nasa_power_fallback,
            cache_dir=Path(config.cache_dir) / "imd",
        )
        self.gsi_client = GSIClient(
            use_coolr=config.use_coolr_fallback,
        )
        self.srtm_client = SRTMClient(
            cache_dir=Path(config.cache_dir) / "srtm",
        )
        self.osm_client = OSMRoadClient(
            cache_dir=Path(config.cache_dir) / "osm",
        )
        self.smap_client = SoilMoistureClient(
            use_power_fallback=config.use_nasa_power_fallback,
        )
        self.lulc_client = WorldCoverClient(
            cache_dir=Path(config.cache_dir) / "worldcover",
        )

    async def run(self) -> dict[str, Any]:
        """Run the complete ingestion pipeline.

        Returns:
            Dict with dataset paths and metadata.
        """
        logger.info("Starting ingestion pipeline")
        logger.info(
            f"Bounds: ({self.config.min_lat}, {self.config.min_lon}) - "
            f"({self.config.max_lat}, {self.config.max_lon})"
        )
        logger.info(f"Date range: {self.config.start_date} to {self.config.end_date}")

        # Stage 1: Create grid cells
        logger.info("Creating grid cells...")
        grid_cells = create_grid_cells(
            self.config.min_lat,
            self.config.min_lon,
            self.config.max_lat,
            self.config.max_lon,
            self.config.grid_resolution_km,
        )
        logger.info(f"Created {len(grid_cells)} grid cells")

        # Stage 2: Fetch landslide inventory
        logger.info("Fetching landslide inventory...")
        events = await self.gsi_client.fetch_inventory(
            self.config.min_lat,
            self.config.min_lon,
            self.config.max_lat,
            self.config.max_lon,
            self.config.start_date,
            self.config.end_date,
        )
        logger.info(f"Fetched {len(events)} landslide events")

        if not events:
            logger.warning(
                "No landslide events found - pipeline cannot create positive samples"
            )
            return {"status": "error", "message": "No events found"}

        # Stage 3: Label positive samples
        pipeline_config = PipelineConfig(
            grid_resolution_km=self.config.grid_resolution_km,
            lead_time_days=self.config.lead_time_days,
            exclusion_buffer_km=self.config.exclusion_buffer_km,
            positive_negative_ratio=self.config.positive_negative_ratio,
            data_origin="REAL",
        )

        positives = label_positive_samples(grid_cells, events, pipeline_config)
        logger.info(f"Labeled {len(positives)} positive samples")

        # Stage 4: Fetch all feature data in parallel
        logger.info("Fetching feature data...")

        # Rainfall
        rainfall_records = []
        if self.config.use_imd_rainfall:
            rainfall_records = await self.imd_client.fetch_rainfall_for_bbox(
                self.config.min_lat,
                self.config.min_lon,
                self.config.max_lat,
                self.config.max_lon,
                self.config.start_date,
                self.config.end_date,
            )
            logger.info(f"Fetched {len(rainfall_records)} rainfall records")

        # Terrain
        static_features = {}
        if self.config.use_srtm_dem:
            static_features = self.srtm_client.compute_grid_features(grid_cells)
            logger.info(f"Computed terrain features for {len(static_features)} cells")

        # Roads
        road_features = {}
        if self.config.use_osm_roads:
            road_features = self.osm_client.compute_road_features_for_grid(
                grid_cells,
                self.config.min_lat,
                self.config.min_lon,
                self.config.max_lat,
                self.config.max_lon,
                self.config.include_tracks_in_roads,
            )
            # Merge into static features
            for cell_id, feats in road_features.items():
                if cell_id in static_features:
                    static_features[cell_id].update(feats)
                else:
                    static_features[cell_id] = feats
            logger.info(f"Computed road features for {len(road_features)} cells")

        # Soil moisture
        if self.config.use_soil_moisture:
            soil_records = await self.smap_client.fetch_soil_moisture_for_bbox(
                self.config.min_lat,
                self.config.min_lon,
                self.config.max_lat,
                self.config.max_lon,
                self.config.start_date,
                self.config.end_date,
            )
            # TODO: Integrate soil moisture into features
            logger.info(f"Fetched {len(soil_records)} soil moisture records")

        # Land cover
        if self.config.use_land_cover:
            lulc_features = self.lulc_client.compute_grid_features(grid_cells)
            for cell_id, feats in lulc_features.items():
                if cell_id in static_features:
                    static_features[cell_id].update(feats)
                else:
                    static_features[cell_id] = feats
            logger.info(f"Computed land cover features for {len(lulc_features)} cells")

        # Stage 5: Sample negative samples
        logger.info("Sampling negative samples...")
        negatives = sample_negative_samples(
            grid_cells, events, positives, static_features, pipeline_config
        )
        logger.info(f"Sampled {len(negatives)} negative samples")

        # Stage 6: Assemble dataset (basic features)
        logger.info("Assembling dataset...")
        dataset = assemble_dataset(
            positives, negatives, rainfall_records, static_features, pipeline_config
        )
        logger.info(f"Assembled {len(dataset)} samples")

        # Stage 7: Compute advanced features (proximity, etc.)
        logger.info("Computing advanced features...")
        transformer = FeatureTransformer(
            rainfall_records=rainfall_records,
            static_features=static_features,
            landslide_events=events,
        )
        dataset = transformer.compute_all_for_dataset(dataset)
        logger.info("Advanced features computed")

        # Stage 8: Train/test split
        logger.info("Splitting train/test...")
        train, test = split_train_test(dataset, pipeline_config.train_test_split_date)
        logger.info(f"Train: {len(train)}, Test: {len(test)}")

        # Stage 9: Walk-forward CV splits
        cv_splits = walk_forward_split(
            dataset,
            n_splits=pipeline_config.walk_forward_n_splits,
            gap_days=pipeline_config.walk_forward_gap_days,
        )
        logger.info(f"Generated {len(cv_splits)} CV folds")

        # Stage 10: Class weights
        class_weights = get_class_weights(dataset)
        logger.info(f"Class weights: {class_weights}")

        # Stage 11: Save outputs
        logger.info("Saving outputs...")
        dataset_path = save_dataset(
            dataset, pipeline_config, self.output_dir / "training_dataset.csv"
        )
        train_path = save_dataset(train, pipeline_config, self.output_dir / "train.csv")
        test_path = save_dataset(test, pipeline_config, self.output_dir / "test.csv")

        metadata_path = save_metadata(
            dataset,
            pipeline_config,
            data_sources={
                "rainfall": "IMD/NASA_POWER" if rainfall_records else "none",
                "events": "GSI/COOLR",
                "terrain": "SRTM",
                "roads": "OSM",
                "soil_moisture": "NASA_POWER",
                "land_cover": "ESA_WorldCover",
            },
            output_path=self.output_dir / "dataset_metadata.json",
        )

        # Save CV splits
        cv_path = self.output_dir / "cv_splits.json"
        cv_data = []
        for i, (cv_train, cv_test) in enumerate(cv_splits):
            cv_data.append(
                {
                    "fold": i,
                    "train_size": len(cv_train),
                    "test_size": len(cv_test),
                    "train_date_range": self._date_range(cv_train),
                    "test_date_range": self._date_range(cv_test),
                }
            )
        with open(cv_path, "w") as f:
            json.dump(cv_data, f, indent=2, default=str)

        # Save class weights
        weights_path = self.output_dir / "class_weights.json"
        with open(weights_path, "w") as f:
            json.dump(class_weights, f, indent=2)

        logger.info("Pipeline completed successfully")

        return {
            "status": "success",
            "dataset_path": str(dataset_path),
            "train_path": str(train_path),
            "test_path": str(test_path),
            "metadata_path": str(metadata_path),
            "cv_splits_path": str(cv_path),
            "class_weights_path": str(weights_path),
            "n_samples": len(dataset),
            "n_train": len(train),
            "n_test": len(test),
            "n_positives": len(positives),
            "n_negatives": len(negatives),
            "n_cv_folds": len(cv_splits),
        }

    def _date_range(self, samples: list[dict]) -> dict:
        """Get min/max dates from samples."""
        dates = [date.fromisoformat(s["sample_date"]) for s in samples]
        return {"min": min(dates).isoformat(), "max": max(dates).isoformat()}


async def run_ingestion_pipeline(config: IngestionConfig | None = None) -> dict:
    """Convenience function to run ingestion with default config."""
    if config is None:
        config = IngestionConfig()
    pipeline = IngestionPipeline(config)
    return await pipeline.run()


def run_ingestion_pipeline_sync(config: IngestionConfig | None = None) -> dict:
    """Synchronous entry point for Celery tasks and management commands.

    Wraps the async pipeline in asyncio.run() so it can be called from
    synchronous Django/Celery contexts.
    """
    return asyncio.run(run_ingestion_pipeline(config))
