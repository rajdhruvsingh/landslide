"""Celery tasks for data ingestion and risk recomputation.

These tasks wrap the framework-agnostic ingestion clients and ML pipeline,
making them callable from Celery Beat schedules and Django management commands.
"""

import logging
import asyncio
from datetime import date, timedelta

from config.celery import app as celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="apps.ml_bridge.tasks.ingest_rainfall")
def ingest_rainfall():
    """Fetch recent rainfall data from IMD/NASA POWER for all monitored zones."""
    from apps.weather.ingestion.imd_client import IMDClient

    client = IMDClient(use_fallback=True)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=7)

    loop = asyncio.new_event_loop()
    try:
        records = loop.run_until_complete(
            client.fetch_rainfall_for_bbox(
                min_lat=27.0, min_lon=88.0,
                max_lat=28.0, max_lon=89.0,
                start_date=start, end_date=end,
            )
        )
        logger.info("Rainfall ingestion completed: %d records", len(records))
        return {"status": "success", "records": len(records)}
    except Exception as e:
        logger.exception("Rainfall ingestion failed")
        return {"status": "error", "error": str(e)}
    finally:
        loop.close()


@celery_app.task(name="apps.ml_bridge.tasks.ingest_soil_moisture")
def ingest_soil_moisture():
    """Fetch recent soil moisture data from NASA POWER for all monitored zones."""
    from apps.weather.ingestion.smap_client import SoilMoistureClient

    client = SoilMoistureClient(use_power_fallback=True)
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=1)

    loop = asyncio.new_event_loop()
    try:
        records = loop.run_until_complete(
            client.fetch_soil_moisture_for_bbox(
                min_lat=27.0, min_lon=88.0,
                max_lat=28.0, max_lon=89.0,
                start_date=start, end_date=end,
            )
        )
        logger.info("Soil moisture ingestion completed: %d records", len(records))
        return {"status": "success", "records": len(records)}
    except Exception as e:
        logger.exception("Soil moisture ingestion failed")
        return {"status": "error", "error": str(e)}
    finally:
        loop.close()


@celery_app.task(name="apps.ml_bridge.tasks.recompute_risk")
def recompute_risk():
    """Recompute baseline risk levels for all monitored zones.

    Uses the always-active published threshold model over the latest stored
    weather readings and persists the result back onto each RiskZone
    (current_risk_level / last_computed_at). Fast, deterministic and offline.

    The full dataset-builder pipeline remains available as
    ``apps.ml_bridge.ingestion.pipeline.run_ingestion_pipeline_sync`` for
    Phase-2 ML feature generation; it is intentionally not wired into the
    operational recompute path.
    """
    from apps.ml_bridge.risk_evaluator import recompute_zone_risks

    try:
        result = recompute_zone_risks()
        updated = [z for z in result["zones"] if z.get("updated")]
        logger.info("Risk recomputation completed: %d zones updated", len(updated))
        return result
    except Exception as e:
        logger.exception("Risk recomputation failed")
        return {"status": "error", "error": str(e)}