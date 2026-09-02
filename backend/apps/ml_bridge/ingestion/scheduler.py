"""Celery beat scheduler configuration for periodic data ingestion.

Actual task implementations are in apps.ml_bridge.tasks.
"""

INGESTION_SCHEDULE = {
    "imd-rainfall-every-3h": {
        "task": "apps.ml_bridge.tasks.ingest_rainfall",
        "schedule": 10800.0,  # 3 hours in seconds
    },
    "soil-moisture-daily": {
        "task": "apps.ml_bridge.tasks.ingest_soil_moisture",
        "schedule": 86400.0,  # 24 hours
    },
    "risk-recompute-daily": {
        "task": "apps.ml_bridge.tasks.recompute_risk",
        "schedule": 86400.0,  # 24 hours
    },
}