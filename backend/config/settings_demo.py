from .settings import *  # noqa: F401, F403

# Dev/demo override: fire the REAL Celery tasks on short intervals so the
# beat -> broker -> worker -> execution chain can be observed quickly.
# Run beat with:  $env:DJANGO_SETTINGS_MODULE = "config.settings_demo"
# (worker/backend keep the default config.settings — only the scheduler changes.)
CELERY_BEAT_SCHEDULE = {
    "ingest-rainfall-demo-20s": {
        "task": "apps.ml_bridge.tasks.ingest_rainfall",
        "schedule": 20.0,
    },
    "ingest-soil-moisture-demo-30s": {
        "task": "apps.ml_bridge.tasks.ingest_soil_moisture",
        "schedule": 30.0,
    },
    "recompute-risk-demo-60s": {
        "task": "apps.ml_bridge.tasks.recompute_risk",
        "schedule": 60.0,
    },
}