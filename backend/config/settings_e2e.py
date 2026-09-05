from .settings import *  # noqa: F401, F403

# Local demo / E2E override for machines WITHOUT PostgreSQL or Docker.
# Run the backend with:
#   python manage.py runserver --settings=config.settings_e2e
# Uses SQLite so the whole Django API (models, DRF, auth, GeoJSON JSON
# fallbacks) works without a database server.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "e2e_demo.sqlite3",
    }
}