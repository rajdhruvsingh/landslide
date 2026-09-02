import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-CHANGE-ME-IN-PRODUCTION",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True").lower() in ("true", "1", "yes")

ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(
    ","
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "apps.risk_zones",
    "apps.reports",
    "apps.alerts",
    "apps.users",
    "apps.weather",
    "apps.ml_bridge",
]

# GeoDjango: enable django.contrib.gis only when GDAL is available
# (present in Docker/production, absent on some dev machines)
try:
    import gdal  # noqa: F401

    INSTALLED_APPS.insert(5, "django.contrib.gis")
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

_POSTGRES_DB = os.environ.get("POSTGRES_DB", "landslide_ews")
_POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
_POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")
_POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
_POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")

if GIS_AVAILABLE:
    DATABASES = {
        "default": {
            "ENGINE": "django.contrib.gis.db.backends.postgis",
            "NAME": _POSTGRES_DB,
            "USER": _POSTGRES_USER,
            "PASSWORD": _POSTGRES_PASSWORD,
            "HOST": _POSTGRES_HOST,
            "PORT": _POSTGRES_PORT,
        }
    }
else:
    # Fallback for dev machines without GDAL — plain PostgreSQL (no spatial)
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": _POSTGRES_DB,
            "USER": _POSTGRES_USER,
            "PASSWORD": _POSTGRES_PASSWORD,
            "HOST": _POSTGRES_HOST,
            "PORT": _POSTGRES_PORT,
        }
    }

AUTH_USER_MODEL = "users.User"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 50,
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ],
}

# Celery
CELERY_BROKER_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CELERY_BEAT_SCHEDULE = {
    "ingest-rainfall-3h": {
        "task": "apps.ml_bridge.tasks.ingest_rainfall",
        "schedule": 10800.0,
    },
    "ingest-soil-moisture-daily": {
        "task": "apps.ml_bridge.tasks.ingest_soil_moisture",
        "schedule": 86400.0,
    },
    "recompute-risk-daily": {
        "task": "apps.ml_bridge.tasks.recompute_risk",
        "schedule": 86400.0,
    },
}

# SMS Gateway
SMS_GATEWAY_MOCK = os.environ.get("SMS_GATEWAY_MOCK", "True").lower() in (
    "true",
    "1",
    "yes",
)
SMS_GATEWAY_API_KEY = os.environ.get("SMS_GATEWAY_API_KEY", "")
SMS_GATEWAY_SENDER_ID = os.environ.get("SMS_GATEWAY_SENDER_ID", "")

# FCM
FCM_CREDENTIALS_PATH = os.environ.get("FCM_CREDENTIALS_PATH", "")

# S3 / MinIO
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "minioadmin")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "landslide-reports")
