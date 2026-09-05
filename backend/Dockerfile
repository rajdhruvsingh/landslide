# Backend image for the Landslide EWS.
#
# Dev workflow (see infra/docker-compose.yml) overrides CMD to run Django's
# runserver for live reload; the default CMD here serves via gunicorn for
# production builds.

FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# --- GeoDjango system dependencies (GDAL, GEOS, PROJ) ---
# libgdal-dev + gdal-bin provide the native libs; the GDAL Python wheel is
# built against the same version so `from osgeo import gdal` works and
# settings.py enables the PostGIS backend (GIS_AVAILABLE=True).
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    binutils \
    libproj-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir GDAL==$(gdal-config --version)

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]