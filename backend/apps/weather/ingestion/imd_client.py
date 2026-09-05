"""IMD rainfall data ingestion client.

This module provides a unified interface for fetching rainfall data.
Primary source: IMD (requires institutional MoU - not publicly accessible).
Fallback source: NASA POWER API (freely available, satellite/reanalysis-based).
Both sources return data in the same standardized format.

Data Format (standardized):
    List of dicts with keys: station_id, station_lat, station_lon, reading_date, rainfall_mm, source

Sources:
    - "IMD": India Meteorological Department (official ground stations)
    - "NASA_POWER": NASA POWER API (reanalysis/satellite, 0.5° grid)
"""

from __future__ import annotations

import asyncio
import csv
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# NASA POWER API base URL
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters for rainfall
NASA_POWER_PARAMS = "PRECTOTCORR"  # Precipitation (mm/day)


class IMDClient:
    """IMD Rainfall Data Client.

    Note: IMD does not provide a public REST API. This client is a placeholder
    for when institutional credentials are available. Currently delegates to
    NASA POWER API as a freely available fallback.
    """

    def __init__(
        self,
        use_fallback: bool = True,
        cache_dir: str | Path | None = None,
        timeout_seconds: int = 30,
    ):
        """
        Args:
            use_fallback: If True, use NASA POWER when IMD unavailable.
            cache_dir: Directory to cache downloaded CSV files.
            timeout_seconds: HTTP request timeout.
        """
        self.use_fallback = use_fallback
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def fetch_rainfall(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
        station_lat: float | None = None,
        station_lon: float | None = None,
    ) -> list[dict]:
        """Fetch rainfall data for a station/location.

        Args:
            station_id: Station identifier (IMD code or generated for POWER grid).
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            station_lat: Latitude (required for POWER fallback).
            station_lon: Longitude (required for POWER fallback).

        Returns:
            List of dicts: station_id, station_lat, station_lon, reading_date, rainfall_mm, source
        """
        # Try IMD first (placeholder - would need real credentials)
        if not self.use_fallback:
            return await self._fetch_imd(station_id, start_date, end_date)

        # Use NASA POWER as fallback
        if station_lat is None or station_lon is None:
            logger.warning(
                f"No coordinates for {station_id}, cannot use POWER fallback"
            )
            return []

        return await self._fetch_nasa_power(
            station_id, station_lat, station_lon, start_date, end_date
        )

    async def fetch_rainfall_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        start_date: date,
        end_date: date,
        grid_step: float = 0.5,
    ) -> list[dict]:
        """Fetch rainfall for a bounding box using NASA POWER grid.

        Args:
            min_lat, min_lon, max_lat, max_lon: Bounding box.
            start_date, end_date: Date range.
            grid_step: Grid resolution in degrees (POWER is 0.5°).

        Returns:
            List of standardized rainfall records.
        """
        all_records = []
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                station_id = f"POWER_{lat:.2f}_{lon:.2f}"
                records = await self._fetch_nasa_power(
                    station_id, lat, lon, start_date, end_date
                )
                all_records.extend(records)
                lon += grid_step
            lat += grid_step
            await asyncio.sleep(0.1)  # Rate limiting
        return all_records

    async def _fetch_imd(
        self,
        station_id: str,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch from IMD (placeholder - requires institutional access).

        When IMD credentials are available, implement:
        - FTP/SFTP download from IMD server
        - Or API call to institutional endpoint
        """
        logger.warning("IMD client not configured with credentials. Returning empty.")
        return []

    async def _fetch_nasa_power(
        self,
        station_id: str,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch rainfall from NASA POWER API.

        NASA POWER provides daily meteorological data from satellite/reanalysis.
        Resolution: 0.5° x 0.5° (~50km at equator).
        Parameter: PRECTOTCORR = corrected precipitation (mm/day).
        """
        params = {
            "parameters": NASA_POWER_PARAMS,
            "community": "AG",
            "longitude": lon,
            "latitude": lat,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(NASA_POWER_BASE, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"NASA POWER API error: {resp.status}")
                        return []
                    data = await resp.json()

            # Parse NASA POWER response
            records = []
            param_data = (
                data.get("properties", {})
                .get("parameter", {})
                .get(NASA_POWER_PARAMS, {})
            )

            for date_str, value in param_data.items():
                try:
                    reading_date = datetime.strptime(date_str, "%Y%m%d").date()
                    # POWER returns -999 for missing data
                    rainfall_mm = float(value) if value > -900 else 0.0
                    records.append(
                        {
                            "station_id": station_id,
                            "station_lat": lat,
                            "station_lon": lon,
                            "reading_date": reading_date,
                            "rainfall_mm": round(rainfall_mm, 2),
                            "source": "NASA_POWER",
                        }
                    )
                except (ValueError, TypeError):
                    continue

            logger.info(
                f"Fetched {len(records)} records from NASA POWER for {station_id}"
            )
            return records

        except Exception as e:
            logger.error(f"NASA POWER fetch failed for {station_id}: {e}")
            return []

    async def fetch_from_csv(
        self,
        csv_path: str | Path,
        date_column: str = "reading_date",
        rain_column: str = "rainfall_mm",
    ) -> list[dict]:
        """Load rainfall from local CSV file (for IMD manual downloads).

        Expected columns: station_id, station_lat, station_lon, reading_date, rainfall_mm
        """
        path = Path(csv_path)
        if not path.exists():
            logger.error(f"CSV not found: {path}")
            return []

        records = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    records.append(
                        {
                            "station_id": row.get("station_id", ""),
                            "station_lat": float(row.get("station_lat", 0)),
                            "station_lon": float(row.get("station_lon", 0)),
                            "reading_date": date.fromisoformat(row[date_column]),
                            "rainfall_mm": float(row[rain_column]),
                            "source": "IMD_CSV",
                        }
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
        return records


async def fetch_rainfall(station_id: str) -> list[dict]:
    """Backward-compatible stub function."""
    client = IMDClient()
    # Default to last 30 days
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30)
    return await client.fetch_rainfall(station_id, start, end)
