"""SMAP/Soil Moisture data ingestion client.

Primary source: NASA SMAP (Soil Moisture Active Passive) - requires Earthdata login.
  - SPL3SMP: Enhanced L3 Radiometer Global Daily 9km (requires auth)
  - SPL4SMAU: L4 Carbon Net Ecosystem Exchange (requires auth)

Fallback source: NASA POWER API (freely available, no auth required).
  - Provides soil moisture from GLDAS/NOAH land surface model
  - Parameters: GWETROOT (root zone soil wetness), GWETTOP (surface soil wetness)
  - Resolution: 0.5° x 0.5° (~50km)
  - Temporal: Daily, from 1981 to near real-time

This module:
1. Fetches soil moisture from NASA POWER (no auth fallback)
2. Provides interface for SMAP when Earthdata credentials available
3. Standardizes output format for ML pipeline

Data Format (standardized):
    List of dicts with keys: zone_id, latitude, longitude, reading_date,
                              soil_moisture_pct, source

Sources:
    - "NASA_POWER": GLDAS/NOAH reanalysis (free)
    - "SMAP": NASA SMAP L3/L4 (requires Earthdata)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

import aiohttp

logger = logging.getLogger(__name__)

# NASA POWER API
NASA_POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Soil moisture parameters from POWER
# GWETROOT = Root zone soil wetness (0-1 fraction)
# GWETTOP = Surface soil wetness (0-1 fraction)
SOIL_MOISTURE_PARAMS = "GWETROOT,GWETTOP"


class SoilMoistureClient:
    """Soil Moisture Client with NASA POWER fallback and SMAP support."""

    def __init__(
        self,
        use_power_fallback: bool = True,
        earthdata_username: str | None = None,
        earthdata_password: str | None = None,
        timeout_seconds: int = 30,
    ):
        """
        Args:
            use_power_fallback: Use NASA POWER when SMAP unavailable.
            earthdata_username: NASA Earthdata username (for SMAP).
            earthdata_password: NASA Earthdata password (for SMAP).
            timeout_seconds: HTTP timeout.
        """
        self.use_power_fallback = use_power_fallback
        self.earthdata_username = earthdata_username
        self.earthdata_password = earthdata_password
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # SMAP endpoints (require auth)
        self.smap_base = "https://n5eil01u.ecs.nsidc.org/egi/request"
        self._auth = None
        if earthdata_username and earthdata_password:
            self._auth = aiohttp.BasicAuth(earthdata_username, earthdata_password)

    async def fetch_soil_moisture(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch soil moisture for a point location.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            List of dicts: latitude, longitude, reading_date, soil_moisture_pct, source
        """
        # Try SMAP first if credentials available
        if self._auth:
            smap_data = await self._fetch_smap(
                latitude, longitude, start_date, end_date
            )
            if smap_data:
                return smap_data

        # Fallback to NASA POWER
        if self.use_power_fallback:
            return await self._fetch_power(latitude, longitude, start_date, end_date)

        return []

    async def fetch_soil_moisture_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        start_date: date,
        end_date: date,
        grid_step: float = 0.5,
    ) -> list[dict]:
        """Fetch soil moisture for a bounding box grid."""
        all_records = []
        lat = min_lat
        while lat <= max_lat:
            lon = min_lon
            while lon <= max_lon:
                records = await self.fetch_soil_moisture(lat, lon, start_date, end_date)
                all_records.extend(records)
                lon += grid_step
            lat += grid_step
            await asyncio.sleep(0.1)  # Rate limiting
        return all_records

    async def _fetch_power(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Fetch soil moisture from NASA POWER API."""
        params = {
            "parameters": SOIL_MOISTURE_PARAMS,
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
                        logger.error(f"NASA POWER soil moisture error: {resp.status}")
                        return []
                    data = await resp.json()

            records = []
            param_data = data.get("properties", {}).get("parameter", {})

            # Merge GWETROOT and GWETTOP by date
            gwetroot = param_data.get("GWETROOT", {})
            gwetroot = param_data.get("GWETTOP", {})

            all_dates = set(gwetroot.keys()) | set(gwetroot.keys())

            for date_str in all_dates:
                try:
                    reading_date = datetime.strptime(date_str, "%Y%m%d").date()

                    # Use root zone if available, else surface
                    root_val = gwetroot.get(date_str, -999)
                    top_val = gwetroot.get(date_str, -999)

                    # Prefer root zone, fallback to surface
                    val = root_val if root_val > -900 else top_val
                    if val <= -900:
                        continue

                    # Convert fraction to percentage
                    soil_moisture_pct = round(val * 100, 2)

                    records.append(
                        {
                            "latitude": lat,
                            "longitude": lon,
                            "reading_date": reading_date,
                            "soil_moisture_pct": soil_moisture_pct,
                            "source": "NASA_POWER",
                        }
                    )
                except (ValueError, TypeError):
                    continue

            logger.info(f"Fetched {len(records)} soil moisture records from NASA POWER")
            return records

        except Exception as e:
            logger.error(f"NASA POWER soil moisture fetch failed: {e}")
            return []

    async def _fetch_smap(
        self,
        lat: float,
        lon: float,
        start_date: date,
        end_date: date,
    ) -> list[dict] | None:
        """Fetch from SMAP (requires Earthdata auth)."""
        if not self._auth:
            return None

        # SMAP SPL3SMP daily 9km tiles
        # Would need to find correct tile for lat/lon and query NSIDC
        # This is a placeholder for full implementation
        logger.warning(
            "SMAP fetch not fully implemented - requires NSIDC API integration"
        )
        return None


async def fetch_soil_moisture(zone_id: int) -> dict | None:
    """Backward-compatible stub."""
    SoilMoistureClient()
    return None
