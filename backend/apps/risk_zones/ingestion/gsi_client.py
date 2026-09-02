"""GSI / Bhukosh landslide inventory ingestion client.

This module provides a unified interface for fetching historical landslide data.
Primary source: GSI Bhukosh portal (requires institutional MoU - web map only, no bulk API).
Secondary source: Published research papers / curated datasets (freely available).
Tertiary source: Global landslide catalogs (NASA COOLR, GLC).

Data Format (standardized):
    List of dicts with keys: event_id, event_date, latitude, longitude, severity,
                              source_reference, data_origin

Sources:
    - "GSI_BHUKOSH": GSI Bhukosh portal (official)
    - "PUBLISHED_PAPER": Curated from peer-reviewed papers
    - "NASA_COOLR": NASA Cooperative Open Online Landslide Repository
    - "GLC": Global Landslide Catalog
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import date
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# NASA COOLR API (public)
NASA_COOLR_BASE = "https://cooler.gsfc.nasa.gov/api/landslides"

# GLC - no public API, would need download


class GSIClient:
    """GSI Landslide Inventory Client.

    Note: GSI Bhukosh does not provide a public bulk API. This client supports:
    1. Loading from local CSV (for curated/paper data)
    2. Fetching from NASA COOLR (public global catalog)
    3. Placeholder for GSI Bhukosh when MoU/credentials available
    """

    def __init__(
        self,
        use_coolr: bool = True,
        timeout_seconds: int = 30,
    ):
        """
        Args:
            use_coolr: If True, fetch from NASA COOLR as fallback.
            timeout_seconds: HTTP request timeout.
        """
        self.use_coolr = use_coolr
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def fetch_inventory(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> list[dict]:
        """Fetch landslide events for a bounding box and date range.

        Args:
            min_lat, min_lon, max_lat, max_lon: Bounding box.
            start_date: Start date (inclusive), None for no limit.
            end_date: End date (inclusive), None for no limit.

        Returns:
            List of dicts: event_id, event_date, latitude, longitude, severity,
                          source_reference, data_origin
        """
        records = []

        # Try NASA COOLR (global, public)
        if self.use_coolr:
            coolr_records = await self._fetch_coolr(
                min_lat, min_lon, max_lat, max_lon, start_date, end_date
            )
            records.extend(coolr_records)

        # Try GSI Bhukosh (placeholder)
        # gsi_records = await self._fetch_gsi_bhukosh(...)
        # records.extend(gsi_records)

        # Deduplicate by location+date (approximate)
        return self._deduplicate(records)

    async def _fetch_coolr(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        start_date: date | None,
        end_date: date | None,
    ) -> list[dict]:
        """Fetch from NASA COOLR API."""
        params = {
            "min_lat": min_lat,
            "max_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
        }
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(NASA_COOLR_BASE, params=params) as resp:
                    if resp.status != 200:
                        logger.warning(f"COOLR API error: {resp.status}")
                        return []
                    data = await resp.json()

            records = []
            for item in data.get("features", []):
                props = item.get("properties", {})
                geom = item.get("geometry", {})
                coords = geom.get("coordinates", [])

                try:
                    event_date = date.fromisoformat(props.get("event_date", "")[:10])
                    if start_date and event_date < start_date:
                        continue
                    if end_date and event_date > end_date:
                        continue

                    records.append(
                        {
                            "event_id": f"COOLR_{props.get('objectid', '')}",
                            "event_date": event_date,
                            "latitude": float(coords[1]) if len(coords) > 1 else 0,
                            "longitude": float(coords[0]) if len(coords) > 0 else 0,
                            "severity": props.get("severity", "Unknown"),
                            "source_reference": "NASA COOLR",
                            "data_origin": "REAL",
                        }
                    )
                except (ValueError, TypeError, IndexError):
                    continue

            logger.info(f"Fetched {len(records)} events from NASA COOLR")
            return records

        except Exception as e:
            logger.error(f"COOLR fetch failed: {e}")
            return []

    def _deduplicate(self, records: list[dict], tol_km: float = 1.0) -> list[dict]:
        """Deduplicate events by location and date (approximate)."""
        if not records:
            return []

        # Simple deduplication: same date + nearby location
        unique = []
        seen = set()

        for r in records:
            key = (
                r["event_date"].isoformat()
                if isinstance(r["event_date"], date)
                else str(r["event_date"]),
                round(r["latitude"], 3),
                round(r["longitude"], 3),
            )
            if key not in seen:
                seen.add(key)
                unique.append(r)

        logger.info(f"Deduplicated {len(records)} -> {len(unique)} events")
        return unique

    def load_from_csv(
        self,
        csv_path: str | Path,
        date_column: str = "event_date",
        lat_column: str = "latitude",
        lon_column: str = "longitude",
    ) -> list[dict]:
        """Load landslide inventory from local CSV (curated/paper data).

        Expected columns: event_id, event_date, latitude, longitude,
                          severity, source_reference
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
                            "event_id": row.get("event_id", ""),
                            "event_date": date.fromisoformat(row[date_column]),
                            "latitude": float(row[lat_column]),
                            "longitude": float(row[lon_column]),
                            "severity": row.get("severity", "Unknown"),
                            "source_reference": row.get("source_reference", ""),
                            "data_origin": "REAL",
                        }
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
        return records

    def load_from_geojson(self, geojson_path: str | Path) -> list[dict]:
        """Load landslide inventory from GeoJSON."""
        path = Path(geojson_path)
        if not path.exists():
            logger.error(f"GeoJSON not found: {path}")
            return []

        with open(path) as f:
            data = json.load(f)

        records = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [])

            try:
                records.append(
                    {
                        "event_id": props.get("event_id", props.get("id", "")),
                        "event_date": date.fromisoformat(
                            props.get("event_date", "")[:10]
                        ),
                        "latitude": float(coords[1]) if len(coords) > 1 else 0,
                        "longitude": float(coords[0]) if len(coords) > 0 else 0,
                        "severity": props.get("severity", "Unknown"),
                        "source_reference": props.get("source", "GeoJSON"),
                        "data_origin": "REAL",
                    }
                )
            except (ValueError, TypeError, IndexError):
                continue
        return records


async def fetch_inventory(zone_id: int) -> list[dict]:
    """Backward-compatible stub function."""
    GSIClient()
    # Return empty for now - would need zone bounds
    return []
