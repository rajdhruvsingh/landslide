"""Satellite soil moisture data ingestion client.

This is a stub. SMAP/ESA CCI data access requires earthdata.nasa.gov
account or ESA CCI registration.

TODO: Replace with real SMAP API or bulk download when access is configured.
Format reference: SMAP L3 daily soil moisture (9km or 36km resolution).
"""


async def fetch_soil_moisture(zone_id: int) -> dict | None:
    """Fetch soil moisture for a zone centroid.

    Returns:
        dict with keys: zone_id, soil_moisture_pct, timestamp, source.
    """
    return None
