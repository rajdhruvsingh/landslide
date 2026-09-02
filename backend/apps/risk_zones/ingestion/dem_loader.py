"""SRTM/DEM data ingestion and processing client.

Primary source: SRTM (Shuttle Radar Topography Mission) - freely available globally.
  - SRTM GL1: 30m resolution (1 arc-second), global coverage
  - SRTM GL3: 90m resolution (3 arc-second), global coverage
Fallback source: NASADEM (improved SRTM), ASTER GDEM v3.
Secondary source: ISRO Bhuvan (requires registration - higher resolution for India).

This module handles:
1. Downloading SRTM tiles for a bounding box
2. Computing slope, aspect, elevation at grid cell centroids
3. Storing static features per grid cell for ML pipeline

Dependencies:
    - rasterio: for reading GeoTIFF
    - numpy/scipy: for slope/aspect computation
    - requests: for downloading tiles

Install: pip install rasterio numpy scipy requests
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None

logger = logging.getLogger(__name__)

# SRTM tile server (USGS/NASA)
SRTM_BASE_URL = "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11"
# Alternative: AWS S3 SRTM mirror
SRTM_AWS_BASE = "https://srtmtiles.s3.amazonaws.com"


class SRTMClient:
    """SRTM DEM Client for downloading and processing elevation data.

    Uses SRTM GL1 (30m) tiles from USGS/NASA or AWS mirror.
    Computes slope, aspect, elevation at grid cell centroids.
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/srtm_cache",
        resolution: str = "30m",  # "30m" (GL1) or "90m" (GL3)
        max_workers: int = 4,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded tiles.
            resolution: "30m" for SRTM GL1, "90m" for SRTM GL3.
            max_workers: Parallel downloads.
        """
        if not RASTERIO_AVAILABLE:
            logger.warning(
                "rasterio/scipy not available. DEM features will use fallback."
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = resolution
        self.max_workers = max_workers

        # Tile naming: N27E088.hgt for 27°N, 88°E
        self.tile_size_deg = 1.0  # SRTM tiles are 1°x1°

    def _tile_name(self, lat: float, lon: float) -> str:
        """Generate SRTM tile name from lat/lon."""
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"{lat_dir}{abs(int(lat)):02d}{lon_dir}{abs(int(lon)):03d}.hgt"

    def _tile_url(self, tile_name: str) -> str:
        """Generate download URL for SRTM tile."""
        # AWS mirror is more reliable
        return f"{SRTM_AWS_BASE}/{tile_name}"

    def _tile_path(self, tile_name: str) -> Path:
        """Local cache path for tile."""
        return self.cache_dir / tile_name

    def download_tile(self, lat: float, lon: float) -> Path | None:
        """Download a single SRTM tile if not cached."""
        tile_name = self._tile_name(lat, lon)
        tile_path = self._tile_path(tile_name)

        if tile_path.exists():
            return tile_path

        url = self._tile_url(tile_name)
        try:
            logger.info(f"Downloading SRTM tile: {tile_name}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(tile_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # SRTM .hgt files are raw 16-bit signed integers
            # Some sources provide .zip, handle if needed
            if tile_path.suffix == ".zip":
                import zipfile

                with zipfile.ZipFile(tile_path, "r") as z:
                    z.extractall(self.cache_dir)
                tile_path.unlink()  # Remove zip
                # Find extracted .hgt
                extracted = list(
                    self.cache_dir.glob(f"{tile_name.replace('.hgt', '')}*.hgt")
                )
                if extracted:
                    return extracted[0]

            return tile_path

        except Exception as e:
            logger.error(f"Failed to download {tile_name}: {e}")
            if tile_path.exists():
                tile_path.unlink()
            return None

    def download_tiles_for_bbox(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
    ) -> list[Path]:
        """Download all SRTM tiles covering a bounding box."""
        tiles_needed = []
        lat = math.floor(min_lat)
        while lat <= max_lat:
            lon = math.floor(min_lon)
            while lon <= max_lon:
                tiles_needed.append((lat, lon))
                lon += 1
            lat += 1

        downloaded = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.download_tile, lat, lon): (lat, lon)
                for lat, lon in tiles_needed
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    downloaded.append(result)

        return downloaded

    def read_elevation_at_points(
        self,
        points: list[tuple[float, float]],  # [(lat, lon), ...]
        bbox: tuple[float, float, float, float] | None = None,
    ) -> dict[tuple[float, float], dict]:
        """Read elevation, slope, aspect at specific points.

        Args:
            points: List of (lat, lon) tuples.
            bbox: Optional (min_lat, min_lon, max_lat, max_lon) to limit tile downloads.

        Returns:
            Dict mapping (lat, lon) -> {elevation_m, slope_angle_deg, slope_aspect_deg}
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available, returning zeros")
            return {
                p: {"elevation_m": 0.0, "slope_angle_deg": 0.0, "slope_aspect_deg": 0.0}
                for p in points
            }

        # Determine required tiles
        if bbox:
            min_lat, min_lon, max_lat, max_lon = bbox
        else:
            lats = [p[0] for p in points]
            lons = [p[1] for p in points]
            min_lat, max_lat = math.floor(min(lats)) - 1, math.ceil(max(lats)) + 1
            min_lon, max_lon = math.floor(min(lons)) - 1, math.ceil(max(lons)) + 1

        # Download tiles
        self.download_tiles_for_bbox(min_lat, min_lon, max_lat, max_lon)

        # Mosaic tiles into a single array (simplified - in production use rasterio.merge)
        # For now, process each point by finding its tile
        results = {}
        for lat, lon in points:
            tile_path = self._tile_path(self._tile_name(lat, lon))
            if not tile_path.exists():
                results[(lat, lon)] = {
                    "elevation_m": 0.0,
                    "slope_angle_deg": 0.0,
                    "slope_aspect_deg": 0.0,
                }
                continue

            try:
                elev, slope, aspect = self._extract_from_hgt(tile_path, lat, lon)
                results[(lat, lon)] = {
                    "elevation_m": elev,
                    "slope_angle_deg": slope,
                    "slope_aspect_deg": aspect,
                }
            except Exception as e:
                logger.error(f"Failed to extract from {tile_path}: {e}")
                results[(lat, lon)] = {
                    "elevation_m": 0.0,
                    "slope_angle_deg": 0.0,
                    "slope_aspect_deg": 0.0,
                }

        return results

    def _extract_from_hgt(
        self,
        hgt_path: Path,
        lat: float,
        lon: float,
    ) -> tuple[float, float, float]:
        """Extract elevation, slope, aspect from SRTM .hgt file at given lat/lon.

        SRTM .hgt format: 16-bit signed integers, big-endian, row-major (N to S).
        GL1: 3601 x 3601 pixels (1° at 30m ≈ 3600 pixels + 1 overlap)
        """
        # Read .hgt as raw binary
        with open(hgt_path, "rb") as f:
            data = np.fromfile(f, dtype=">i2")  # big-endian int16

        # Determine dimensions
        n_pixels = int(math.sqrt(len(data)))
        if n_pixels * n_pixels != len(data):
            # Try common sizes
            for size in [3601, 1201, 6001]:
                if size * size == len(data):
                    n_pixels = size
                    break

        elevation_grid = data.reshape((n_pixels, n_pixels))

        # SRTM: rows go North to South, cols West to East
        # Tile covers [tile_lat, tile_lat+1] x [tile_lon, tile_lon+1]
        tile_lat = int(math.floor(lat))
        tile_lon = int(math.floor(lon))

        # Pixel coordinates (0,0) = NW corner
        # lat increases North, so row = (tile_lat + 1 - lat) * (n_pixels - 1)
        row = int(round((tile_lat + 1 - lat) * (n_pixels - 1)))
        col = int(round((lon - tile_lon) * (n_pixels - 1)))

        # Clamp to valid range
        row = max(0, min(n_pixels - 1, row))
        col = max(0, min(n_pixels - 1, col))

        elevation = float(elevation_grid[row, col])
        if elevation <= -32768:  # SRTM void value
            elevation = 0.0

        # Compute slope/aspect using 3x3 window (Horn's method)
        # Need neighbors - handle edges
        if row == 0 or row == n_pixels - 1 or col == 0 or col == n_pixels - 1:
            return elevation, 0.0, 0.0

        # 3x3 window
        window = elevation_grid[row - 1 : row + 2, col - 1 : col + 2]

        # Horn's method: dz/dx = (z3 + 2*z6 + z9 - z1 - 2*z4 - z7) / 8
        #                  dz/dy = (z1 + 2*z2 + z3 - z7 - 2*z8 - z9) / 8
        # where:
        # z1 z2 z3
        # z4 z5 z6
        # z7 z8 z9
        z1, z2, z3 = window[0, 0], window[0, 1], window[0, 2]
        z4, _z5, z6 = window[1, 0], window[1, 1], window[1, 2]
        z7, z8, z9 = window[2, 0], window[2, 1], window[2, 2]

        # Cell size in meters (approximate at this latitude)
        cell_size = 30.0  # SRTM GL1 is 30m

        dzdx = (z3 + 2 * z6 + z9 - z1 - 2 * z4 - z7) / (8 * cell_size)
        dzdy = (z1 + 2 * z2 + z3 - z7 - 2 * z8 - z9) / (8 * cell_size)

        slope_rad = math.atan(math.sqrt(dzdx**2 + dzdy**2))
        slope_deg = math.degrees(slope_rad)

        # Aspect: 0=N, 90=E, 180=S, 270=W
        if slope_deg < 0.1:
            aspect = 0.0
        else:
            aspect_rad = math.atan2(-dzdx, dzdy)  # Note: atan2(-dz/dx, dz/dy)
            aspect = math.degrees(aspect_rad)
            if aspect < 0:
                aspect += 360

        return elevation, round(slope_deg, 2), round(aspect, 2)

    def compute_grid_features(
        self,
        grid_cells: list[dict],
    ) -> dict[str, dict]:
        """Compute terrain features for all grid cells.

        Args:
            grid_cells: List of dicts with grid_cell_id, centroid_lat, centroid_lon.

        Returns:
            Dict mapping grid_cell_id -> {slope_angle_deg, slope_aspect_deg, elevation_m}
        """
        points = [(c["centroid_lat"], c["centroid_lon"]) for c in grid_cells]
        results = self.read_elevation_at_points(points)

        features = {}
        for cell in grid_cells:
            key = (cell["centroid_lat"], cell["centroid_lon"])
            features[cell["grid_cell_id"]] = results.get(
                key,
                {
                    "elevation_m": 0.0,
                    "slope_angle_deg": 0.0,
                    "slope_aspect_deg": 0.0,
                },
            )
        return features


# Backward compatibility
def load_dem_to_postgis(tiff_path: Path) -> dict:
    """Legacy function - kept for compatibility."""
    if not RASTERIO_AVAILABLE:
        return {"status": "error", "message": "rasterio not installed"}

    try:
        with rasterio.open(tiff_path) as src:
            # Would use raster2pgsql in production
            return {
                "status": "success",
                "bounds": src.bounds,
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}
