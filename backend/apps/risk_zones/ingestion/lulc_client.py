"""ESA WorldCover Land Cover ingestion client.

Primary source: ESA WorldCover (10m resolution, global, free).
  - 2020 v100: https://esa-worldcover.s3.eu-central-1.amazonaws.com
  - 2021 v200: Updated version
  - Classes: 11 land cover types (Tree cover, Shrubland, Grassland, Cropland,
             Built-up, Bare/sparse vegetation, Snow/ice, Water bodies,
             Wetlands, Mangroves, Moss/lichen)

Alternative: ESA CCI Land Cover (300m, annual 1992-2020).
Alternative: Bhuvan LULC (requires registration, India-specific).

This module:
1. Downloads ESA WorldCover tiles for bounding box
2. Extracts land cover class at grid cell centroids
3. Computes majority class within buffer for each cell

Dependencies:
    - rasterio: for reading Cloud Optimized GeoTIFFs
    - numpy: for array operations

Install: pip install rasterio numpy requests

ESA WorldCover Classes (v100/v200):
    10: Tree cover
    20: Shrubland
    30: Grassland
    40: Cropland
    50: Built-up
    60: Bare / sparse vegetation
    70: Snow and ice
    80: Open water
    90: Herbaceous wetland
    95: Mangroves
    100: Moss and lichen
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

# ESA WorldCover S3 base (v200 = 2021)
WORLDCOVER_BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_V200 = "v200/2021/map"  # 2021 version
WORLDCOVER_V100 = "v100/2020/map"  # 2020 version

# Tile naming: ESA_WorldCover_10m_2021_v200_N27E088_Map.tif
# Tiles are 3°x3°

# Land cover class mapping
LULC_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Open water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Grouped classes for simplified modeling
LULC_GROUPS = {
    "Forest": [10],
    "Shrubland": [20],
    "Grassland": [30],
    "Cropland": [40],
    "Built-up": [50],
    "Barren": [60, 70],
    "Water": [80],
    "Wetland": [90, 95],
    "Other": [100],
}

# Reverse mapping
CLASS_TO_GROUP = {}
for group, classes in LULC_GROUPS.items():
    for c in classes:
        CLASS_TO_GROUP[c] = group


class WorldCoverClient:
    """ESA WorldCover Land Cover Client."""

    def __init__(
        self,
        cache_dir: str | Path = "data/worldcover_cache",
        version: str = "v200",  # "v100" (2020) or "v200" (2021)
        max_workers: int = 4,
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded tiles.
            version: "v100" for 2020, "v200" for 2021.
            max_workers: Parallel downloads.
        """
        if not RASTERIO_AVAILABLE:
            logger.warning(
                "rasterio not available. Land cover features will use fallback."
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.max_workers = max_workers

        if version == "v200":
            self.base_url = f"{WORLDCOVER_BASE}/{WORLDCOVER_V200}"
            self.year = 2021
        else:
            self.base_url = f"{WORLDCOVER_BASE}/{WORLDCOVER_V100}"
            self.year = 2020

        # Tile grid: 3°x3°
        self.tile_size_deg = 3.0

    def _tile_coords(self, lat: float, lon: float) -> tuple[int, int]:
        """Get tile grid coordinates (3°x3° tiles)."""
        # Tiles start at 180°W, 84°N
        # Tile (0,0) = 180°W to 177°W, 84°N to 81°N
        # But easier: just floor to 3° grid
        tile_x = math.floor((lon + 180) / 3)
        tile_y = math.floor((84 - lat) / 3)
        return tile_x, tile_y

    def _tile_name(self, lat: float, lon: float) -> str:
        """Generate WorldCover tile filename."""
        tile_x, tile_y = self._tile_coords(lat, lon)

        # Convert tile coords to lat/lon bounds for naming
        # tile_x=0 -> 180°W, tile_y=0 -> 84°N
        tile_lon_min = -180 + tile_x * 3
        tile_lat_max = 84 - tile_y * 3

        # Naming uses N/S/E/W at tile center
        center_lat = tile_lat_max - 1.5
        center_lon = tile_lon_min + 1.5

        lat_dir = "N" if center_lat >= 0 else "S"
        lon_dir = "E" if center_lon >= 0 else "W"

        return f"ESA_WorldCover_10m_{self.year}_{self.version}_{lat_dir}{abs(int(center_lat)):02d}{lon_dir}{abs(int(center_lon)):03d}_Map.tif"

    def _tile_url(self, tile_name: str) -> str:
        return f"{self.base_url}/{tile_name}"

    def _tile_path(self, tile_name: str) -> Path:
        return self.cache_dir / tile_name

    def download_tile(self, lat: float, lon: float) -> Path | None:
        """Download a single WorldCover tile if not cached."""
        tile_name = self._tile_name(lat, lon)
        tile_path = self._tile_path(tile_name)

        if tile_path.exists():
            return tile_path

        url = self._tile_url(tile_name)
        try:
            logger.info(f"Downloading WorldCover tile: {tile_name}")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(tile_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

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
        """Download all WorldCover tiles covering a bounding box."""
        # Determine tile range
        min_tx, max_ty = self._tile_coords(max_lat, min_lon)  # top-left
        max_tx, min_ty = self._tile_coords(min_lat, max_lon)  # bottom-right

        tiles_needed = []
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                # Convert tile coords back to lat/lon for download
                tile_lon_min = -180 + tx * 3
                tile_lat_max = 84 - ty * 3
                center_lat = tile_lat_max - 1.5
                center_lon = tile_lon_min + 1.5
                tiles_needed.append((center_lat, center_lon))

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

    def read_lulc_at_points(
        self,
        points: list[tuple[float, float]],  # [(lat, lon), ...]
        buffer_m: float = 500,  # 500m buffer for majority class
    ) -> dict[tuple[float, float], dict]:
        """Read land cover class at points with buffer majority voting.

        Args:
            points: List of (lat, lon) tuples.
            buffer_m: Buffer radius in meters for majority vote.

        Returns:
            Dict mapping (lat, lon) -> {lulc_class, lulc_group, lulc_name}
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("rasterio not available, returning default class")
            return {
                p: {"lulc_class": 0, "lulc_group": "Unknown", "lulc_name": "Unknown"}
                for p in points
            }

        # Determine required tiles
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        min_lat, max_lat = min(lats) - 0.01, max(lats) + 0.01
        min_lon, max_lon = min(lons) - 0.01, max(lons) + 0.01

        # Download tiles
        self.download_tiles_for_bbox(min_lat, min_lon, max_lat, max_lon)

        results = {}
        for lat, lon in points:
            tile_path = self._tile_path(self._tile_name(lat, lon))
            if not tile_path.exists():
                results[(lat, lon)] = {
                    "lulc_class": 0,
                    "lulc_group": "Unknown",
                    "lulc_name": "Unknown",
                }
                continue

            try:
                class_id, group = self._extract_lulc(tile_path, lat, lon, buffer_m)
                results[(lat, lon)] = {
                    "lulc_class": class_id,
                    "lulc_group": group,
                    "lulc_name": LULC_CLASSES.get(class_id, "Unknown"),
                }
            except Exception as e:
                logger.error(f"Failed to extract LULC from {tile_path}: {e}")
                results[(lat, lon)] = {
                    "lulc_class": 0,
                    "lulc_group": "Unknown",
                    "lulc_name": "Unknown",
                }

        return results

    def _extract_lulc(
        self,
        tif_path: Path,
        lat: float,
        lon: float,
        buffer_m: float,
    ) -> tuple[int, str]:
        """Extract land cover class from WorldCover GeoTIFF at lat/lon with buffer."""
        with rasterio.open(tif_path) as src:
            # WorldCover is in EPSG:4326 (WGS84)
            # Convert buffer meters to degrees (approximate)
            buffer_m / 111000

            # Window around point
            row, col = src.index(lon, lat)

            # Calculate pixel window for buffer
            # 10m resolution -> 1 pixel = 10m
            pixel_buffer = max(1, int(buffer_m / 10))

            window = rasterio.windows.Window(
                col - pixel_buffer,
                row - pixel_buffer,
                2 * pixel_buffer + 1,
                2 * pixel_buffer + 1,
            )

            # Clamp to valid range
            window = window.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )

            data = src.read(1, window=window)

            # Majority vote (ignore 0/nodata)
            valid = data[data > 0]
            if len(valid) == 0:
                return 0, "Unknown"

            # Most frequent class
            unique, counts = np.unique(valid, return_counts=True)
            majority_class = int(unique[np.argmax(counts)])

            return majority_class, CLASS_TO_GROUP.get(majority_class, "Unknown")

    def compute_grid_features(
        self,
        grid_cells: list[dict],
    ) -> dict[str, dict]:
        """Compute land cover features for all grid cells.

        Args:
            grid_cells: List of dicts with grid_cell_id, centroid_lat, centroid_lon.

        Returns:
            Dict mapping grid_cell_id -> {lulc_class, lulc_group, lulc_name}
        """
        points = [(c["centroid_lat"], c["centroid_lon"]) for c in grid_cells]
        results = self.read_lulc_at_points(points)

        features = {}
        for cell in grid_cells:
            key = (cell["centroid_lat"], cell["centroid_lon"])
            features[cell["grid_cell_id"]] = results.get(
                key,
                {
                    "lulc_class": 0,
                    "lulc_group": "Unknown",
                    "lulc_name": "Unknown",
                },
            )
        return features
