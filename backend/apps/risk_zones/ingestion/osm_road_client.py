"""OpenStreetMap road network ingestion client.

Primary source: OpenStreetMap via Overpass API (free, public).
Alternative: Geofabrik OSM extracts (downloadable PBF/Shapefile for regions).

This module:
1. Queries Overpass API for road network in bounding box
2. Computes distance from grid cell centroids to nearest road
3. Computes road density within buffers

Road types (highway=*):
    - motorway, trunk, primary, secondary, tertiary: Major roads
    - unclassified, residential, service: Minor roads
    - track, path: Trails/paths (optional inclusion)

Dependencies:
    - shapely: for geometric operations
    - requests: for Overpass API
    - networkx: optional, for network analysis

Install: pip install shapely requests networkx
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

logger = logging.getLogger(__name__)

# Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Alternative: https://overpass.kumi.systems/api/interpreter

# Road classifications (OSM highway values)
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
MINOR_ROADS = {"unclassified", "residential", "service", "living_street"}
TRACKS_PATHS = {"track", "path", "footway", "cycleway", "bridleway"}
ALL_ROADS = MAJOR_ROADS | MINOR_ROADS | TRACKS_PATHS


@dataclass
class RoadSegment:
    """Represents a road segment from OSM."""

    osmid: int
    highway_type: str
    name: str | None
    geometry: LineString
    length_m: float


class OSMRoadClient:
    """OpenStreetMap Road Network Client using Overpass API."""

    def __init__(
        self,
        cache_dir: str | Path = "data/osm_cache",
        timeout_seconds: int = 180,
        max_query_area_km2: float = 10000,  # Overpass limit ~10k km2
    ):
        """
        Args:
            cache_dir: Directory to cache downloaded data.
            timeout_seconds: HTTP timeout (Overpass queries can be slow).
            max_query_area_km2: Maximum area per query (Overpass has limits).
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.max_query_area_km2 = max_query_area_km2

    def _build_overpass_query(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        include_tracks: bool = False,
    ) -> str:
        """Build Overpass QL query for roads in bounding box."""
        road_types = list(MAJOR_ROADS | MINOR_ROADS)
        if include_tracks:
            road_types.extend(TRACKS_PATHS)

        highway_filter = "|".join(road_types)
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
          way["highway"~"{highway_filter}"]
            ({min_lat},{min_lon},{max_lat},{max_lon});
        );
        out body;
        >;
        out skel qt;
        """
        return query.strip()

    def fetch_roads(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        include_tracks: bool = False,
        use_cache: bool = True,
    ) -> list[RoadSegment]:
        """Fetch road network for a bounding box.

        Args:
            min_lat, min_lon, max_lat, max_lon: Bounding box.
            include_tracks: Include tracks/paths.
            use_cache: Use cached data if available.

        Returns:
            List of RoadSegment objects.
        """
        cache_key = f"roads_{min_lat:.4f}_{min_lon:.4f}_{max_lat:.4f}_{max_lon:.4f}_tracks{include_tracks}.json"
        cache_path = self.cache_dir / cache_key

        if use_cache and cache_path.exists():
            logger.info(f"Loading roads from cache: {cache_path}")
            return self._load_from_cache(cache_path)

        # Check area
        area_km2 = self._bbox_area_km2(min_lat, min_lon, max_lat, max_lon)
        if area_km2 > self.max_query_area_km2:
            logger.warning(
                f"Area {area_km2:.0f} km² exceeds Overpass limit. Splitting..."
            )
            return self._fetch_large_area(
                min_lat, min_lon, max_lat, max_lon, include_tracks
            )

        query = self._build_overpass_query(
            min_lat, min_lon, max_lat, max_lon, include_tracks
        )

        logger.info(
            f"Querying Overpass API for roads in bbox ({min_lat},{min_lon},{max_lat},{max_lon})"
        )
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

        except Exception as e:
            logger.error(f"Overpass query failed: {e}")
            return []

        # Parse OSM elements
        nodes = {
            el["id"]: (el["lon"], el["lat"])
            for el in data["elements"]
            if el["type"] == "node"
        }
        roads = []

        for el in data["elements"]:
            if el["type"] != "way":
                continue

            tags = el.get("tags", {})
            highway = tags.get("highway", "")
            if highway not in ALL_ROADS:
                continue

            # Build LineString from node coordinates
            coords = []
            for node_id in el.get("nodes", []):
                if node_id in nodes:
                    coords.append(nodes[node_id])

            if len(coords) < 2:
                continue

            line = LineString(coords)
            length = self._line_length_m(line)

            roads.append(
                RoadSegment(
                    osmid=el["id"],
                    highway_type=highway,
                    name=tags.get("name"),
                    geometry=line,
                    length_m=length,
                )
            )

        # Save to cache
        if roads:
            self._save_to_cache(roads, cache_path)

        logger.info(f"Fetched {len(roads)} road segments from OSM")
        return roads

    def _fetch_large_area(
        self,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        include_tracks: bool,
    ) -> list[RoadSegment]:
        """Split large area into smaller tiles and query each."""
        # Split into 1° x 1° tiles (~10k km² at equator, less at higher lat)
        lat_step = 1.0
        lon_step = 1.0

        all_roads = []
        lat = math.floor(min_lat)
        while lat < max_lat:
            lon = math.floor(min_lon)
            while lon < max_lon:
                tile_min_lat = max(lat, min_lat)
                tile_min_lon = max(lon, min_lon)
                tile_max_lat = min(lat + lat_step, max_lat)
                tile_max_lon = min(lon + lon_step, max_lon)

                roads = self.fetch_roads(
                    tile_min_lat,
                    tile_min_lon,
                    tile_max_lat,
                    tile_max_lon,
                    include_tracks,
                    use_cache=True,
                )
                all_roads.extend(roads)

                # Rate limiting
                time.sleep(1)
                lon += lon_step
            lat += lat_step

        # Deduplicate by OSM ID
        seen = set()
        unique = []
        for r in all_roads:
            if r.osmid not in seen:
                seen.add(r.osmid)
                unique.append(r)

        return unique

    def _bbox_area_km2(self, min_lat, min_lon, max_lat, max_lon) -> float:
        """Approximate area of bbox in km²."""
        lat_km = (max_lat - min_lat) * 111
        avg_lat = (min_lat + max_lat) / 2
        lon_km = (max_lon - min_lon) * 111 * math.cos(math.radians(avg_lat))
        return lat_km * lon_km

    def _line_length_m(self, line: LineString) -> float:
        """Calculate line length in meters (approximate)."""
        # Simple approximation: sum of segment lengths in degrees * 111km
        length_deg = line.length
        # At centroid latitude
        centroid_lat = line.centroid.y
        return length_deg * 111000 * math.cos(math.radians(centroid_lat))

    def _save_to_cache(self, roads: list[RoadSegment], cache_path: Path) -> None:
        """Save roads to JSON cache."""
        import json

        data = [
            {
                "osmid": r.osmid,
                "highway_type": r.highway_type,
                "name": r.name,
                "geometry": list(r.geometry.coords),
                "length_m": r.length_m,
            }
            for r in roads
        ]
        with open(cache_path, "w") as f:
            json.dump(data, f)

    def _load_from_cache(self, cache_path: Path) -> list[RoadSegment]:
        """Load roads from JSON cache."""
        import json

        with open(cache_path) as f:
            data = json.load(f)
        roads = []
        for d in data:
            roads.append(
                RoadSegment(
                    osmid=d["osmid"],
                    highway_type=d["highway_type"],
                    name=d["name"],
                    geometry=LineString(d["geometry"]),
                    length_m=d["length_m"],
                )
            )
        return roads

    def compute_road_features_for_grid(
        self,
        grid_cells: list[dict],
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float,
        include_tracks: bool = False,
    ) -> dict[str, dict]:
        """Compute road distance and density features for grid cells.

        Args:
            grid_cells: List of dicts with grid_cell_id, centroid_lat, centroid_lon.
            min_lat, min_lon, max_lat, max_lon: Bounding box for road fetch.
            include_tracks: Include tracks/paths.

        Returns:
            Dict mapping grid_cell_id -> {road_distance_km, road_density_per_km2}
        """
        roads = self.fetch_roads(min_lat, min_lon, max_lat, max_lon, include_tracks)
        if not roads:
            return {
                c["grid_cell_id"]: {
                    "road_distance_km": 999.0,
                    "road_density_per_km2": 0.0,
                }
                for c in grid_cells
            }

        # Build spatial index for nearest neighbor queries
        # For simplicity, use brute force (fine for < 10k cells x < 10k roads)
        features = {}

        for cell in grid_cells:
            point = Point(cell["centroid_lon"], cell["centroid_lat"])

            min_dist = float("inf")
            for road in roads:
                # Distance from point to line
                nearest = nearest_points(point, road.geometry)[1]
                dist = point.distance(nearest)
                # Convert to km (approximate)
                dist_km = dist * 111 * math.cos(math.radians(cell["centroid_lat"]))
                if dist_km < min_dist:
                    min_dist = dist_km

            # Road density: total road length within 5km radius / area
            buffer_deg = 5.0 / (111 * math.cos(math.radians(cell["centroid_lat"])))
            buffer_poly = point.buffer(buffer_deg)
            total_len = 0.0
            for road in roads:
                if road.geometry.intersects(buffer_poly):
                    inter = road.geometry.intersection(buffer_poly)
                    total_len += (
                        inter.length
                        * 111000
                        * math.cos(math.radians(cell["centroid_lat"]))
                    )

            area_km2 = math.pi * 25  # 5km radius circle
            density = total_len / 1000 / area_km2  # km/km²

            features[cell["grid_cell_id"]] = {
                "road_distance_km": round(min_dist, 3)
                if min_dist != float("inf")
                else 999.0,
                "road_density_per_km2": round(density, 3),
            }

        return features


# Backward-compatible function
def fetch_road_data(zone_id: int) -> dict | None:
    """Stub for backward compatibility."""
    return None
