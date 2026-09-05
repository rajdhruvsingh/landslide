"""Feature engineering pipeline for the risk classifier.

Planned features (Phase 2):
    - Rolling rainfall aggregates: 3/7/15/30-day sums
    - Antecedent moisture index
    - Slope angle and aspect from DEM
    - Distance to nearest historical landslide point
    - Road density within buffer
    - Land use/land cover category (if Bhuvan data available)

This module is a stub for Phase 0. Implement in Phase 2.
"""


def compute_features(station_data: dict) -> dict:
    """Compute feature vector for a single station/location.

    Args:
        station_data: Raw readings and static attributes.

    Returns:
        dict of feature_name -> value.
    """
    return {
        "rainfall_3d_mm": 0.0,
        "rainfall_7d_mm": 0.0,
        "rainfall_15d_mm": 0.0,
        "rainfall_30d_mm": 0.0,
        "antecedent_moisture_index": 0.0,
        "slope_angle_deg": 0.0,
        "slope_aspect_deg": 0.0,
        "distance_to_nearest_landslide_km": 0.0,
        "road_density_per_km2": 0.0,
        "lulc_category": 0,
    }
