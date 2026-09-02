"""DEM processing methodology documentation (ML-3).

STATUS: No real DEM data available. All terrain features use DEV_FIXTURE values.

When real DEM data is obtained, replace this module with actual processing.
"""

# Terrain dev fixture: synthetic slope/aspect values for pipeline testing.
# These values are NOT derived from any real DEM.
# They are designed to exercise the feature pipeline with realistic ranges
# for the NER region (Sikkim/NE Himalaya).

TERRAIN_DEV_FIXTURE = {
    # Grid cells from the 1km grid used in dev fixtures
    # Format: grid_cell_id -> {slope_angle_deg, slope_aspect_deg, elevation_m}
    "27.3200N_88.6100E": {
        "slope_angle_deg": 28.5,
        "slope_aspect_deg": 225.0,
        "elevation_m": 1850.0,
    },
    "27.3200N_88.6200E": {
        "slope_angle_deg": 15.2,
        "slope_aspect_deg": 180.0,
        "elevation_m": 1720.0,
    },
    "27.3200N_88.6300E": {
        "slope_angle_deg": 35.8,
        "slope_aspect_deg": 315.0,
        "elevation_m": 2100.0,
    },
    "27.3300N_88.6100E": {
        "slope_angle_deg": 12.0,
        "slope_aspect_deg": 90.0,
        "elevation_m": 1650.0,
    },
    "27.3300N_88.6200E": {
        "slope_angle_deg": 42.3,
        "slope_aspect_deg": 270.0,
        "elevation_m": 2350.0,
    },
    "27.3300N_88.6300E": {
        "slope_angle_deg": 8.5,
        "slope_aspect_deg": 45.0,
        "elevation_m": 1580.0,
    },
    "27.3400N_88.6100E": {
        "slope_angle_deg": 22.1,
        "slope_aspect_deg": 135.0,
        "elevation_m": 1920.0,
    },
    "27.3400N_88.6200E": {
        "slope_angle_deg": 38.7,
        "slope_aspect_deg": 0.0,
        "elevation_m": 2280.0,
    },
    "27.3400N_88.6300E": {
        "slope_angle_deg": 18.3,
        "slope_aspect_deg": 360.0,
        "elevation_m": 1780.0,
    },
    "27.3500N_88.6100E": {
        "slope_angle_deg": 5.0,
        "slope_aspect_deg": 200.0,
        "elevation_m": 1500.0,
    },
    "27.3500N_88.6200E": {
        "slope_angle_deg": 31.2,
        "slope_aspect_deg": 330.0,
        "elevation_m": 2050.0,
    },
    "27.3500N_88.6300E": {
        "slope_angle_deg": 45.6,
        "slope_aspect_deg": 160.0,
        "elevation_m": 2500.0,
    },
    "27.3600N_88.6100E": {
        "slope_angle_deg": 20.4,
        "slope_aspect_deg": 290.0,
        "elevation_m": 1820.0,
    },
    "27.3600N_88.6200E": {
        "slope_angle_deg": 10.8,
        "slope_aspect_deg": 110.0,
        "elevation_m": 1640.0,
    },
    "27.3600N_88.6300E": {
        "slope_angle_deg": 26.9,
        "slope_aspect_deg": 250.0,
        "elevation_m": 1950.0,
    },
    "27.3700N_88.6100E": {
        "slope_angle_deg": 33.1,
        "slope_aspect_deg": 20.0,
        "elevation_m": 2180.0,
    },
    "27.3700N_88.6200E": {
        "slope_angle_deg": 16.5,
        "slope_aspect_deg": 150.0,
        "elevation_m": 1750.0,
    },
    "27.3700N_88.6300E": {
        "slope_angle_deg": 40.2,
        "slope_aspect_deg": 70.0,
        "elevation_m": 2420.0,
    },
}

# Note: Grid cells not in this dict default to slope=0, aspect=0, elevation=0.
# This is the correct behavior when DEM data is unavailable.

DEM_PROCESSING_NOTES = """
DEM Processing Methodology
==========================

STATUS: NO REAL DEM DATA IS AVAILABLE.

When real DEM data is obtained, the following methodology will be applied:

1. Data Source
   - Primary: ISRO Bhuvan DEM (30m resolution)
   - Fallback: SRTM GL1 (30m) or ASTER GDEM v3 (30m)

2. Coordinate Reference System
   - Input DEM: As provided (typically WGS84 / EPSG:4326)
   - Processing: Projected to UTM Zone 45N (EPSG:32645) for metric calculations
   - Grid cells defined in WGS84; DEM values extracted in native CRS

3. Slope Angle Computation
   - Method: Horn's method (3x3 window)
   - Input: DEM elevation values
   - Output: Slope in degrees (0-90)
   - Resampled to grid cell centroid using bilinear interpolation

4. Slope Aspect Computation
   - Method: Horn's method (3x3 window)
   - Input: DEM elevation values
   - Output: Aspect in degrees (0=N, 90=E, 180=S, 270=W, 360=N)
   - Flat cells (slope < 0.1°) assigned aspect = 0

5. Elevation
   - Extracted at grid cell centroid using bilinear interpolation

6. Processing Tools
   - GDAL/OGR for DEM reading and reprojection
   - rasterio for feature extraction
   - scipy.ndimage for slope/aspect computation

7. Validation
   - Slope values checked against published susceptibility maps
   - Aspect values verified against known valley orientations
"""
