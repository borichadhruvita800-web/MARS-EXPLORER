import numpy as np

MARS_RADIUS_M = 3396190.0
PPD = 128                      # MOLA MEGDR 128 pixels per degree
TILE_NORTH = 0.0               # megt00n090hb: top edge latitude
TILE_WEST = 90.0               # megt00n090hb: left edge longitude

# Gale Crater study area
LAT_MIN, LAT_MAX = -17.3, -1.5
LON_MIN, LON_MAX = 129.9, 145.7


def pixel_size_m(lat_deg, ppd=PPD):
    """Ground size of one pixel (dx east-west, dy north-south) in metres."""
    dy = MARS_RADIUS_M * np.radians(1.0 / ppd)
    dx = dy * np.cos(np.radians(lat_deg))
    return float(dx), float(dy)


def extract_region(elevation, lat_min=LAT_MIN, lat_max=LAT_MAX,
                   lon_min=LON_MIN, lon_max=LON_MAX,
                   ppd=PPD, tile_north=TILE_NORTH, tile_west=TILE_WEST):
    """Cut the study area out of the MOLA tile.

    Row 0 is the northern edge, so lats run from north to south.
    """
    elevation = np.asarray(elevation, dtype=float)
    r0 = int(np.floor((tile_north - lat_max) * ppd))
    r1 = int(np.ceil((tile_north - lat_min) * ppd))
    c0 = int(np.floor((lon_min - tile_west) * ppd))
    c1 = int(np.ceil((lon_max - tile_west) * ppd))

    if r0 < 0 or c0 < 0 or r1 > elevation.shape[0] or c1 > elevation.shape[1]:
        raise ValueError("Study area lies outside this MOLA tile.")

    terrain = elevation[r0:r1, c0:c1].copy()
    terrain[~np.isfinite(terrain) | (np.abs(terrain) > 2e4)] = np.nan

    lats = tile_north - (np.arange(r0, r1) + 0.5) / ppd
    lons = tile_west + (np.arange(c0, c1) + 0.5) / ppd
    return {"terrain": terrain, "lats": lats, "lons": lons}