import os
import sys
import time
import json
import zipfile
import logging
import traceback
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona

from shapely.geometry import LineString, Point, box
from tqdm import tqdm

# Monkey‑patch missing internal module (some envs require this)
sys.modules["numpy._core.numeric"] = np.core.numeric

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIG
# ============================================================

# Candidates
INPUT_FILE  = "Features.pkl"

# GeoPackage (Geofabrik style)
GPKG_ZIP  = "new-york-260522-free.gpkg.zip"
GPKG_FILE = "new-york-260522-free.gpkg"   # created if zip exists

# Output
OUTPUT_FILE = "meter_floc_candidates_with_spatial_features.csv"
LOG_FILE    = "run_diagnostics_local_gpkg.log"

# --- NEW: checkpoint (resume) ---
CHECKPOINT_FILE = "processed_tiles.json"  # stores completed tile ids
CHECKPOINT_EVERY = 1                      # write checkpoint every N successful tiles (1 = safest)

# CRS
TARGET_CRS = "EPSG:26918"   # NAD83 / UTM Zone 18N
WGS84 = "EPSG:4326"

# AOI buffer for each tile (meters)
AOI_BUFFER_METERS = 20

# Strategy B tiling
GRID_SIZE_M = 50            # 500 / 1000 / 2000
GRID_PADDING_M = 500        # pad bounds for grid creation

# Optional filters
TARGET_DIVISIONS = ["Ithaca"]  # e.g. ["Canandaigua"] or None
LIMIT_ROWS = None              # None = process all filtered rows

# Candidate column mapping (adjust ONLY if your pickle differs)
COL_METER_LON = "LONGITUDE"
COL_METER_LAT = "LATITUDE"
COL_FLOC_LON  = "Location_X"
COL_FLOC_LAT  = "Location_Y"

COL_DIVISION  = "Division"
COL_METER_ID  = "meter_number"
COL_CANDIDATE = "Candidate_Transformer"
COL_DIST_FT   = "candidate_distance_ft"

# Town not present in your pickle
COL_TOWN = None

# Guards / tuning
MAX_BUILDING_HITS = 30000
INTERSECTION_CORRIDOR_M = None  # set 2–5 if dense areas create too many building hits

# Output behavior
APPEND_OUTPUT = True
SAVE_FAILED_TILES = True
WRITE_DEBUG_TILES = False  # creates tile_####.csv (many files)

# --- NEW: safer append to avoid partial-write corruption ---
SAFE_TILE_APPEND = True    # write tile output to temp file then append into OUTPUT_FILE


# ============================================================
# LOGGING
# ============================================================

def setup_logger():
    logger = logging.getLogger("local_gpkg")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def split(self):
        now = time.perf_counter()
        dt = now - self.t0
        self.t0 = now
        return dt


# ============================================================
# CHECKPOINT HELPERS (Option 1)
# ============================================================

def load_checkpoint(path):
    """
    Returns a set of tile_ids already completed.
    """
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            done = set(int(x) for x in data.get("done_tiles", []))
            return done
        except Exception:
            logger.warning(f"Could not read checkpoint '{path}'. Starting fresh.")
    return set()

def save_checkpoint(path, done_tiles):
    """
    Crash-safe write (atomic replace).
    """
    tmp = path + ".tmp"
    payload = {"done_tiles": sorted(list(done_tiles))}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def mark_tile_done(tid, done_tiles, path, success_counter, every=1):
    done_tiles.add(int(tid))
    if every <= 1 or (success_counter % every) == 0:
        save_checkpoint(path, done_tiles)

def file_exists_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# ============================================================
# HELPERS
# ============================================================

def ensure_gpkg_available(gpkg_zip, gpkg_file):
    """
    If gpkg_file exists, return it.
    If only gpkg_zip exists, unzip first .gpkg into gpkg_file and return it.
    """
    if os.path.exists(gpkg_file):
        return gpkg_file

    if not os.path.exists(gpkg_zip):
        raise FileNotFoundError(f"Neither '{gpkg_file}' nor '{gpkg_zip}' exists in the working directory.")

    logger.info(f"Unzipping {gpkg_zip} ...")
    with zipfile.ZipFile(gpkg_zip, "r") as z:
        gpkg_members = [m for m in z.namelist() if m.lower().endswith(".gpkg")]
        if not gpkg_members:
            raise ValueError(f"No .gpkg found inside {gpkg_zip}. Zip contents sample: {z.namelist()[:20]}")
        member = gpkg_members[0]
        z.extract(member, ".")
        extracted_path = os.path.join(".", member)

        if os.path.abspath(extracted_path) != os.path.abspath(gpkg_file):
            os.replace(extracted_path, gpkg_file)

    logger.info(f"GeoPackage ready: {gpkg_file}")
    return gpkg_file


def build_grid(bounds, grid_size_m, crs, padding_m=0):
    minx, miny, maxx, maxy = bounds
    minx -= padding_m
    miny -= padding_m
    maxx += padding_m
    maxy += padding_m

    xs = np.arange(minx, maxx + grid_size_m, grid_size_m)
    ys = np.arange(miny, maxy + grid_size_m, grid_size_m)

    cell_ids, geoms = [], []
    cid = 0
    for x in xs[:-1]:
        for y in ys[:-1]:
            cell_ids.append(cid)
            geoms.append(box(x, y, x + grid_size_m, y + grid_size_m))
            cid += 1

    return gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=geoms, crs=crs)


def clean_layer(gdf, target_crs):
    """
    Ensure geometry valid, ensure CRS, reproject to target CRS.
    """
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)
    gdf = gdf[gdf.geometry.notna()].copy()

    try:
        gdf = gdf[gdf.geometry.is_valid].copy()
    except Exception:
        pass

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)

    return gdf.to_crs(target_crs)


def empty_gdf(crs):
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def aoi_clip(gdf, aoi_geom):
    """
    Fast subset using spatial index bbox prefilter + intersects refine.
    gdf and aoi_geom must be in same CRS.
    """
    if gdf is None or gdf.empty:
        return gdf

    _ = gdf.sindex  # build spatial index (lazy)
    minx, miny, maxx, maxy = aoi_geom.bounds
    idx = list(gdf.sindex.intersection((minx, miny, maxx, maxy)))
    if not idx:
        return gdf.iloc[0:0].copy()

    subset = gdf.iloc[idx].copy()
    return subset[subset.intersects(aoi_geom)]


def read_layer_safe(gpkg_path, layer_name, columns=None):
    """
    Read a layer from GPKG with optional column selection (if supported).
    Always reprojects to TARGET_CRS.
    """
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name, columns=columns)
    except TypeError:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

    return clean_layer(gdf, TARGET_CRS)


def load_osm_layers_from_geofabrik_gpkg(gpkg_path):
    """
    Loads OSM-like layers from the Geofabrik GIS GeoPackage schema you have:
      - buildings polygons: gis_osm_buildings_a_free
      - roads lines: gis_osm_roads_free
      - waterways lines: gis_osm_waterways_free
      - railways lines: gis_osm_railways_free
      - landuse polygons: gis_osm_landuse_a_free

    Barriers are not guaranteed as a standalone layer in this dataset.
    We'll attempt to derive barriers from traffic layers if possible, otherwise empty.
    """
    layers = fiona.listlayers(gpkg_path)
    layers_lower = {lyr.lower(): lyr for lyr in layers}
    logger.info(f"GPKG layers found: {layers}")

    # Required layers in your file
    buildings_layer = layers_lower.get("gis_osm_buildings_a_free")
    roads_layer     = layers_lower.get("gis_osm_roads_free")
    waterways_layer = layers_lower.get("gis_osm_waterways_free")
    railways_layer  = layers_lower.get("gis_osm_railways_free")
    landuse_layer   = layers_lower.get("gis_osm_landuse_a_free")

    missing = [n for n, v in [
        ("gis_osm_buildings_a_free", buildings_layer),
        ("gis_osm_roads_free", roads_layer),
        ("gis_osm_waterways_free", waterways_layer),
        ("gis_osm_railways_free", railways_layer),
        ("gis_osm_landuse_a_free", landuse_layer),
    ] if v is None]

    if missing:
        raise ValueError(
            f"Missing expected Geofabrik layers: {missing}. Available layers: {layers}"
        )

    # Read minimal columns where possible (helps memory)
    buildings = read_layer_safe(gpkg_path, buildings_layer, columns=["osm_id", "name", "code", "fclass", "geometry"])
    roads     = read_layer_safe(gpkg_path, roads_layer,     columns=["osm_id", "name", "ref", "code", "fclass", "geometry"])
    waterways = read_layer_safe(gpkg_path, waterways_layer, columns=["osm_id", "name", "code", "fclass", "geometry"])
    railways  = read_layer_safe(gpkg_path, railways_layer,  columns=["osm_id", "name", "code", "fclass", "geometry"])
    landuse   = read_layer_safe(gpkg_path, landuse_layer,   columns=["osm_id", "name", "code", "fclass", "geometry"])

    # Attempt to derive barriers (optional)
    barriers = empty_gdf(TARGET_CRS)
    traffic_layer = layers_lower.get("gis_osm_traffic_free")
    traffic_a_layer = layers_lower.get("gis_osm_traffic_a_free")

    traffic_to_try = traffic_layer or traffic_a_layer
    if traffic_to_try:
        try:
            traffic = read_layer_safe(gpkg_path, traffic_to_try, columns=["osm_id", "name", "code", "fclass", "geometry"])
            if "fclass" in traffic.columns:
                barrier_like = {"gate", "bollard", "block", "barrier", "lift_gate"}
                barriers = traffic[traffic["fclass"].isin(barrier_like)].copy()
                if barriers.empty:
                    barriers = empty_gdf(TARGET_CRS)
        except Exception:
            barriers = empty_gdf(TARGET_CRS)

    return {
        "buildings": buildings,
        "roads": roads,
        "waterways": waterways,
        "railways": railways,
        "barriers": barriers,
        "landuse": landuse
    }


# ============================================================
# FEATURE ENGINEERING (your original logic)
# ============================================================

def add_crossing_count(lines_gdf, layer_gdf, feature_name, tile_tag=""):
    """
    Adds:
      - {feature_name}_crossing_count
      - {feature_name}_crossing_flag
    """
    if layer_gdf is None or layer_gdf.empty:
        lines_gdf[f"{feature_name}_crossing_count"] = 0
        lines_gdf[f"{feature_name}_crossing_flag"] = 0
        return lines_gdf

    layer = layer_gdf[["geometry"]].copy()

    t = Timer()
    hits = gpd.sjoin(lines_gdf, layer, how="left", predicate="intersects")
    dt = t.split()

    counts = (
        hits.dropna(subset=["index_right"])
            .groupby([COL_METER_ID, COL_CANDIDATE])
            .size()
            .reset_index(name=f"{feature_name}_crossing_count")
    )

    out = lines_gdf.merge(counts, on=[COL_METER_ID, COL_CANDIDATE], how="left")
    out[f"{feature_name}_crossing_count"] = out[f"{feature_name}_crossing_count"].fillna(0).astype(int)
    out[f"{feature_name}_crossing_flag"] = (out[f"{feature_name}_crossing_count"] > 0).astype(int)

    logger.info(
        f"{tile_tag} crossing '{feature_name}': "
        f"layer={len(layer_gdf):,} join_time={dt:.2f}s nonzero={(out[f'{feature_name}_crossing_count']>0).sum():,}"
    )
    return out


def add_building_blocked_length(lines_gdf, buildings_gdf, tile_tag=""):
    """
    Adds:
      - building_count_between
      - building_between_flag
      - blocked_length_ft
      - pct_line_blocked_by_building
    """
    if buildings_gdf is None or buildings_gdf.empty:
        lines_gdf["building_count_between"] = 0
        lines_gdf["building_between_flag"] = 0
        lines_gdf["blocked_length_ft"] = 0.0
        lines_gdf["pct_line_blocked_by_building"] = 0.0
        logger.info(f"{tile_tag} buildings: empty")
        return lines_gdf

    t = Timer()
    b = buildings_gdf[buildings_gdf.geometry.notna()].copy()
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    try:
        b = b[b.geometry.is_valid].copy()
    except Exception:
        pass

    if b.empty:
        lines_gdf["building_count_between"] = 0
        lines_gdf["building_between_flag"] = 0
        lines_gdf["blocked_length_ft"] = 0.0
        lines_gdf["pct_line_blocked_by_building"] = 0.0
        logger.info(f"{tile_tag} buildings: none after filtering")
        return lines_gdf

    b = b.copy()
    b["BuildingID"] = b.index.astype(str)

    # Optional corridor join to reduce hits
    if INTERSECTION_CORRIDOR_M is not None and INTERSECTION_CORRIDOR_M > 0:
        corridor = lines_gdf[[COL_METER_ID, COL_CANDIDATE, "geometry"]].copy()
        corridor["geometry"] = corridor.geometry.buffer(INTERSECTION_CORRIDOR_M)
        corridor = gpd.GeoDataFrame(corridor, geometry="geometry", crs=lines_gdf.crs)

        hits = gpd.sjoin(corridor, b[["BuildingID", "geometry"]], how="left", predicate="intersects")

        # restore original line geometry for intersection computation
        hits = hits.drop(columns=["geometry"], errors="ignore").merge(
            lines_gdf[[COL_METER_ID, COL_CANDIDATE, "geometry"]],
            on=[COL_METER_ID, COL_CANDIDATE],
            how="left"
        )
    else:
        hits = gpd.sjoin(lines_gdf, b[["BuildingID", "geometry"]], how="left", predicate="intersects")

    dt_sjoin = t.split()
    valid_hits = hits.dropna(subset=["BuildingID"]).copy()

    logger.info(f"{tile_tag} buildings: layer={len(b):,} sjoin_time={dt_sjoin:.2f}s hits={len(valid_hits):,}")

    if len(valid_hits) > MAX_BUILDING_HITS:
        raise RuntimeError(
            f"{tile_tag} Too many building hits ({len(valid_hits):,}) > MAX_BUILDING_HITS={MAX_BUILDING_HITS:,}. "
            f"Reduce GRID_SIZE_M or set INTERSECTION_CORRIDOR_M=2..5"
        )

    building_count = (
        valid_hits.groupby([COL_METER_ID, COL_CANDIDATE])
                  .size()
                  .reset_index(name="building_count_between")
    )

    building_geom = b[["BuildingID", "geometry"]].rename(columns={"geometry": "building_geom"})
    valid_hits = valid_hits.merge(building_geom, on="BuildingID", how="left")
    valid_hits = gpd.GeoDataFrame(valid_hits, geometry="geometry", crs=lines_gdf.crs)

    # Safe intersection loop
    t_int = Timer()
    blocked_lengths = []

    for g_line, g_bldg in tqdm(
        zip(valid_hits.geometry.values, valid_hits["building_geom"].values),
        total=len(valid_hits),
        desc=f"{tile_tag} intersections",
        unit="hit",
        leave=False
    ):
        try:
            if g_line is None or g_bldg is None:
                blocked_lengths.append(0.0)
            else:
                inter = g_line.intersection(g_bldg)
                blocked_lengths.append((inter.length * 3.28084) if inter is not None else 0.0)
        except Exception:
            blocked_lengths.append(0.0)

    valid_hits["blocked_length_ft"] = blocked_lengths
    logger.info(f"{tile_tag} buildings: intersection_time={t_int.split():.2f}s")

    blocked_length = (
        valid_hits.groupby([COL_METER_ID, COL_CANDIDATE])["blocked_length_ft"]
                 .sum()
                 .reset_index()
    )

    out = lines_gdf.merge(building_count, on=[COL_METER_ID, COL_CANDIDATE], how="left")
    out = out.merge(blocked_length, on=[COL_METER_ID, COL_CANDIDATE], how="left")

    out["building_count_between"] = out["building_count_between"].fillna(0).astype(int)
    out["blocked_length_ft"] = out["blocked_length_ft"].fillna(0.0)
    out["building_between_flag"] = (out["building_count_between"] > 0).astype(int)

    out["pct_line_blocked_by_building"] = (
        out["blocked_length_ft"] / out["straight_line_length_ft"].replace({0: np.nan})
    ).fillna(0.0)

    return out


# ============================================================
# TILE PROCESSOR (LOCAL GPKG)
# ============================================================

def process_tile(tile_df, tile_tag, osm_layers):
    """
    Build lines, AOI, subset local layers by AOI, compute features.
    """
    # ---- points ----
    t_points = Timer()
    meter_points = gpd.GeoSeries(
        gpd.points_from_xy(tile_df[COL_METER_LON], tile_df[COL_METER_LAT]),
        crs=WGS84
    ).to_crs(TARGET_CRS)

    floc_points = gpd.GeoSeries(
        gpd.points_from_xy(tile_df[COL_FLOC_LON], tile_df[COL_FLOC_LAT]),
        crs=WGS84
    ).to_crs(TARGET_CRS)

    logger.info(f"{tile_tag} phase points: {t_points.split():.2f}s")

    # ---- lines ----
    t_lines = Timer()
    df = tile_df.copy()
    df["meter_geom"] = meter_points.values
    df["floc_geom"]  = floc_points.values

    df["geometry"] = [
        LineString([mg, fg])
        for mg, fg in zip(df["meter_geom"].values, df["floc_geom"].values)
    ]

    lines = gpd.GeoDataFrame(df, geometry="geometry", crs=TARGET_CRS)
    lines["straight_line_length_ft"] = lines.geometry.length * 3.28084
    logger.info(f"{tile_tag} phase lines: {t_lines.split():.2f}s")

    # ---- AOI ----
    t_aoi = Timer()
    try:
        union_geom = lines.geometry.union_all()
    except Exception:
        union_geom = lines.unary_union

    aoi = union_geom.buffer(AOI_BUFFER_METERS)
    logger.info(f"{tile_tag} phase AOI: {t_aoi.split():.2f}s")

    # ---- subset local layers ----
    t_sub = Timer()
    buildings = aoi_clip(osm_layers["buildings"], aoi)
    roads     = aoi_clip(osm_layers["roads"], aoi)
    waterways = aoi_clip(osm_layers["waterways"], aoi)
    railways  = aoi_clip(osm_layers["railways"], aoi)
    barriers  = aoi_clip(osm_layers["barriers"], aoi)  # might be empty
    landuse   = aoi_clip(osm_layers["landuse"], aoi)

    logger.info(
        f"{tile_tag} phase subset: {t_sub.split():.2f}s | "
        f"buildings={len(buildings):,} roads={len(roads):,} waterways={len(waterways):,} "
        f"railways={len(railways):,} barriers={len(barriers):,} landuse={len(landuse):,}"
    )

    # ---- features ----
    features = lines.copy()

    features = add_building_blocked_length(features, buildings, tile_tag=tile_tag)
    features = add_crossing_count(features, roads, "street", tile_tag=tile_tag)
    features = add_crossing_count(features, waterways, "waterway", tile_tag=tile_tag)
    features = add_crossing_count(features, railways, "railway", tile_tag=tile_tag)
    features = add_crossing_count(features, barriers, "barrier", tile_tag=tile_tag)
    features = add_crossing_count(features, landuse, "landuse", tile_tag=tile_tag)

    # ---- major/local roads split ----
    road_class_col = None
    for c in ["fclass", "highway", "class", "type"]:
        if c in roads.columns:
            road_class_col = c
            break

    major_road_types = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    local_road_types = {"residential", "service", "unclassified", "living_street"}

    if (not roads.empty) and road_class_col is not None:
        def classify(val, allowed):
            if isinstance(val, list):
                return any(v in allowed for v in val)
            return val in allowed

        roads_major = roads[roads[road_class_col].apply(lambda x: classify(x, major_road_types))].copy()
        roads_local = roads[roads[road_class_col].apply(lambda x: classify(x, local_road_types))].copy()

        logger.info(f"{tile_tag} roads_major={len(roads_major):,} roads_local={len(roads_local):,} using '{road_class_col}'")

        features = add_crossing_count(features, roads_major, "major_road", tile_tag=tile_tag)
        features = add_crossing_count(features, roads_local, "local_road", tile_tag=tile_tag)
    else:
        features["major_road_crossing_count"] = 0
        features["major_road_crossing_flag"] = 0
        features["local_road_crossing_count"] = 0
        features["local_road_crossing_flag"] = 0
        logger.info(f"{tile_tag} major/local road split skipped (no road class column found)")

    # ---- obstruction score ----
    features["spatial_obstruction_score"] = (
        features["building_between_flag"] * 2.0
        + features["building_count_between"] * 0.5
        + features["waterway_crossing_count"] * 3.0
        + features["major_road_crossing_count"] * 1.5
        + features["local_road_crossing_count"] * 0.3
        + features["railway_crossing_count"] * 2.0
        + features["barrier_crossing_count"] * 2.0
    )

    return features


# ============================================================
# OUTPUT APPEND (safer)
# ============================================================

def append_tile_to_output(features_tile, available_cols, output_path, tile_id):
    """
    Safer than direct mode='a' write:
    - write tile to temp CSV fully
    - append temp contents to final output
    """
    header_needed = not file_exists_nonempty(output_path)

    if not SAFE_TILE_APPEND:
        # Direct append
        features_tile[available_cols].to_csv(
            output_path,
            mode="a",
            header=header_needed,
            index=False
        )
        return

    # Safer: tile temp then append
    tile_tmp = f"_tile_{tile_id}.tmp.csv"
    features_tile[available_cols].to_csv(tile_tmp, index=False)

    with open(output_path, "a", encoding="utf-8", newline="") as out_f:
        with open(tile_tmp, "r", encoding="utf-8") as in_f:
            if not header_needed:
                # skip header line of temp file
                next(in_f, None)
            out_f.write(in_f.read())
        out_f.flush()
        os.fsync(out_f.fileno())

    os.remove(tile_tmp)


# ============================================================
# MAIN
# ============================================================

def main():
    gpkg = ensure_gpkg_available(GPKG_ZIP, GPKG_FILE)

    logger.info(f"Loading candidates: {INPUT_FILE}")
    candidates = pd.read_pickle(INPUT_FILE)

    # Validate columns
    required_cols = [
        COL_METER_LON, COL_METER_LAT, COL_FLOC_LON, COL_FLOC_LAT,
        COL_DIVISION, COL_METER_ID, COL_CANDIDATE
    ]
    missing = [c for c in required_cols if c not in candidates.columns]
    if missing:
        raise ValueError(f"Missing required columns in candidates pickle: {missing}")

    df = candidates.copy()
    if TARGET_DIVISIONS is not None:
        df = df[df[COL_DIVISION].isin(TARGET_DIVISIONS)].copy()

    if LIMIT_ROWS is not None:
        df = df.head(LIMIT_ROWS).copy()

    logger.info(f"Filtered rows: {len(df):,}")

    # ---- Build midpoint points for tile assignment ----
    t_assign = Timer()
    meter_pts = gpd.GeoSeries(
        gpd.points_from_xy(df[COL_METER_LON], df[COL_METER_LAT]),
        crs=WGS84
    ).to_crs(TARGET_CRS)

    floc_pts = gpd.GeoSeries(
        gpd.points_from_xy(df[COL_FLOC_LON], df[COL_FLOC_LAT]),
        crs=WGS84
    ).to_crs(TARGET_CRS)

    mid_x = (meter_pts.x.values + floc_pts.x.values) / 2.0
    mid_y = (meter_pts.y.values + floc_pts.y.values) / 2.0
    mid_pts = gpd.GeoSeries([Point(x, y) for x, y in zip(mid_x, mid_y)], crs=TARGET_CRS)

    assign_gdf = gpd.GeoDataFrame(df.copy(), geometry=mid_pts, crs=TARGET_CRS)
    logger.info(f"Built tile assignment points in {t_assign.split():.2f}s")

    # ---- Grid + assign tiles ----
    grid = build_grid(assign_gdf.total_bounds, GRID_SIZE_M, TARGET_CRS, padding_m=GRID_PADDING_M)
    assigned = gpd.sjoin(assign_gdf, grid, how="left", predicate="intersects")
    assigned = assigned.drop(columns=["index_right"], errors="ignore")

    tile_ids = sorted(assigned["cell_id"].dropna().astype(int).unique().tolist())
    logger.info(f"Tiles with data: {len(tile_ids):,} | GRID_SIZE_M={GRID_SIZE_M}m | AOI_BUFFER={AOI_BUFFER_METERS}m")

    # ---- Load OSM layers from GPKG once ----
    logger.info(f"Loading OSM layers from local GeoPackage: {gpkg}")
    t_osm = Timer()
    osm_layers = load_osm_layers_from_geofabrik_gpkg(gpkg)

    # Build spatial indexes once (big speedup)
    for k, layer in osm_layers.items():
        if layer is not None and not layer.empty:
            _ = layer.sindex

    logger.info(
        f"Loaded local layers in {t_osm.split():.2f}s | "
        f"buildings={len(osm_layers['buildings']):,} roads={len(osm_layers['roads']):,} "
        f"waterways={len(osm_layers['waterways']):,} railways={len(osm_layers['railways']):,} "
        f"barriers={len(osm_layers['barriers']):,} landuse={len(osm_layers['landuse']):,}"
    )

    # ---- Output columns ----
    output_cols = [
        COL_METER_ID,
        COL_CANDIDATE,
        COL_DIVISION,
        COL_DIST_FT,
        "straight_line_length_ft",

        "building_between_flag",
        "building_count_between",
        "blocked_length_ft",
        "pct_line_blocked_by_building",

        "street_crossing_flag",
        "street_crossing_count",
        "major_road_crossing_flag",
        "major_road_crossing_count",
        "local_road_crossing_flag",
        "local_road_crossing_count",

        "waterway_crossing_flag",
        "waterway_crossing_count",
        "railway_crossing_flag",
        "railway_crossing_count",
        "barrier_crossing_flag",
        "barrier_crossing_count",

        "landuse_crossing_flag",
        "landuse_crossing_count",

        "spatial_obstruction_score"
    ]

    # ---- Checkpoint load (resume) ----
    done_tiles = load_checkpoint(CHECKPOINT_FILE)
    logger.info(f"Checkpoint loaded: {len(done_tiles):,} tiles already completed (from {CHECKPOINT_FILE})")

    # Important:
    # - If APPEND_OUTPUT=True, we DO NOT delete OUTPUT_FILE because we want resume.
    # - If APPEND_OUTPUT=False, we start clean.
    if not APPEND_OUTPUT:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            logger.info(f"Removed existing output file (APPEND_OUTPUT=False): {OUTPUT_FILE}")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            done_tiles = set()
            logger.info(f"Removed existing checkpoint (APPEND_OUTPUT=False): {CHECKPOINT_FILE}")

    failed_tiles = []
    overall = Timer()
    success_counter = 0

    # ---- Tile loop ----
    for idx, tid in enumerate(tqdm(tile_ids, desc="Processing tiles (LOCAL GPKG)", unit="tile"), start=1):
        tid_int = int(tid)

        # Resume: skip tiles already completed
        if tid_int in done_tiles:
            continue

        tile_tag = f"[Tile {idx}/{len(tile_ids)} | cell_id={tid_int}]"
        t_tile = Timer()

        tile_df = pd.DataFrame(
            assigned[assigned["cell_id"] == tid_int].drop(columns=["geometry"], errors="ignore")
        ).copy()

        if tile_df.empty:
            # Nothing to do, still mark done to avoid revisiting
            success_counter += 1
            mark_tile_done(tid_int, done_tiles, CHECKPOINT_FILE, success_counter, every=CHECKPOINT_EVERY)
            continue

        logger.info(f"{tile_tag} rows={len(tile_df):,}")

        try:
            features_tile = process_tile(tile_df, tile_tag, osm_layers)

            available_cols = [c for c in output_cols if c in features_tile.columns]

            # Write output for this tile (crash-safe-ish)
            append_tile_to_output(features_tile, available_cols, OUTPUT_FILE, tid_int)

            # Only after successful write: mark tile done in checkpoint
            success_counter += 1
            mark_tile_done(tid_int, done_tiles, CHECKPOINT_FILE, success_counter, every=CHECKPOINT_EVERY)

            logger.info(f"{tile_tag} DONE in {t_tile.split():.2f}s | wrote_rows={len(features_tile):,}")

            if WRITE_DEBUG_TILES:
                dbg = f"tile_{tid_int}_features.csv"
                features_tile[available_cols].to_csv(dbg, index=False)
                logger.info(f"{tile_tag} wrote debug tile output: {dbg}")

        except Exception as e:
            msg = f"❌ {tile_tag} FAILED: {e}"
            tqdm.write(msg)
            logger.error(msg)
            logger.error(traceback.format_exc())

            failed_tiles.append((tid_int, str(e)))

            if SAVE_FAILED_TILES:
                tile_df.to_csv(f"failed_tile_{tid_int}.csv", index=False)
                logger.info(f"{tile_tag} saved failing tile input: failed_tile_{tid_int}.csv")

            # Do NOT mark as done; it will be retried on next run
            continue

    # Final checkpoint flush (in case CHECKPOINT_EVERY > 1)
    save_checkpoint(CHECKPOINT_FILE, done_tiles)

    logger.info("============================================")
    logger.info(f"ALL DONE in {overall.split():.2f}s")
    logger.info(f"Output saved to: {OUTPUT_FILE}")
    logger.info(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    logger.info(f"Failed tiles: {len(failed_tiles):,}")

    print("\n✅ Finished.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Failed tiles: {len(failed_tiles)}")
    if failed_tiles:
        print("See log for details:", LOG_FILE)


if __name__ == "__main__":
    main()