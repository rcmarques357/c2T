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

sys.modules["numpy._core.numeric"] = np.core.numeric
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "Features.pkl"

GPKG_ZIP = "new-york-260522-free.gpkg.zip"
GPKG_FILE = "new-york-260522-free.gpkg"

OUTPUT_FILE = "meter_floc_candidates_with_spatial_features.csv"
OUTPUT_PARTITION_DIR = "tile_outputs"

LOG_FILE = "run_diagnostics_local_gpkg.log"

CHECKPOINT_FILE = "processed_tiles.json"
TILE_STATUS_FILE = "tile_status.json"
FAILED_TILES_FILE = "failed_tiles.json"

CHECKPOINT_EVERY = 1

TARGET_CRS = "EPSG:26918"
WGS84 = "EPSG:4326"

TILE_AOI_BUFFER_M = 50

GRID_SIZE_M = 100
GRID_PADDING_M = 500

TARGET_DIVISIONS = ["Ithaca"]
LIMIT_ROWS = None

COL_METER_LON = "LONGITUDE"
COL_METER_LAT = "LATITUDE"
COL_FLOC_LON = "Location_X"
COL_FLOC_LAT = "Location_Y"
COL_DIVISION = "Division"
COL_METER_ID = "meter_number"
COL_CANDIDATE = "Candidate_Transformer"
COL_DIST_FT = "candidate_distance_ft"
COL_TOWN = None

MAX_BUILDING_HITS = 30000
INTERSECTION_CORRIDOR_M = None

MAX_ROWS_PER_TILE = 1000
LARGE_TILE_CHUNK_SIZE = 500

APPEND_OUTPUT = True
SAVE_FAILED_TILES = True
WRITE_DEBUG_TILES = False
SAFE_TILE_APPEND = True

USE_PARTITIONED_OUTPUT = True

DRY_RUN_TILE_IDS = None
DRY_RUN_TILE_RANGE = None
MAX_TILES_TO_PROCESS = None

GEOMETRY_REPAIR_ENABLED = True

#RUN FAILED
RETRY_FAILED_TILES = True #When True, after the main processing loop, the script will attempt to reprocess any tiles that were marked as failed and had their input saved. This allows for a second chance to successfully process tiles that may have encountered transient issues during the initial run.
FAILED_TILE_INPUT_PREFIX = "failed_tile_"
RETRY_OUTPUT_PARTITION_DIR = "tile_outputs_retry"

RETRY_TILE_AOI_BUFFER_M = 75
RETRY_MAX_ROWS_PER_TILE = 250
RETRY_LARGE_TILE_CHUNK_SIZE = 100

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
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return set(int(x) for x in data.get("done_tiles", []))
        except Exception:
            logger.warning(f"Could not read checkpoint '{path}'. Starting fresh.")
    return set()


def save_checkpoint(path, done_tiles):
    tmp = path + ".tmp"
    payload = {"done_tiles": sorted(list(done_tiles))}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def mark_tile_done(tid, done_tiles, path, success_counter, every=1):
    done_tiles.add(int(tid))
    if every <= 1 or success_counter % every == 0:
        save_checkpoint(path, done_tiles)


def file_exists_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_json_dict(path, default=None):
    if default is None:
        default = {}

    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning(f"Could not read {path}. Starting with empty status.")
        return default


def save_json_dict(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def update_tile_status(status, tile_id, state, message=None, rows=None):
    status[str(int(tile_id))] = {
        "state": state,
        "message": message,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json_dict(TILE_STATUS_FILE, status)


def validate_tile_output(tile_id):
    if USE_PARTITIONED_OUTPUT:
        path = os.path.join(OUTPUT_PARTITION_DIR, f"tile_{int(tile_id)}.csv")
        return file_exists_nonempty(path)

    return file_exists_nonempty(OUTPUT_FILE)


def repair_geometries(gdf):
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf[gdf.geometry.notna()].copy()

    if not GEOMETRY_REPAIR_ENABLED:
        return gdf

    try:
        invalid_mask = ~gdf.geometry.is_valid

        if invalid_mask.any():
            try:
                gdf.loc[invalid_mask, "geometry"] = (
                    gdf.loc[invalid_mask, "geometry"].make_valid()
                )
            except Exception:
                gdf.loc[invalid_mask, "geometry"] = (
                    gdf.loc[invalid_mask, "geometry"].buffer(0)
                )
    except Exception:
        pass

    return gdf[gdf.geometry.notna()].copy()


def log_tile_diagnostics(tile_tag, tile_df, lines=None, aoi=None):
    msg = f"{tile_tag} diagnostics | rows={len(tile_df):,}"

    if lines is not None and not lines.empty:
        minx, miny, maxx, maxy = lines.total_bounds
        msg += (
            f" | line_count={len(lines):,}"
            f" | bounds=({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})"
        )

    if aoi is not None:
        try:
            msg += f" | aoi_area_m2={aoi.area:,.2f}"
        except Exception:
            pass

    logger.info(msg)


def split_large_tile(tile_df, chunk_size=LARGE_TILE_CHUNK_SIZE):
    for i in range(0, len(tile_df), chunk_size):
        yield i // chunk_size + 1, tile_df.iloc[i:i + chunk_size].copy()


# ============================================================
# HELPERS
# ============================================================

def ensure_gpkg_available(gpkg_zip, gpkg_file):
    if os.path.exists(gpkg_file):
        return gpkg_file

    if not os.path.exists(gpkg_zip):
        raise FileNotFoundError(
            f"Neither '{gpkg_file}' nor '{gpkg_zip}' exists."
        )

    logger.info(f"Unzipping {gpkg_zip} ...")

    with zipfile.ZipFile(gpkg_zip, "r") as z:
        gpkg_members = [m for m in z.namelist() if m.lower().endswith(".gpkg")]

        if not gpkg_members:
            raise ValueError(f"No .gpkg found inside {gpkg_zip}")

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

    cell_ids = []
    geoms = []
    cid = 0

    for x in xs[:-1]:
        for y in ys[:-1]:
            cell_ids.append(cid)
            geoms.append(box(x, y, x + grid_size_m, y + grid_size_m))
            cid += 1

    return gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=geoms, crs=crs)


def clean_layer(gdf, target_crs):
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=target_crs)

    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs)
    gdf = gdf[gdf.geometry.notna()].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)

    gdf = gdf.to_crs(target_crs)
    gdf = repair_geometries(gdf)

    return gdf


def empty_gdf(crs):
    return gpd.GeoDataFrame(geometry=[], crs=crs)


def aoi_clip(gdf, aoi_geom):
    if gdf is None or gdf.empty:
        return gdf

    _ = gdf.sindex

    minx, miny, maxx, maxy = aoi_geom.bounds
    idx = list(gdf.sindex.intersection((minx, miny, maxx, maxy)))

    if not idx:
        return gdf.iloc[0:0].copy()

    subset = gdf.iloc[idx].copy()
    return subset[subset.intersects(aoi_geom)]


def read_layer_safe(gpkg_path, layer_name, columns=None):
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name, columns=columns)
    except TypeError:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)

    return clean_layer(gdf, TARGET_CRS)


def load_osm_layers_from_geofabrik_gpkg(gpkg_path):
    layers = fiona.listlayers(gpkg_path)
    layers_lower = {lyr.lower(): lyr for lyr in layers}

    logger.info(f"GPKG layers found: {layers}")

    buildings_layer = layers_lower.get("gis_osm_buildings_a_free")
    roads_layer = layers_lower.get("gis_osm_roads_free")
    waterways_layer = layers_lower.get("gis_osm_waterways_free")
    railways_layer = layers_lower.get("gis_osm_railways_free")
    landuse_layer = layers_lower.get("gis_osm_landuse_a_free")

    missing = [
        n for n, v in [
            ("gis_osm_buildings_a_free", buildings_layer),
            ("gis_osm_roads_free", roads_layer),
            ("gis_osm_waterways_free", waterways_layer),
            ("gis_osm_railways_free", railways_layer),
            ("gis_osm_landuse_a_free", landuse_layer),
        ]
        if v is None
    ]

    if missing:
        raise ValueError(f"Missing expected Geofabrik layers: {missing}")

    buildings = read_layer_safe(
        gpkg_path,
        buildings_layer,
        columns=["osm_id", "name", "code", "fclass", "geometry"]
    )

    roads = read_layer_safe(
        gpkg_path,
        roads_layer,
        columns=["osm_id", "name", "ref", "code", "fclass", "geometry"]
    )

    waterways = read_layer_safe(
        gpkg_path,
        waterways_layer,
        columns=["osm_id", "name", "code", "fclass", "geometry"]
    )

    railways = read_layer_safe(
        gpkg_path,
        railways_layer,
        columns=["osm_id", "name", "code", "fclass", "geometry"]
    )

    landuse = read_layer_safe(
        gpkg_path,
        landuse_layer,
        columns=["osm_id", "name", "code", "fclass", "geometry"]
    )

    barriers = empty_gdf(TARGET_CRS)

    traffic_layer = layers_lower.get("gis_osm_traffic_free")
    traffic_a_layer = layers_lower.get("gis_osm_traffic_a_free")
    traffic_to_try = traffic_layer or traffic_a_layer

    if traffic_to_try:
        try:
            traffic = read_layer_safe(
                gpkg_path,
                traffic_to_try,
                columns=["osm_id", "name", "code", "fclass", "geometry"]
            )

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
# FEATURE ENGINEERING
# ============================================================

def add_crossing_count(lines_gdf, layer_gdf, feature_name, tile_tag=""):
    if layer_gdf is None or layer_gdf.empty:
        lines_gdf[f"{feature_name}_crossing_count"] = 0
        lines_gdf[f"{feature_name}_crossing_flag"] = 0
        return lines_gdf

    layer = layer_gdf[["geometry"]].copy()

    t = Timer()

    hits = gpd.sjoin(
        lines_gdf,
        layer,
        how="left",
        predicate="intersects"
    )

    dt = t.split()

    counts = (
        hits.dropna(subset=["index_right"])
        .groupby([COL_METER_ID, COL_CANDIDATE])
        .size()
        .reset_index(name=f"{feature_name}_crossing_count")
    )

    out = lines_gdf.merge(
        counts,
        on=[COL_METER_ID, COL_CANDIDATE],
        how="left"
    )

    out[f"{feature_name}_crossing_count"] = (
        out[f"{feature_name}_crossing_count"]
        .fillna(0)
        .astype(int)
    )

    out[f"{feature_name}_crossing_flag"] = (
        out[f"{feature_name}_crossing_count"] > 0
    ).astype(int)

    logger.info(
        f"{tile_tag} crossing '{feature_name}': "
        f"layer={len(layer_gdf):,} "
        f"join_time={dt:.2f}s "
        f"nonzero={(out[f'{feature_name}_crossing_count'] > 0).sum():,}"
    )

    return out


def add_building_blocked_length(lines_gdf, buildings_gdf, tile_tag=""):
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
    b = repair_geometries(b)

    if b.empty:
        lines_gdf["building_count_between"] = 0
        lines_gdf["building_between_flag"] = 0
        lines_gdf["blocked_length_ft"] = 0.0
        lines_gdf["pct_line_blocked_by_building"] = 0.0
        logger.info(f"{tile_tag} buildings: none after filtering")
        return lines_gdf

    b = b.copy()
    b["BuildingID"] = b.index.astype(str)

    if INTERSECTION_CORRIDOR_M is not None and INTERSECTION_CORRIDOR_M > 0:
        corridor = lines_gdf[[COL_METER_ID, COL_CANDIDATE, "geometry"]].copy()
        corridor["geometry"] = corridor.geometry.buffer(INTERSECTION_CORRIDOR_M)
        corridor = gpd.GeoDataFrame(corridor, geometry="geometry", crs=lines_gdf.crs)

        hits = gpd.sjoin(
            corridor,
            b[["BuildingID", "geometry"]],
            how="left",
            predicate="intersects"
        )

        hits = hits.drop(columns=["geometry"], errors="ignore").merge(
            lines_gdf[[COL_METER_ID, COL_CANDIDATE, "geometry"]],
            on=[COL_METER_ID, COL_CANDIDATE],
            how="left"
        )
    else:
        hits = gpd.sjoin(
            lines_gdf,
            b[["BuildingID", "geometry"]],
            how="left",
            predicate="intersects"
        )

    dt_sjoin = t.split()

    valid_hits = hits.dropna(subset=["BuildingID"]).copy()

    logger.info(
        f"{tile_tag} buildings: "
        f"layer={len(b):,} "
        f"sjoin_time={dt_sjoin:.2f}s "
        f"hits={len(valid_hits):,}"
    )

    if len(valid_hits) > MAX_BUILDING_HITS:
        raise RuntimeError(
            f"{tile_tag} Too many building hits "
            f"({len(valid_hits):,}) > MAX_BUILDING_HITS={MAX_BUILDING_HITS:,}"
        )

    building_count = (
        valid_hits
        .groupby([COL_METER_ID, COL_CANDIDATE])
        .size()
        .reset_index(name="building_count_between")
    )

    building_geom = b[["BuildingID", "geometry"]].rename(
        columns={"geometry": "building_geom"}
    )

    valid_hits = valid_hits.merge(building_geom, on="BuildingID", how="left")
    valid_hits = gpd.GeoDataFrame(valid_hits, geometry="geometry", crs=lines_gdf.crs)

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
        valid_hits
        .groupby([COL_METER_ID, COL_CANDIDATE])["blocked_length_ft"]
        .sum()
        .reset_index()
    )

    out = lines_gdf.merge(
        building_count,
        on=[COL_METER_ID, COL_CANDIDATE],
        how="left"
    )

    out = out.merge(
        blocked_length,
        on=[COL_METER_ID, COL_CANDIDATE],
        how="left"
    )

    out["building_count_between"] = out["building_count_between"].fillna(0).astype(int)
    out["blocked_length_ft"] = out["blocked_length_ft"].fillna(0.0)
    out["building_between_flag"] = (out["building_count_between"] > 0).astype(int)

    out["pct_line_blocked_by_building"] = (
        out["blocked_length_ft"] /
        out["straight_line_length_ft"].replace({0: np.nan})
    ).fillna(0.0)

    return out


# ============================================================
# TILE PROCESSOR
# ============================================================

def process_tile(tile_df, tile_tag, osm_layers):
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

    t_lines = Timer()

    df = tile_df.copy()
    df["meter_geom"] = meter_points.values
    df["floc_geom"] = floc_points.values

    df["geometry"] = [
        LineString([mg, fg])
        for mg, fg in zip(df["meter_geom"].values, df["floc_geom"].values)
    ]

    lines = gpd.GeoDataFrame(df, geometry="geometry", crs=TARGET_CRS)
    lines["straight_line_length_ft"] = lines.geometry.length * 3.28084

    logger.info(f"{tile_tag} phase lines: {t_lines.split():.2f}s")

    t_aoi = Timer()

    minx, miny, maxx, maxy = lines.total_bounds
    aoi = box(minx, miny, maxx, maxy).buffer(TILE_AOI_BUFFER_M)

    log_tile_diagnostics(tile_tag, tile_df, lines=lines, aoi=aoi)

    logger.info(
        f"{tile_tag} phase AOI: {t_aoi.split():.2f}s | "
        f"strategy=bounds_buffer | TILE_AOI_BUFFER_M={TILE_AOI_BUFFER_M}"
    )

    t_sub = Timer()

    buildings = aoi_clip(osm_layers["buildings"], aoi)
    roads = aoi_clip(osm_layers["roads"], aoi)
    waterways = aoi_clip(osm_layers["waterways"], aoi)
    railways = aoi_clip(osm_layers["railways"], aoi)
    barriers = aoi_clip(osm_layers["barriers"], aoi)
    landuse = aoi_clip(osm_layers["landuse"], aoi)

    logger.info(
        f"{tile_tag} phase subset: {t_sub.split():.2f}s | "
        f"buildings={len(buildings):,} "
        f"roads={len(roads):,} "
        f"waterways={len(waterways):,} "
        f"railways={len(railways):,} "
        f"barriers={len(barriers):,} "
        f"landuse={len(landuse):,}"
    )

    features = lines.copy()

    features = add_building_blocked_length(features, buildings, tile_tag=tile_tag)
    features = add_crossing_count(features, roads, "street", tile_tag=tile_tag)
    features = add_crossing_count(features, waterways, "waterway", tile_tag=tile_tag)
    features = add_crossing_count(features, railways, "railway", tile_tag=tile_tag)
    features = add_crossing_count(features, barriers, "barrier", tile_tag=tile_tag)
    features = add_crossing_count(features, landuse, "landuse", tile_tag=tile_tag)

    road_class_col = None

    for c in ["fclass", "highway", "class", "type"]:
        if c in roads.columns:
            road_class_col = c
            break

    major_road_types = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    local_road_types = {"residential", "service", "unclassified", "living_street"}

    if not roads.empty and road_class_col is not None:
        def classify(val, allowed):
            if isinstance(val, list):
                return any(v in allowed for v in val)
            return val in allowed

        roads_major = roads[
            roads[road_class_col].apply(lambda x: classify(x, major_road_types))
        ].copy()

        roads_local = roads[
            roads[road_class_col].apply(lambda x: classify(x, local_road_types))
        ].copy()

        logger.info(
            f"{tile_tag} roads_major={len(roads_major):,} "
            f"roads_local={len(roads_local):,} "
            f"using '{road_class_col}'"
        )

        features = add_crossing_count(features, roads_major, "major_road", tile_tag=tile_tag)
        features = add_crossing_count(features, roads_local, "local_road", tile_tag=tile_tag)
    else:
        features["major_road_crossing_count"] = 0
        features["major_road_crossing_flag"] = 0
        features["local_road_crossing_count"] = 0
        features["local_road_crossing_flag"] = 0

        logger.info(f"{tile_tag} major/local road split skipped")

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

def process_tile_with_chunking(tile_df, tile_tag, osm_layers):
    if len(tile_df) <= MAX_ROWS_PER_TILE:
        return process_tile(tile_df, tile_tag, osm_layers)

    logger.warning(
        f"{tile_tag} large tile detected: rows={len(tile_df):,}. "
        f"Processing in chunks of {LARGE_TILE_CHUNK_SIZE:,}."
    )

    chunk_outputs = []

    for chunk_no, chunk_df in split_large_tile(tile_df):
        chunk_tag = f"{tile_tag} [chunk {chunk_no}]"

        try:
            chunk_features = process_tile(chunk_df, chunk_tag, osm_layers)
            chunk_outputs.append(chunk_features)
        except Exception as e:
            logger.error(f"{chunk_tag} chunk failed: {e}")
            logger.error(traceback.format_exc())

            if SAVE_FAILED_TILES:
                safe_tag = (
                    tile_tag.replace(" ", "_")
                    .replace("|", "")
                    .replace("[", "")
                    .replace("]", "")
                    .replace("/", "_")
                )

                failed_chunk_path = f"failed_{safe_tag}_chunk_{chunk_no}.csv"
                chunk_df.to_csv(failed_chunk_path, index=False)

                logger.info(
                    f"{chunk_tag} saved failed chunk input: {failed_chunk_path}"
                )

            continue

    if not chunk_outputs:
        raise RuntimeError(f"{tile_tag} all chunks failed.")

    return pd.concat(chunk_outputs, ignore_index=True)


def append_tile_to_output(features_tile, available_cols, output_path, tile_id):
    if USE_PARTITIONED_OUTPUT:
        os.makedirs(OUTPUT_PARTITION_DIR, exist_ok=True)

        tile_output = os.path.join(
            OUTPUT_PARTITION_DIR,
            f"tile_{int(tile_id)}.csv"
        )

        tmp_output = tile_output + ".tmp"

        features_tile[available_cols].to_csv(tmp_output, index=False)
        os.replace(tmp_output, tile_output)

        return tile_output

    header_needed = not file_exists_nonempty(output_path)

    if not SAFE_TILE_APPEND:
        features_tile[available_cols].to_csv(
            output_path,
            mode="a",
            header=header_needed,
            index=False
        )
        return output_path

    tile_tmp = f"_tile_{tile_id}.tmp.csv"
    features_tile[available_cols].to_csv(tile_tmp, index=False)

    with open(output_path, "a", encoding="utf-8", newline="") as out_f:
        with open(tile_tmp, "r", encoding="utf-8") as in_f:
            if not header_needed:
                next(in_f, None)

            out_f.write(in_f.read())

        out_f.flush()
        os.fsync(out_f.fileno())

    os.remove(tile_tmp)

    return output_path


# ============================================================
# RETRY FAILED TILES
# ============================================================

def retry_failed_tiles(osm_layers, output_cols):
    failed_registry = load_json_dict(FAILED_TILES_FILE, default={})

    if not failed_registry:
        print("No failed tiles found.")
        return

    os.makedirs(RETRY_OUTPUT_PARTITION_DIR, exist_ok=True)

    retry_results = {}

    for tile_id in failed_registry.keys():
        failed_path = f"{FAILED_TILE_INPUT_PREFIX}{tile_id}.csv"

        if not os.path.exists(failed_path):
            logger.warning(f"Failed tile input not found: {failed_path}")
            continue

        logger.info(f"Retrying failed tile {tile_id} from {failed_path}")

        tile_df = pd.read_csv(failed_path)

        original_max_rows = globals()["MAX_ROWS_PER_TILE"]
        original_chunk_size = globals()["LARGE_TILE_CHUNK_SIZE"]
        original_buffer = globals()["TILE_AOI_BUFFER_M"]
        original_output_dir = globals()["OUTPUT_PARTITION_DIR"]

        globals()["MAX_ROWS_PER_TILE"] = RETRY_MAX_ROWS_PER_TILE
        globals()["LARGE_TILE_CHUNK_SIZE"] = RETRY_LARGE_TILE_CHUNK_SIZE
        globals()["TILE_AOI_BUFFER_M"] = RETRY_TILE_AOI_BUFFER_M
        globals()["OUTPUT_PARTITION_DIR"] = RETRY_OUTPUT_PARTITION_DIR

        try:
            features_tile = process_tile_with_chunking(
                tile_df,
                f"[Retry failed tile | cell_id={tile_id}]",
                osm_layers
            )

            available_cols = [c for c in output_cols if c in features_tile.columns]

            retry_output = os.path.join(
                RETRY_OUTPUT_PARTITION_DIR,
                f"tile_{int(tile_id)}_retry.csv"
            )

            features_tile[available_cols].to_csv(retry_output, index=False)

            retry_results[str(tile_id)] = {
                "state": "retried_success",
                "rows": len(tile_df),
                "output": retry_output,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            logger.info(f"Retry success for tile {tile_id}: {retry_output}")

        except Exception as e:
            retry_results[str(tile_id)] = {
                "state": "retried_failed",
                "rows": len(tile_df),
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            logger.error(f"Retry failed again for tile {tile_id}: {e}")
            logger.error(traceback.format_exc())

        finally:
            globals()["MAX_ROWS_PER_TILE"] = original_max_rows
            globals()["LARGE_TILE_CHUNK_SIZE"] = original_chunk_size
            globals()["TILE_AOI_BUFFER_M"] = original_buffer
            globals()["OUTPUT_PARTITION_DIR"] = original_output_dir

    save_json_dict("retry_failed_tiles_status.json", retry_results)

    print("Retry completed.")
    print("Retry status: retry_failed_tiles_status.json")



# ============================================================
# MAIN
# ============================================================

def main():
    gpkg = ensure_gpkg_available(GPKG_ZIP, GPKG_FILE)

    logger.info(f"Loading candidates: {INPUT_FILE}")
    candidates = pd.read_pickle(INPUT_FILE)

    required_cols = [
        COL_METER_LON,
        COL_METER_LAT,
        COL_FLOC_LON,
        COL_FLOC_LAT,
        COL_DIVISION,
        COL_METER_ID,
        COL_CANDIDATE
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

    mid_pts = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(mid_x, mid_y)],
        crs=TARGET_CRS
    )

    assign_gdf = gpd.GeoDataFrame(df.copy(), geometry=mid_pts, crs=TARGET_CRS)

    logger.info(f"Built tile assignment points in {t_assign.split():.2f}s")

    grid = build_grid(
        assign_gdf.total_bounds,
        GRID_SIZE_M,
        TARGET_CRS,
        padding_m=GRID_PADDING_M
    )

    assigned = gpd.sjoin(assign_gdf, grid, how="left", predicate="intersects")
    assigned = assigned.drop(columns=["index_right"], errors="ignore")

    tile_ids = sorted(
        assigned["cell_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    logger.info(
        f"Tiles with data: {len(tile_ids):,} | "
        f"GRID_SIZE_M={GRID_SIZE_M}m | "
        f"TILE_AOI_BUFFER_M={TILE_AOI_BUFFER_M}m"
    )

    if DRY_RUN_TILE_IDS is not None:
        tile_ids = [
            tid for tid in tile_ids
            if int(tid) in set(map(int, DRY_RUN_TILE_IDS))
        ]

        logger.info(f"DRY_RUN_TILE_IDS active. Tiles selected: {tile_ids}")

    if DRY_RUN_TILE_RANGE is not None:
        start_tid, end_tid = DRY_RUN_TILE_RANGE

        tile_ids = [
            tid for tid in tile_ids
            if start_tid <= int(tid) <= end_tid
        ]

        logger.info(
            f"DRY_RUN_TILE_RANGE active. Tiles selected: {len(tile_ids):,}"
        )

    if MAX_TILES_TO_PROCESS is not None:
        tile_ids = tile_ids[:MAX_TILES_TO_PROCESS]

        logger.info(
            f"MAX_TILES_TO_PROCESS active. Processing only {len(tile_ids):,} tiles."
        )

    logger.info(f"Loading OSM layers from local GeoPackage: {gpkg}")

    t_osm = Timer()
    osm_layers = load_osm_layers_from_geofabrik_gpkg(gpkg)

    for k, layer in osm_layers.items():
        if layer is not None and not layer.empty:
            _ = layer.sindex

    logger.info(
        f"Loaded local layers in {t_osm.split():.2f}s | "
        f"buildings={len(osm_layers['buildings']):,} "
        f"roads={len(osm_layers['roads']):,} "
        f"waterways={len(osm_layers['waterways']):,} "
        f"railways={len(osm_layers['railways']):,} "
        f"barriers={len(osm_layers['barriers']):,} "
        f"landuse={len(osm_layers['landuse']):,}"
    )

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

    done_tiles = load_checkpoint(CHECKPOINT_FILE)

    logger.info(
        f"Checkpoint loaded: {len(done_tiles):,} tiles already completed "
        f"(from {CHECKPOINT_FILE})"
    )

    tile_status = load_json_dict(TILE_STATUS_FILE, default={})
    failed_tiles_registry = load_json_dict(FAILED_TILES_FILE, default={})

    if not APPEND_OUTPUT:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            logger.info(f"Removed existing output file: {OUTPUT_FILE}")

        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            done_tiles = set()
            logger.info(f"Removed existing checkpoint: {CHECKPOINT_FILE}")

        if os.path.exists(TILE_STATUS_FILE):
            os.remove(TILE_STATUS_FILE)
            tile_status = {}

        if os.path.exists(FAILED_TILES_FILE):
            os.remove(FAILED_TILES_FILE)
            failed_tiles_registry = {}

    failed_tiles = []
    overall = Timer()
    success_counter = 0

    for idx, tid in enumerate(
        tqdm(tile_ids, desc="Processing tiles (LOCAL GPKG)", unit="tile"),
        start=1
    ):
        tid_int = int(tid)

        if tid_int in done_tiles:
            continue

        tile_tag = f"[Tile {idx}/{len(tile_ids)} | cell_id={tid_int}]"
        t_tile = Timer()

        tile_df = pd.DataFrame(
            assigned[assigned["cell_id"] == tid_int]
            .drop(columns=["geometry"], errors="ignore")
        ).copy()

        if tile_df.empty:
            success_counter += 1
            mark_tile_done(
                tid_int,
                done_tiles,
                CHECKPOINT_FILE,
                success_counter,
                every=CHECKPOINT_EVERY
            )

            update_tile_status(
                tile_status,
                tid_int,
                "completed",
                message="empty tile",
                rows=0
            )

            continue

        logger.info(f"{tile_tag} rows={len(tile_df):,}")

        try:
            update_tile_status(
                tile_status,
                tid_int,
                "processing",
                rows=len(tile_df)
            )

            features_tile = process_tile_with_chunking(
                tile_df,
                tile_tag,
                osm_layers
            )

            available_cols = [
                c for c in output_cols
                if c in features_tile.columns
            ]

            written_path = append_tile_to_output(
                features_tile,
                available_cols,
                OUTPUT_FILE,
                tid_int
            )

            if not validate_tile_output(tid_int):
                raise RuntimeError(
                    f"{tile_tag} output validation failed after write: {written_path}"
                )

            success_counter += 1

            mark_tile_done(
                tid_int,
                done_tiles,
                CHECKPOINT_FILE,
                success_counter,
                every=CHECKPOINT_EVERY
            )

            update_tile_status(
                tile_status,
                tid_int,
                "completed",
                message=f"wrote_rows={len(features_tile):,}; output={written_path}",
                rows=len(tile_df)
            )

            logger.info(
                f"{tile_tag} DONE in {t_tile.split():.2f}s | "
                f"wrote_rows={len(features_tile):,} | output={written_path}"
            )

            if WRITE_DEBUG_TILES:
                dbg = f"tile_{tid_int}_features_debug.csv"
                features_tile[available_cols].to_csv(dbg, index=False)
                logger.info(f"{tile_tag} wrote debug tile output: {dbg}")

            continue

        except Exception as e:
            msg = f"❌ {tile_tag} FAILED: {e}"

            tqdm.write(msg)
            logger.error(msg)
            logger.error(traceback.format_exc())

            failed_tiles.append((tid_int, str(e)))

            failed_tiles_registry[str(tid_int)] = {
                "error": str(e),
                "rows": len(tile_df),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            save_json_dict(FAILED_TILES_FILE, failed_tiles_registry)

            update_tile_status(
                tile_status,
                tid_int,
                "failed",
                message=str(e),
                rows=len(tile_df)
            )

            if SAVE_FAILED_TILES:
                failed_path = f"failed_tile_{tid_int}.csv"
                tile_df.to_csv(failed_path, index=False)
                logger.info(f"{tile_tag} saved failing tile input: {failed_path}")

            continue

    if RETRY_FAILED_TILES:
        logger.info("Starting retry of failed tiles...")
        retry_failed_tiles(osm_layers, output_cols)  

    save_checkpoint(CHECKPOINT_FILE, done_tiles)
    save_json_dict(TILE_STATUS_FILE, tile_status)
    save_json_dict(FAILED_TILES_FILE, failed_tiles_registry)

    completed_count = sum(
        1 for v in tile_status.values()
        if v.get("state") == "completed"
    )

    failed_count = sum(
        1 for v in tile_status.values()
        if v.get("state") == "failed"
    )

    processing_count = sum(
        1 for v in tile_status.values()
        if v.get("state") == "processing"
    )

    logger.info("============================================")
    logger.info(f"ALL DONE in {overall.split():.2f}s")
    logger.info(
        f"Output saved to: "
        f"{OUTPUT_FILE if not USE_PARTITIONED_OUTPUT else OUTPUT_PARTITION_DIR}"
    )
    logger.info(f"Checkpoint saved to: {CHECKPOINT_FILE}")
    logger.info(f"Tile status saved to: {TILE_STATUS_FILE}")
    logger.info(f"Failed tiles registry saved to: {FAILED_TILES_FILE}")
    logger.info(f"Completed tiles: {completed_count:,}")
    logger.info(f"Failed tiles: {failed_count:,}")
    logger.info(f"Still marked processing: {processing_count:,}")

    print("\n✅ Finished.")
    print(
        f"Output: "
        f"{OUTPUT_FILE if not USE_PARTITIONED_OUTPUT else OUTPUT_PARTITION_DIR}"
    )
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Tile status: {TILE_STATUS_FILE}")
    print(f"Failed tiles: {failed_count}")

    if failed_count:
        print("Failed tile details:", FAILED_TILES_FILE)
        print("See log for details:", LOG_FILE)
    


if __name__ == "__main__":
    main()
