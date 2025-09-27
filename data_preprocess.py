import os
import csv
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import rasterio
from rasterio.features import rasterize
import fiona
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import unary_union


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

PERCENTILE_CLIP = 2.0   # Per-crop per-band contrast stretch
PAD_PIXELS = 50   # Context padding around each building
ALL_TOUCHED = True  # Use all_touched when rasterizing/masking

PHASE = 2
if PHASE == 1:
    DATA_ROOT = "Training"
    SELECT_FILE = "config/selected_id.txt"
    OUT_ROOT = f"CROPS_{PAD_PIXELS}_all"   # Final output root
elif PHASE == 2:
    DATA_ROOT = "Testing"
    SELECT_FILE = "config/selected_id_phase2.txt"
    OUT_ROOT = f"CROPS_{PAD_PIXELS}_all_phase2"   # Final output root

# Path containing per-band RGB GeoTIFFs
BAND_FOLDER = "Optical_Calibration"
R_NAME, G_NAME, B_NAME = "r-red.tif", "r-green.tif", "r-blue.tif"
LOG_LEVEL = logging.INFO


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("eq-preproc")


def find_dataset_path(dataset_id: str, data_root: str) -> Optional[str]:
    for root, dirs, _ in os.walk(data_root):
        for d in dirs:
            if dataset_id in d:
                return os.path.join(root, d)
    logger.warning(f"Dataset ID not found: {dataset_id}")
    return None


def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_first_layer_gpkg(path: str) -> gpd.GeoDataFrame:
    layer_names = fiona.listlayers(path)
    if not layer_names:
        raise RuntimeError(f"No layers in GPKG: {path}")
    layer = layer_names[0]
    gdf = gpd.read_file(path, layer=layer, driver="GPKG")
    return gdf


def get_extent_geometry(extent_path: Optional[str], target_crs) -> Optional[object]:
    if extent_path is None:
        return None
    gdf = read_first_layer_gpkg(extent_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    # dissolve to a single (multi)polygon
    geom = unary_union(gdf.geometry.values)
    return geom


def first_match_column(cols: List[str], keywords=('fid', 'id')):
    cols_lower = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(cols_lower):
            if kw in c:
                return cols[i]
    return None


def robust_percentiles(arr: np.ndarray, p: float) -> Tuple[float, float]:
    flat = arr.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size < 10:
        return float(np.nanmin(arr)), float(np.nanmax(arr))
    vmin = np.nanpercentile(flat, p)
    vmax = np.nanpercentile(flat, 100 - p)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        vmin = float(np.nanmin(flat))
        vmax = float(np.nanmax(flat))
    return float(vmin), float(vmax)


def clip_and_scale_uint8(arr: np.ndarray, pclip: float, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    vmin, vmax = robust_percentiles(arr, pclip)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        out = np.zeros(arr.shape, np.uint8)
        if valid_mask is not None:
            out[valid_mask] = 255
        return out
    scaled = ((np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin) * 255.0)
    scaled = np.nan_to_num(scaled, nan=0.0).astype(np.uint8)
    if valid_mask is not None:
        scaled = np.where(valid_mask, scaled, 0)
    return scaled


@dataclass
class SelectedItem:
    dataset_id: str
    split: str            # "train" or "test"
    gpkg_path: str
    extent_path: Optional[str]


def parse_selected_file(path: str) -> List[SelectedItem]:
    """
    One-line example per entry:
    DS_XXX-calibrated O antakya_east_training.gpkg Antakya_TRAINING-Extent.gpkg
    DS_YYY-calibrated O adiyaman_training.gpkg None
    """
    items: List[SelectedItem] = []
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                logger.warning(f"Skip malformed line: {line}")
                continue
            dataset_id, is_ok, gpkg_name, extent_name = parts
            if is_ok.upper() != "O":
                logger.info(f"Skip not-OK dataset: {dataset_id}")
                continue
            gpkg_path = os.path.join(
                # NOTE: Assume all GPKGs are inside the "All" folder.
                DATA_ROOT, "Annotations", "All", gpkg_name)
            extent_path = None if extent_name == "None" else os.path.join(
                DATA_ROOT, "Annotations", "All", extent_name)
            split = "train" if "training" in gpkg_name.lower() else "test"
            items.append(SelectedItem(
                dataset_id, split, gpkg_path, extent_path))
    return items


# ------------------------------------------------------------
# Core processing
# ------------------------------------------------------------
def open_rgb_band_readers(ds_path: str):
    rgb_dir = os.path.join(ds_path, BAND_FOLDER)
    pre_rgb_path = os.path.join(rgb_dir, "rgb_preprocessed.tif")

    if os.path.exists(pre_rgb_path):
        # If a single combined RGB GeoTIFF exists
        ds = rasterio.open(pre_rgb_path)
        if ds.count < 3:
            raise RuntimeError(f"{pre_rgb_path} does not have 3 bands.")
        # Return the same dataset for R/G/B so callers can use ds.read(1/2/3)
        # Later in processing we will call ds.read(1), ds.read(2), ds.read(3)
        return ds, ds, ds
    else:
        # Open individual band files
        r_path = os.path.join(rgb_dir, R_NAME)
        g_path = os.path.join(rgb_dir, G_NAME)
        b_path = os.path.join(rgb_dir, B_NAME)
        for p in (r_path, g_path, b_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing band: {p}")
        r = rasterio.open(r_path)
        g = rasterio.open(g_path)
        b = rasterio.open(b_path)

        # sanity checks
        if not (r.crs == g.crs == b.crs and
                r.transform == g.transform == b.transform and
                r.shape == g.shape == b.shape):
            raise RuntimeError(
                "RGB bands have mismatched CRS/transform/shape. Consider resampling to a common grid."
            )
        return r, g, b


def make_building_mask(geom, out_shape: Tuple[int, int], out_transform) -> np.ndarray:
    mask = rasterize(
        [(geom, 1)],
        out_shape=out_shape,
        transform=out_transform,
        fill=0,
        all_touched=ALL_TOUCHED,
        dtype="uint8"
    )
    return mask


def save_geotiff(path: str, array: np.ndarray, transform, crs, nodata=None):
    """
    array: (C,H,W) for multi-band or (H,W) for single band
    """
    if array.ndim == 2:
        count = 1
        height, width = array.shape
        dtype = array.dtype
    elif array.ndim == 3:
        count, height, width = array.shape
        dtype = array.dtype
    else:
        raise ValueError("Array must be 2D or 3D")

    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata
    }
    with rasterio.open(path, "w", **meta) as dst:
        if count == 1:
            dst.write(array, 1)
        else:
            dst.write(array)


def save_rgb_with_mask(path, rgb_u8, transform, crs, valid_mask):
    meta = {
        "driver": "GTiff", "height": rgb_u8.shape[1], "width": rgb_u8.shape[2],
        "count": 3, "dtype": "uint8", "crs": crs, "transform": transform,
        "tiled": True, "compress": "lzw", "interleave": "pixel"
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(rgb_u8)  # (3,H,W)
        # 255=valid, 0=nodata
        alpha = (valid_mask.astype(np.uint8) * 255)
        dst.write_mask(alpha)


def coerce_damaged(val) -> Optional[int]:
    """
    Normalize a "damaged" label into 0/1/None.
    """
    if val is None:
        return None
    if isinstance(val, (int, np.integer)):
        if val in (0, 1):
            return int(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("1", "true", "damaged", "yes", "y"):
            return 1
        if v in ("0", "false", "undamaged", "no", "n"):
            return 0
    # float or others
    try:
        f = float(val)
        return 1 if f > 0.5 else 0
    except Exception:
        return None


def build_sample_name(dataset_id: str, idx: int, fid_name: Optional[str], fid_val, damaged: Optional[int]) -> str:
    fid_part = ""
    if fid_name is not None and fid_val is not None:
        fid_part = f"_{fid_name}_{fid_val}"
    dmg_part = "dn" if damaged is None else f"d{damaged}"
    return f"{dataset_id}_idx{idx}{fid_part}_{dmg_part}"


def open_rgb_readers(ds_path: str):
    """Use combined RGB if available; otherwise open separate r/g/b bands."""
    rgb_dir = os.path.join(ds_path, BAND_FOLDER)

    r_path = os.path.join(rgb_dir, R_NAME)
    g_path = os.path.join(rgb_dir, G_NAME)
    b_path = os.path.join(rgb_dir, B_NAME)
    for p in (r_path, g_path, b_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing band: {p}")
    r = rasterio.open(r_path)
    g = rasterio.open(g_path)
    b = rasterio.open(b_path)

    if not (r.crs == g.crs == b.crs and r.transform == g.transform == b.transform and r.shape == g.shape == b.shape):
        raise RuntimeError(
            "RGB bands have mismatched CRS/transform/shape. Consider resampling to a common grid.")
    return {"mode": "separate", "r": r, "g": g, "b": b}


def process_one_dataset(item: SelectedItem, out_root: str):
    ds_path = find_dataset_path(item.dataset_id, DATA_ROOT)
    if ds_path is None:
        return

    split_root = os.path.join(out_root, item.split)
    ds_out_root = os.path.join(split_root, item.dataset_id)
    img_dir = os.path.join(ds_out_root, "images")
    msk_dir = os.path.join(ds_out_root, "masks")
    meta_csv = os.path.join(ds_out_root, f"{item.dataset_id}_samples.csv")
    safe_mkdir(img_dir)
    safe_mkdir(msk_dir)

    # Open RGB sources (prefer combined if available)
    try:
        rgb_src = open_rgb_readers(ds_path)
        base_src = rgb_src["src"] if rgb_src["mode"] == "combined" else rgb_src["r"]
    except Exception as e:
        logger.error(f"{item.dataset_id}: open RGB failed: {e}")
        return

    # Labels
    try:
        gdf = read_first_layer_gpkg(item.gpkg_path)
    except Exception as e:
        logger.error(f"{item.dataset_id}: read gpkg failed: {e}")
        # Close opened resources
        if rgb_src["mode"] == "combined":
            rgb_src["src"].close()
        else:
            for src in (rgb_src["r"], rgb_src["g"], rgb_src["b"]):
                src.close()
        return

    # Align CRS
    if gdf.crs != base_src.crs:
        gdf = gdf.to_crs(base_src.crs)

    # Extent mask
    extent_geom = get_extent_geometry(item.extent_path, base_src.crs)

    fid_col = first_match_column(list(gdf.columns), keywords=("fid", "id"))
    dmg_col = next((c for c in gdf.columns if c.lower() == "damaged"), None)

    rows_meta = []
    total = len(gdf)
    logger.info(
        f"{item.dataset_id}: {total} features to process (split={item.split})")

    from rasterio.windows import from_bounds
    from rasterio.windows import transform as win2transform

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            logger.warning(f"{item.dataset_id}[{idx}] geometry empty")
            continue
        if extent_geom is not None:
            geom = geom.intersection(extent_geom)
            if geom.is_empty:
                logger.warning(
                    f"{item.dataset_id}[{idx}] extent intersection empty")
                continue

        # 1) Bounding box
        minx, miny, maxx, maxy = geom.bounds

        # 2) Convert PAD_PIXELS into map units
        px = abs(base_src.transform.a)
        py = abs(base_src.transform.e)
        pad_x = PAD_PIXELS * px
        pad_y = PAD_PIXELS * py

        # 3) Padded bounding box
        minx_p, miny_p = minx - pad_x, miny - pad_y
        maxx_p, maxy_p = maxx + pad_x, maxy + pad_y

        # 4) Raster window
        win = from_bounds(minx_p, miny_p, maxx_p, maxy_p, base_src.transform)
        out_transform = win2transform(win, base_src.transform)
        fill = 0 if base_src.nodata is None else base_src.nodata

        # 5) Read RGB (branch)
        if rgb_src["mode"] == "combined":
            # Read bands 1,2,3 (R,G,B) from combined TIFF (no contrast stretch)
            rgb_arr = base_src.read(
                [1, 2, 3], window=win, boundless=True, fill_value=fill)
            # Valid-data mask
            valid = np.any(rgb_arr != fill, axis=0)

            # Ensure dtype uint8 (scale float 0..1 to 0..255)
            if rgb_arr.dtype != np.uint8:
                amax = float(np.nanmax(rgb_arr)) if np.isfinite(
                    rgb_arr).any() else 0.0
                if np.issubdtype(rgb_arr.dtype, np.floating) and amax <= 1.0:
                    rgb_arr = (rgb_arr * 255.0)
                rgb_arr = np.clip(rgb_arr, 0, 255).astype(np.uint8)
            rgb_u8 = rgb_arr  # (3,H,W)
        else:
            # Read separate r/g/b bands and apply contrast stretch
            r_arr = rgb_src["r"].read(
                1, window=win, boundless=True, fill_value=fill).astype(np.float32)
            g_arr = rgb_src["g"].read(
                1, window=win, boundless=True, fill_value=fill).astype(np.float32)
            b_arr = rgb_src["b"].read(
                1, window=win, boundless=True, fill_value=fill).astype(np.float32)
            valid = (r_arr != fill) | (g_arr != fill) | (b_arr != fill)
            r_u8 = clip_and_scale_uint8(
                r_arr, PERCENTILE_CLIP, valid_mask=valid)
            g_u8 = clip_and_scale_uint8(
                g_arr, PERCENTILE_CLIP, valid_mask=valid)
            b_u8 = clip_and_scale_uint8(
                b_arr, PERCENTILE_CLIP, valid_mask=valid)
            rgb_u8 = np.stack([r_u8, g_u8, b_u8], axis=0)

        H, W = rgb_u8.shape[1], rgb_u8.shape[2]

        # Building mask
        bld_mask = make_building_mask(geom, (H, W), out_transform)

        # Label fields
        fid_val = row[fid_col] if (
            fid_col and fid_col in gdf.columns) else None
        damaged = coerce_damaged(row[dmg_col]) if dmg_col is not None else None

        base = build_sample_name(item.dataset_id, idx,
                                 fid_col, fid_val, damaged)
        img_path = os.path.join(img_dir, base + ".tif")
        msk_path = os.path.join(msk_dir, base + "_mask.tif")

        try:
            save_rgb_with_mask(
                img_path, rgb_u8, out_transform, base_src.crs, valid)
            save_geotiff(msk_path, bld_mask.astype(np.uint8),
                         out_transform, base_src.crs, nodata=0)
        except Exception as e:
            logger.error(f"{item.dataset_id}[{idx}] save failed: {e}")
            continue

        rows_meta.append([
            item.dataset_id, idx, str(fid_col) if fid_col else "",
            str(fid_val) if fid_val is not None else "",
            damaged if damaged is not None else "",
            img_path, msk_path, H, W
        ])

        if (idx + 1) % 200 == 0:
            logger.info(f"{item.dataset_id}: processed {idx+1}/{total}")

    if rows_meta:
        with open(meta_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset_id", "row_idx", "fid_col", "fid_val", "damaged",
                        "image_path", "mask_path", "height", "width"])
            w.writerows(rows_meta)
        logger.info(f"{item.dataset_id}: saved meta -> {meta_csv}")

    # Cleanup resources
    if rgb_src["mode"] == "combined":
        rgb_src["src"].close()
    else:
        for src in (rgb_src["r"], rgb_src["g"], rgb_src["b"]):
            src.close()

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------


def main():
    safe_mkdir(OUT_ROOT)
    safe_mkdir(os.path.join(OUT_ROOT, "train"))
    safe_mkdir(os.path.join(OUT_ROOT, "test"))

    items = parse_selected_file(SELECT_FILE)
    logger.info(f"Selected datasets: {len(items)}")

    for it in items:
        logger.info(f"==> Start: {it.dataset_id} (split={it.split})")
        process_one_dataset(it, OUT_ROOT)

    logger.info("All done.")


if __name__ == "__main__":
    main()
