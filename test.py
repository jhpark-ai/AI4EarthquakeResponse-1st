import timm
import os
import cv2
import fiona
import torch
import numpy as np
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
import torch.nn as nn
import rasterio

import albumentations as A
from albumentations.pytorch import ToTensorV2


# --------------------------------------------------
# Environment optimizations
# --------------------------------------------------
cv2.setNumThreads(0)                 # Prevent excessive OpenCV threads
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


phase = 1

# All annotations should be in 'All' folder
if phase == 1:  # phase 1
    ANNOT_ROOT = "/earthquake/All"
    test_gpkg_list = {
        'adiyaman_testing.gpkg': 'DS_PHR1A_202302090832223_FR1_PX_E038N37_0419_00810-calibrated',
        'china_testing.gpkg':    'DS_PHR1A_202501080451578_FR1_PX_E087N28_0616_01014-calibrated',
        'marash_testing.gpkg':   'WV03N37_567916E036_9318052023021300000000MS00-calibrated',
        'antakya_east_testing.gpkg': 'DS_PHR1B_202302070800206_FR1_PX_E036N36_0206_01222-calibrated'
    }
else:  # phase 2
    ANNOT_ROOT = "/earthquake/phase2/charter-eo4ai-etq-challenge-testing/Testing/Annotations/All"
    DATA_ROOT = 'CROPS_50_all_phase2'
    test_gpkg_list = {
        'Antakya_West_Testing.gpkg': 'DS_PHR1B_202302070800206_FR1_PX_E036N36_0206_01222_testing-calibrated',
        'Mandalay_Testing.gpkg':    'DS_PHR1A_202503310423499_FR1_PX_E096N21_0123_02290-calibrated',
    }


def crop_around_mask(img: np.ndarray, mask: np.ndarray, margin_px: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the mask bounding box by the margin and crop first.
    If the mask is empty (all zeros), return the original inputs.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img, mask  # If empty, return as is

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    H, W = mask.shape
    y1 = max(0, y1 - margin_px)
    y2 = min(H - 1, y2 + margin_px)
    x1 = max(0, x1 - margin_px)
    x2 = min(W - 1, x2 + margin_px)

    # Slicing end index is exclusive, hence +1
    crop = img[y1:y2+1, x1:x2+1]
    mask_c = mask[y1:y2+1, x1:x2+1]
    return crop, mask_c


def letterbox_square(img: np.ndarray, mask: np.ndarray | None, size: int = 384,
                     interp_img=cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Scale the longer side to size without distortion (keep aspect ratio), then pad the shorter side to square.
    Apply the same transform to the mask (nearest-neighbor).
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    # Resize
    img_rs = cv2.resize(img, (new_w, new_h), interpolation=interp_img)
    if mask is not None:
        mask_rs = cv2.resize(mask.astype(np.uint8),
                             (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_rs = None

    # Padding (centered)
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    img_pad = cv2.copyMakeBorder(img_rs, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
    if mask_rs is not None:
        mask_pad = cv2.copyMakeBorder(mask_rs, top, bottom, left, right,
                                      borderType=cv2.BORDER_CONSTANT, value=0)
    else:
        mask_pad = None

    return img_pad, mask_pad


def prepare_test_image(img: np.ndarray, mask: np.ndarray | None,
                       out_size: int = 384, margin_px: int = 64,
                       zero_outside: bool = True) -> np.ndarray:
    """
    1) If provided, crop around the mask with the given margin
    2) Letterbox to square 384×384
    3) Optionally zero out the pixels outside the mask
    """
    if mask is not None:
        # 1) Reduce background first (crop around the mask)
        img, mask = crop_around_mask(img, mask, margin_px=margin_px)

    # 2) Letterbox resize/pad to exactly 384×384
    img, mask = letterbox_square(img, mask, size=out_size)

    # 3) Zero out outside
    if mask is not None and zero_outside:
        img = img.copy()
        img[mask == 0] = 0

    return img


def dilate_mask(m: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate a binary mask m(H, W) by radius_px using a circular structuring element."""
    if radius_px <= 0:
        return m
    k = 2 * radius_px + 1  # Kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    md = cv2.dilate((m > 0).astype(np.uint8), kernel, iterations=1)
    return (md > 0).astype(np.uint8)


def to_tensor_normalize(imagenet_norm: bool = True):
    return A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])


def read_tif(path: str) -> np.ndarray:
    # returns HWC uint8 (0..255), fill invalid (masked) pixels with 0
    with rasterio.open(path) as src:
        arr = src.read(masked=True)  # (C,H,W) MaskedArray
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        # Fill invalid (masked) pixels with 0
        arr = np.ma.filled(arr, fill_value=0).astype(np.uint8)
    hwc = np.transpose(arr, (1, 2, 0))
    return hwc


def read_mask(path: str) -> np.ndarray:
    # returns HW uint8 in {0,1}
    with rasterio.open(path) as src:
        m = src.read(1)
    return (m > 0).astype(np.uint8)


def read_first_layer_gpkg(path: str) -> gpd.GeoDataFrame:
    layer_names = fiona.listlayers(path)
    if not layer_names:
        raise RuntimeError(f"No layers in GPKG: {path}")
    layer = layer_names[0]
    return gpd.read_file(path, layer=layer, driver="GPKG")


def first_match_column(cols: List[str], keywords=('fid', 'id')):
    cols_lower = [c.lower() for c in cols]
    for kw in keywords:
        for i, c in enumerate(cols_lower):
            if kw in c:
                return cols[i]
    return None


def build_paths_for_gpkg(gpkg_name: str) -> Tuple[str, str]:
    img_folder_path = os.path.join(
        DATA_ROOT, 'test', test_gpkg_list[gpkg_name], 'images')
    mask_folder_path = os.path.join(
        DATA_ROOT, 'test', test_gpkg_list[gpkg_name], 'masks')
    return img_folder_path, mask_folder_path

# --------------------------------------------------
# Cache path utilities
# --------------------------------------------------


def tensor_cache_path(output_dir: str, gpkg_name: str, idx: int) -> str:
    cache_dir = os.path.join(output_dir, "_cache",
                             gpkg_name.replace(".gpkg", ""))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{idx:08d}.pt")

# --------------------------------------------------
# Dataset: preprocessing + (optional) cache save/load
# --------------------------------------------------


class GpkgTiles(Dataset):
    def __init__(self,
                 gpkg_name: str,
                 img_size: int = 384,
                 margin_px: int = 10,
                 imagenet_norm: bool = True,
                 make_cache: bool = False,
                 use_cache: bool = True,
                 output_dir: str = "submission_results",
                 mask_size: int = 30
                 ):
        self.gpkg_name = gpkg_name
        self.gdf = read_first_layer_gpkg(os.path.join(ANNOT_ROOT, gpkg_name))
        self.fid_col = first_match_column(
            list(self.gdf.columns), keywords=("fid", "id"))
        self.img_folder, self.mask_folder = build_paths_for_gpkg(gpkg_name)
        self.img_size = img_size
        self.margin_px = margin_px
        self.imagenet_norm = imagenet_norm
        self.make_cache = make_cache
        self.use_cache = use_cache
        self.mask_size = mask_size
        self.output_dir = output_dir
        self.transform = to_tensor_normalize(imagenet_norm=imagenet_norm)

        # Index sampling
        all_indices = list(range(len(self.gdf)))
        self.indices = all_indices

        # Precompute path list
        self.meta = []
        for idx in self.indices:
            row = self.gdf.iloc[idx]
            fid_part = f"_{self.fid_col}_{row[self.fid_col]}" if self.fid_col is not None else ""
            base = f"{test_gpkg_list[gpkg_name]}_idx{idx}{fid_part}_d0"
            img_path = os.path.join(self.img_folder, f"{base}.tif")
            msk_path = os.path.join(self.mask_folder, f"{base}_mask.tif")
            self.meta.append((idx, img_path, msk_path))

    def __len__(self):
        return len(self.meta)

    def _preprocess_one(self, img_path: str, msk_path: str) -> Optional[torch.Tensor]:
        if not os.path.exists(img_path):
            return None
        img = read_tif(img_path)
        mask = read_mask(msk_path)
        mask = dilate_mask(mask, self.mask_size)
        img_processed = prepare_test_image(img, mask, out_size=self.img_size,
                                           margin_px=self.margin_px, zero_outside=False)
        t = self.transform(image=img_processed)["image"]  # [C,H,W]
        return t

    def __getitem__(self, i: int):
        idx, img_path, msk_path = self.meta[i]
        cache_p = tensor_cache_path(self.output_dir, self.gpkg_name, idx)
        if self.use_cache and os.path.exists(cache_p):
            t = torch.load(cache_p, map_location="cpu")
            return idx, t
        t = self._preprocess_one(img_path, msk_path)
        if t is None:
            return idx, None
        if self.make_cache:
            torch.save(t, cache_p)
        return idx, t


def collate_fn(batch):
    # Filter out None (missing), separate indices and tensors
    idxs, tensors = [], []
    for idx, t in batch:
        if t is not None:
            idxs.append(idx)
            tensors.append(t)
    if len(tensors) == 0:
        return [], None
    x = torch.stack(tensors, dim=0)  # [B,C,H,W]
    return idxs, x


class CustomModel(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k	',
            pretrained=pretrained,
            num_classes=num_classes  # Reinitialize starting from the head layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model(model_path: str,
               model_name: str = 'eva',
               device: Optional[torch.device] = None,
               use_half: bool = False) -> torch.nn.Module:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(model_path, map_location='cpu')

    model = CustomModel(in_channels=3, num_classes=1,
                        pretrained=False)
    state = ckpt.get("model", ckpt)
    new_state = {}
    for k, v in state.items():
        if k.startswith("model.patch_embed.0."):
            # "patch_embed.0." → "patch_embed."
            new_k = k.replace("patch_embed.0.", "patch_embed.")
            new_state[new_k] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def infer_one_fold(model_path: str,
                   gpkg_name: str,
                   device: torch.device,
                   dl: DataLoader,
                   N_total: int,
                   model_name: str = 'eva',
                   amp: bool = True) -> np.ndarray:
    model = load_model(model_path, model_name=model_name, device=device)
    # Should be based on the length of gdf
    probs = np.zeros(N_total, dtype=np.float32)
    with torch.autocast('cuda', dtype=torch.float32, enabled=(amp and device.type == 'cuda')):
        for idxs, x in dl:
            if x is None or len(idxs) == 0:
                print("x is None!")
                continue
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).float().squeeze(1).cpu().numpy()  # [B]
            for i, gi in enumerate(idxs):
                probs[gi] = p[i]
    return probs

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------


def run_all(output_dir: str = "submission_results",
            model_name: str = 'eva',
            fold_paths: List[str] = [],
            threshold: float = 0.5,
            img_size: int = 448,
            batch_size: int = 32,
            num_workers: int = 8,
            make_cache: bool = True,
            use_cache: bool = True,
            ):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    imagenet_norm = True
    for gpkg_name in test_gpkg_list:
        print(f"\n=== {gpkg_name} ===")
        # Dataset/loader: preprocessing is performed only once (+cache)
        fold_probs = []
        mask_size = 30
        ds = GpkgTiles(gpkg_name,
                       img_size=img_size,
                       make_cache=make_cache,
                       use_cache=use_cache,
                       output_dir=output_dir,
                       imagenet_norm=imagenet_norm,
                       mask_size=mask_size
                       )
        N_total = len(read_first_layer_gpkg(
            # Total length (independent of sampling)
            os.path.join(ANNOT_ROOT, gpkg_name)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0),
                        collate_fn=collate_fn)

        for fp in fold_paths:
            print(f"  - inferring: {os.path.basename(fp)}")
            probs = infer_one_fold(fp, gpkg_name, device,
                                   dl, N_total, model_name=model_name, amp=True)
            fold_probs.append(probs)

        ensemble = np.mean(fold_probs, axis=0)   # [N_total]
        prediction = (ensemble >= threshold).astype(np.uint8)

        # Save (add column to match original gdf length)
        gdf_full = read_first_layer_gpkg(os.path.join(ANNOT_ROOT, gpkg_name))
        if len(prediction) != len(gdf_full):
            print(f"  -> sampling: {len(prediction)} != {len(gdf_full)}")
            # If sampling is used: fill only sampled indices
            pred_full = np.zeros(len(gdf_full), dtype=np.uint8)
            pred_full[ds.indices] = prediction[ds.indices]
            gdf_full['damaged'] = pred_full
        else:
            print(f"  -> sampling: {len(prediction)} == {len(gdf_full)}")
            gdf_full['damaged'] = prediction

        out_path = os.path.join(output_dir, gpkg_name)
        gdf_full.to_file(out_path, driver="GPKG")
        print(f"  -> saved: {out_path}")


if __name__ == "__main__":
    run_all(
        output_dir="submission_results",
        model_name='eva',
        fold_paths=[
            f"saved_model/best_swa_model_fold{i}.pt" for i in range(5)],
        threshold=0.5,
        img_size=448,
        # Increase to 48/64 if GPU allows (reduce if OOM)
        batch_size=512,
        num_workers=8,           # Adjust to the number of CPU cores
        make_cache=False,         # Create cache on first run
        use_cache=False,          # Reuse cache on subsequent runs
    )
