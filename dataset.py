import inspect
import os
import re
import glob
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.mixing.domain_adaptation import HistogramMatching

import cv2


class FDAFromRefs(A.ImageOnlyTransform):
    """
    Frequency Domain Adaptation (FDA) transform for Albumentations.
    - At each step, sample a random reference image from the list passed via metadata_key.
    - L is sampled uniformly from (low, high).
    """

    def __init__(self, metadata_key="fda_metadata", L=(0.05, 0.20), always_apply=False, p=0.3):
        super().__init__(always_apply, p)
        self.metadata_key = metadata_key
        self.L = L if isinstance(L, (list, tuple)) else (float(L), float(L))

    def apply(self, img, **params):
        refs = params.get(self.metadata_key, None)
        if not refs:
            return img  # No-op if no references provided
        # Assume refs is a list of np.ndarray
        ref = refs[np.random.randint(len(refs))]
        L = np.random.uniform(self.L[0], self.L[1])

        # Multispectral support: apply FDA to first 3 channels; keep remaining channels
        if img.ndim == 3 and img.shape[2] > 3:
            head = img[..., :3]
            tail = img[..., 3:]
            head_aug = FDA_source_to_target_np(head, ref, L=L)
            return np.concatenate([head_aug, tail], axis=2)
        else:
            return FDA_source_to_target_np(img, ref, L=L)

    def get_transform_init_args_names(self):
        return ("metadata_key", "L")


def dilate_mask(m: np.ndarray, radius_px: int) -> np.ndarray:
    """Dilate a binary mask m(H, W) by radius_px using a circular structuring element."""
    if radius_px <= 0:
        return m
    k = 2 * radius_px + 1  # Kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    md = cv2.dilate((m > 0).astype(np.uint8), kernel, iterations=1)
    return (md > 0).astype(np.uint8)


def parse_label_from_filename(path: str) -> Optional[int]:
    # expects ..._d0.tif / ..._d1.tif / ..._dn.tif
    m = re.search(r"_d([01n])\.tif$", path)
    if not m:
        return None
    g = m.group(1)
    if g == "n":  # unknown
        return None
    return int(g)


def read_tif(path: str) -> np.ndarray:
    # Returns HWC uint8 (0..255); fill invalid (masked) pixels with 0
    with rasterio.open(path) as src:
        arr = src.read(masked=True)  # (C,H,W) MaskedArray
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        # Fill masked-out pixels with 0
        arr = np.ma.filled(arr, fill_value=0).astype(np.uint8)
    hwc = np.transpose(arr, (1, 2, 0))
    return hwc


def read_mask(path: str) -> np.ndarray:
    # returns HW uint8 in {0,1}
    with rasterio.open(path) as src:
        m = src.read(1)
    return (m > 0).astype(np.uint8)


# True if Albumentations >= 2.x
_IS_A2 = int(A.__version__.split(".", 1)[0]) >= 2


def PadIfNeededCompat(min_height, min_width, border_mode=cv2.BORDER_CONSTANT, fill_img=0, fill_msk=0):
    if _IS_A2:
        return A.PadIfNeeded(min_height=min_height, min_width=min_width,
                             border_mode=border_mode, fill=fill_img, fill_mask=fill_msk)
    else:
        return A.PadIfNeeded(min_height=min_height, min_width=min_width,
                             border_mode=border_mode, value=fill_img, mask_value=fill_msk)


def AffineCompat(**kw):
    if _IS_A2:
        # Albumentations 2.x: fill, fill_mask
        return A.Affine(**kw)
    else:
        # Albumentations 1.x: use cval, cval_mask
        kw2 = dict(kw)
        if 'fill' in kw2:
            kw2['cval'] = kw2.pop('fill')
        if 'fill_mask' in kw2:
            kw2['cval_mask'] = kw2.pop('fill_mask')
        return A.Affine(**kw2)


def ImageCompressionCompat(quality_low, quality_high, p=0.3):
    return A.ImageCompression(
        quality_range=(quality_low, quality_high),
        compression_type='jpeg',  # or 'webp'
        p=p
    )


def RandomShadowCompat(num_low, num_high, p=0.2, shadow_dimension=5, **kw):
    return A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1),  # Keep original default
        num_shadows_limit=(num_low, num_high),
        shadow_dimension=shadow_dimension,
        p=p,
        **kw
    )


def DownscaleCompat(scale_min, scale_max, interpolation=cv2.INTER_LINEAR, p=0.35):
    return A.Downscale(
        scale_range=(scale_min, scale_max),
        interpolation_pair={'downscale': cv2.INTER_AREA,
                            'upscale': cv2.INTER_LINEAR},
        p=p
    )


def build_geo_dg_aug(img_size: int = 448, keep_aspect: bool = True, train: bool = True):
    T = []
    # 1) Fix size without distortion – pad smaller, shrink larger
    if keep_aspect:
        T += [
            A.LongestMaxSize(max_size=img_size,
                             interpolation=cv2.INTER_LINEAR),
            PadIfNeededCompat(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT, fill_img=0, fill_msk=0),
        ]
    else:
        T += [A.Resize(height=img_size, width=img_size,
                       interpolation=cv2.INTER_LINEAR)]

    if train:
        # 2) Resolution/GSD & MTF
        T += [
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Sharpen(alpha=(0.05, 0.2), lightness=(0.8, 1.2), p=1.0),
            ], p=0.5),
        ]
        # Direction invariance: random flips and 90-degree rotations
        # Random H/V flip
        T += [A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=0.5)]
        # RandomRotate90 (0/90/180/270)
        T += [A.RandomRotate90(p=0.5)]

        # 3) Light geometric aug
        T += [
            AffineCompat(
                translate_percent=(-0.06, 0.06),
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0, fill_mask=0,
                p=0.6,
            ),
        ]
        # 4) Domain matching (Histogram Matching) → color/tone
        #    Define transform regardless of availability of style_ref_images;
        #    it becomes a no-op if metadata is not provided at call time.
        T += [
            A.OneOf([
                # Histogram Matching: align source histograms to reference
                HistogramMatching(metadata_key="hm_metadata",
                                  blend_ratio=(0.3, 0.7), p=1.0),
            ], p=0.8),  # 80% chance overall to pick one
        ]

        # 5) Sensor/photometric
        T += [
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=(1, 3), tile_grid_size=(8, 8), p=1.0),
            ], p=0.7),
            A.GaussNoise(std_range=(2/255.0, 6/255.0),
                         mean_range=(0.0, 0.0), p=0.3),
        ]

        # 6) Occlusions (cloud/shadow/missing)
        T += [
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(int(0.05*img_size), int(0.15*img_size)),
                hole_width_range=(int(0.05*img_size), int(0.20*img_size)),
                fill=255, p=0.25
            ),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(0.05*img_size), int(0.12*img_size)),
                hole_width_range=(int(0.05*img_size), int(0.15*img_size)),
                fill=0, p=0.20
            ),
        ]

    # Final resize as safety net
    T += [A.Resize(height=img_size, width=img_size,
                   interpolation=cv2.INTER_LINEAR)]
    return A.Compose(T, strict=False)


def to_tensor_normalize(imagenet_norm: bool = True):
    if imagenet_norm:
        return A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
    else:
        return A.Compose([A.Normalize(mean=(0.430, 0.411, 0.296),
                                      std=(0.213, 0.156, 0.143)), ToTensorV2()])


def crop_around_mask(img: np.ndarray, mask: np.ndarray, margin_px: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the mask bounding box by margin and crop first.
    If the mask is empty (all zeros), return inputs as-is.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img, mask  # Return as-is if empty

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    H, W = mask.shape
    y1 = max(0, y1 - margin_px)
    y2 = min(H - 1, y2 + margin_px)
    x1 = max(0, x1 - margin_px)
    x2 = min(W - 1, x2 + margin_px)

    # Slicing end index is exclusive → add +1
    crop = img[y1:y2+1, x1:x2+1]
    mask_c = mask[y1:y2+1, x1:x2+1]
    return crop, mask_c


def letterbox_square(img: np.ndarray, mask: np.ndarray | None, size: int = 448,
                     interp_img=cv2.INTER_LINEAR) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Scale the longer side to size without distortion, then pad to square.
    Apply the same transform to the mask with nearest-neighbor.
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

    # Centered padding
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
                       out_size: int = 448, margin_px: int = 64,
                       zero_outside: bool = True) -> np.ndarray:
    """
    1) If provided, crop around mask with given margin
    2) Letterbox to 448×448
    3) Optionally zero out pixels outside the mask
    """
    if mask is not None:
        # Reduce background (crop around mask first)
        img, mask = crop_around_mask(img, mask, margin_px=margin_px)

    # Letterbox resize/pad to exactly 448×448
    img, mask = letterbox_square(img, mask, size=out_size)

    # Optionally zero out outside
    if mask is not None and zero_outside:
        img = img.copy()
        img[mask == 0] = 0

    return img


def resize_or_crop(img: np.ndarray, img_size: int) -> np.ndarray:
    h, w = img.shape[:2]

    # Case 1: too large → center crop
    if h > img_size and w > img_size:
        start_y = (h - img_size) // 2
        start_x = (w - img_size) // 2
        img = img[start_y:start_y+img_size, start_x:start_x+img_size]

    # Case 2: one side smaller → resize
    else:
        img = cv2.resize(img, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)

    return img

# ===========================
# Dataset
# ===========================


@dataclass
class SampleRow:
    img_path: str
    mask_path: Optional[str]
    label: Optional[int]


class CropsSplit(Dataset):
    """
    root_split_dir: /.../CROPS/train or /.../CROPS/test
    label_source: "filename" or "csv" (csv: matched via each dataset_id's *_samples.csv)
    ignore_unknown: if True, exclude 'dn' samples (unknown labels)
    """

    def __init__(self, root_split_dir: str, img_size: int = 448,
                 label_source: str = "filename", ignore_unknown: bool = True,
                 use_masks: bool = True,
                 train: bool = True,
                 imagenet_norm: bool = True,
                 context_expand_prob: float = 1,          # Probability to include context
                 context_expand_px_range: Tuple[int, int] = (
                     20, 50),  # Range of dilation pixels for context
                 data_aug_type: str = "test",
                 valid: bool = False
                 ):
        self.root = root_split_dir
        self.img_size = img_size
        self.label_source = label_source
        self.ignore_unknown = ignore_unknown
        self.use_masks = use_masks
        self.train = train
        self.context_expand_prob = context_expand_prob
        self.context_expand_px_range = context_expand_px_range
        self.data_aug_type = data_aug_type
        self.valid = valid
        # 1) Collect
        img_glob = glob.glob(os.path.join(self.root, "*", "images", "*.tif"))
        img_glob.sort()
        mask_map: Dict[str, str] = {}
        for m in glob.glob(os.path.join(self.root, "*", "masks", "*_mask.tif")):
            mask_map[os.path.basename(m).replace("_mask.tif", ".tif")] = m

        # 2) Read CSV labels (optional)
        csv_label: Dict[str, int] = {}
        if label_source == "csv":
            for csv_path in glob.glob(os.path.join(self.root, "*", "*_samples.csv")):
                import csv as _csv
                with open(csv_path, "r") as f:
                    reader = _csv.DictReader(f)
                    for r in reader:
                        ip = r["image_path"]
                        dmg = r["damaged"].strip()
                        if dmg == "":
                            continue
                        try:
                            csv_label[os.path.basename(ip)] = int(dmg)
                        except:
                            pass

        self.rows: List[SampleRow] = []
        for im in img_glob:
            base = os.path.basename(im)
            # Label
            label: Optional[int] = None
            if label_source == "filename":
                label = parse_label_from_filename(base)
            else:
                label = csv_label.get(base, None)

            if self.ignore_unknown and label is None:
                continue

            # Mask path
            mk = mask_map.get(base, None) if self.use_masks else None
            self.rows.append(SampleRow(im, mk, label))

        # 3) Augmentations
        self.aug_img = build_geo_dg_aug(
            img_size) if train else build_geo_dg_aug(img_size, train=False)
        self.to_tensor = to_tensor_normalize(imagenet_norm=imagenet_norm)

        test_image_folder = os.path.join(
            'CROPS_50_all', "test").replace("train", '')
        test_regions = ['DS_PHR1A_202501080451578_FR1_PX_E087N28_0616_01014-calibrated',
                        'DS_PHR1B_202302070800206_FR1_PX_E036N36_0206_01222-calibrated',
                        'WV03N37_567916E036_9318052023021300000000MS00-calibrated',
                        'DS_PHR1A_202302090832223_FR1_PX_E038N37_0419_00810-calibrated']
        test_image_paths = []
        for region in test_regions:
            test_image_paths.extend(glob.glob(os.path.join(
                test_image_folder, region, "images", "*.tif")))
        self.test_image_paths = test_image_paths

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = read_tif(row.img_path)  # HWC uint8
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if self.train:
            # Optional: background masking
            m = read_mask(row.mask_path)  # HW {0,1}

            # Probabilistic dilation to include surrounding context
            if self.context_expand_prob > 0 and random.random() < self.context_expand_prob:
                lo, hi = self.context_expand_px_range
                if hi < lo:
                    hi = lo
                radius = random.randint(lo, hi)
                # Expand around building by radius pixels
                m = dilate_mask(m, radius)

            crop, mask_c = crop_around_mask(img, m, 10)
            img, m = letterbox_square(crop, mask_c, self.img_size)

            i = random.randint(0, len(self.test_image_paths) - 1)
            test_img = read_tif(self.test_image_paths[i])
            test_img = resize_or_crop(test_img, self.img_size)

            img = self.aug_img(image=img, hm_metadata=[test_img],
                               fda_metadata=[test_img])["image"]

        else:
            m = read_mask(row.mask_path)
            m = dilate_mask(m, 30)  # Expand around building by 30 pixels
            img = prepare_test_image(img, m,
                                     out_size=self.img_size,
                                     margin_px=10,
                                     zero_outside=False)
        img = self.to_tensor(image=img)["image"]  # Tensor [C,H,W], float

        label = -1 if row.label is None else row.label
        # Shape=(1,) for BCE
        label = torch.tensor([label], dtype=torch.float32)
        return img, label

    def class_counts(self) -> Tuple[int, int]:
        pos = sum(1 for r in self.rows if r.label == 1)
        neg = sum(1 for r in self.rows if r.label == 0)
        return pos, neg
