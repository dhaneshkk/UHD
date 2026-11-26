import glob
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import build_augmentation_pipeline, split_augmentation_cfg
from .resize import (
    Y_BIN_RESIZE_MODE,
    Y_ONLY_RESIZE_MODE,
    Y_TRI_RESIZE_MODE,
    YUV422_RESIZE_MODE,
    normalize_resize_mode,
    resize_image_numpy,
    rgb_to_y,
    rgb_to_yuyv422,
    y_to_bin,
    y_to_tri,
)


def _resolve_path(entry: str, image_dir: str, list_path: Optional[str]) -> Optional[str]:
    """Try multiple locations to find an image path."""
    candidates = []
    if os.path.isabs(entry):
        candidates.append(entry)
    else:
        if list_path:
            candidates.append(os.path.join(os.path.dirname(list_path), entry))
        candidates.append(os.path.join(image_dir, entry))
        candidates.append(os.path.join(image_dir, os.path.basename(entry)))
        candidates.append(entry)

    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


def _read_label(path: str, class_to_idx: Dict[int, int]) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid_raw = int(float(parts[0]))
            if cid_raw not in class_to_idx:
                continue
            cid = class_to_idx[cid_raw]
            x, y, w, h = map(float, parts[1:5])
            boxes.append((cid, x, y, w, h))
    return boxes


# 12-digit zero-padded MS-COCO image id, optionally behind alphabetic prefixes such as
# "dist_" (augmented copies of the same COCO image must land on the same side of the split).
_COCO_ID_RE = re.compile(r"^(?:[A-Za-z]+_)*(\d{12})$")
# Video-like naming "<sequence>_<frame>" / "<sequence>-<frame>", e.g. "frame6_000286".
_SEQ_FRAME_RE = re.compile(r"^(.+?)[_-]\d+$")


def split_group_key(img_path: str, group_regex: Optional[str] = None) -> str:
    """Return the leakage group for an image.

    Images sharing a key are always placed on the same side of the train/val split.
    - group_regex: applied with re.search to the file stem; group(1) if it has a group,
      otherwise the whole match. Non-matching stems form their own group.
    - default ("auto"): COCO ids (incl. prefixed copies like dist_<id>) group by id;
      "<seq>_<frame>" names group by <seq>; anything else is its own group.
    """
    stem = os.path.splitext(os.path.basename(img_path))[0]
    if group_regex:
        m = re.search(group_regex, stem)
        if not m:
            return f"stem:{stem}"
        return f"re:{m.group(1) if m.groups() else m.group(0)}"
    m = _COCO_ID_RE.match(stem)
    if m:
        return f"coco:{m.group(1)}"
    m = _SEQ_FRAME_RE.match(stem)
    if m:
        return f"seq:{m.group(1)}"
    return f"stem:{stem}"


def split_items(
    items: Sequence[Tuple[str, str]],
    train_frac: float,
    val_frac: float,
    seed: int,
    mode: str = "group",
    group_regex: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Deterministic train/val split.

    mode="image": legacy per-image shuffle (adjacent video frames can leak across splits).
    mode="group": whole groups (see split_group_key) are assigned to one side, so near-duplicate
    frames never appear in both train and val. Targets are met approximately (group granularity).
    """
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1.0 + 1e-6:
        raise ValueError("train/val fractions must be >= 0 and sum to <= 1.0")
    items = sorted(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    rng = random.Random(seed)
    if mode == "image":
        rng.shuffle(items)
        return items[:n_train], items[n_train : n_train + n_val]
    if mode != "group":
        raise ValueError(f"Unknown split mode: {mode}")

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for it in items:
        groups.setdefault(split_group_key(it[0], group_regex), []).append(it)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    largest = max((len(v) for v in groups.values()), default=0)
    if n_val > 0 and largest > max(n_train, n_val):
        raise ValueError(
            f"Group split: largest group has {largest} images, more than the train ({n_train}) or val "
            f"({n_val}) target. The auto grouping is probably merging unrelated images; pass "
            "--split-group-regex to define sequences, or --split-mode image for the legacy split."
        )
    train: List[Tuple[str, str]] = []
    val: List[Tuple[str, str]] = []
    for k in keys:
        g = groups[k]
        if len(val) < n_val:
            val.extend(g)
        elif len(train) < n_train:
            train.extend(g)
        # else: dropped, matching the legacy behaviour when train+val < 1.0
    if verbose:
        n_multi = sum(1 for v in groups.values() if len(v) > 1)
        print(
            f"[split] mode=group: {len(groups)} groups ({n_multi} with >1 image, largest={largest}) -> "
            f"train={len(train)} val={len(val)} (targets {n_train}/{n_val})"
        )
    return train, val


class YoloDataset(Dataset):
    """Dataset for YOLO-format labels, filtering by a single class."""

    def __init__(
        self,
        image_dir: str,
        list_path: Optional[str] = None,
        split: str = "train",
        val_split: float = 0.1,
        seed: int = 42,
        img_size: Tuple[int, int] = (64, 64),
        resize_mode: str = "opencv_inter_nearest",
        augment: bool = False,
        class_ids: Sequence[int] = (0,),
        augment_cfg: Optional[Dict] = None,
        items: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self.image_dir = image_dir
        self.list_path = list_path
        self.split = split
        self.val_split = val_split
        self.seed = seed
        if isinstance(img_size, tuple):
            self.img_h, self.img_w = img_size
        else:
            self.img_h = self.img_w = int(img_size)
        self.resize_mode_raw = normalize_resize_mode(resize_mode)
        self.output_y_bin = self.resize_mode_raw == Y_BIN_RESIZE_MODE
        self.output_y_only = self.resize_mode_raw == Y_ONLY_RESIZE_MODE
        self.output_y_tri = self.resize_mode_raw == Y_TRI_RESIZE_MODE
        self.output_yuv422 = self.resize_mode_raw == YUV422_RESIZE_MODE
        if self.output_yuv422 or self.output_y_only or self.output_y_bin or self.output_y_tri:
            self.resize_mode = "opencv_inter_nearest"
        else:
            self.resize_mode = self.resize_mode_raw
        self.augment = augment
        self.class_ids = list(class_ids)
        if not self.class_ids:
            raise ValueError("class_ids must contain at least one class id.")
        self.class_to_idx: Dict[int, int] = {cid: i for i, cid in enumerate(self.class_ids)}
        self.augment_cfg = augment_cfg
        self.pipeline_pre = None
        self.pipeline_post = None
        if self.augment and augment_cfg:
            # convert class_swap_map to internal indices if present
            class_swap_map = None
            hf_cfg = None
            if isinstance(augment_cfg, dict):
                hf_cfg = augment_cfg.get("HorizontalFlip")
                if hf_cfg is None:
                    da_cfg = augment_cfg.get("data_augment")
                    if isinstance(da_cfg, dict):
                        hf_cfg = da_cfg.get("HorizontalFlip")
            if hf_cfg and isinstance(hf_cfg, dict) and "class_swap_map" in hf_cfg:
                class_swap_map = {}
                for k, v in hf_cfg["class_swap_map"].items():
                    if int(k) in self.class_to_idx and int(v) in self.class_to_idx:
                        class_swap_map[self.class_to_idx[int(k)]] = self.class_to_idx[int(v)]
            cfg_body = augment_cfg.get("data_augment", augment_cfg) if isinstance(augment_cfg, dict) else None
            cfg_pre = augment_cfg.get("data_augment_pre") if isinstance(augment_cfg, dict) else None
            cfg_post = augment_cfg.get("data_augment_post") if isinstance(augment_cfg, dict) else None
            if cfg_pre is None and cfg_post is None:
                cfg_pre, cfg_post = split_augmentation_cfg(cfg_body)
            self.pipeline_pre = build_augmentation_pipeline(
                cfg_pre, img_w=self.img_w, img_h=self.img_h, class_swap_map=class_swap_map, dataset=self
            )
            self.pipeline_post = build_augmentation_pipeline(
                cfg_post, img_w=self.img_w, img_h=self.img_h, class_swap_map=class_swap_map, dataset=self
            )

        if items is not None:
            items_list = items
        else:
            items_list = self._gather_items()
        if not items_list:
            raise ValueError("No samples found. Check image_dir/list_path and labels.")

        # Deterministic split
        rng = random.Random(seed)
        rng.shuffle(items_list)
        if val_split <= 0 or split == "all":
            self.items = items_list
        else:
            split_idx = int(len(items_list) * (1.0 - val_split))
            if split == "train":
                self.items = items_list[:split_idx]
            elif split == "val":
                self.items = items_list[split_idx:]
            else:
                self.items = items_list

    def _gather_items(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if self.list_path:
            with open(self.list_path, "r") as f:
                entries = [ln.strip() for ln in f if ln.strip()]
            for ent in entries:
                img_path = _resolve_path(ent, self.image_dir, self.list_path)
                if not img_path:
                    continue
                label_path = os.path.splitext(img_path)[0] + ".txt"
                pairs.append((img_path, label_path))
        else:
            patterns = ["*.jpg", "*.jpeg", "*.png"]
            for pat in patterns:
                # sorted(): glob order is filesystem-dependent, which made the seeded split
                # differ between machines.
                for img_path in sorted(glob.glob(os.path.join(self.image_dir, pat))):
                    label_path = os.path.splitext(img_path)[0] + ".txt"
                    pairs.append((img_path, label_path))

        filtered: List[Tuple[str, str]] = []
        for img_path, label_path in pairs:
            boxes = _read_label(label_path, self.class_to_idx)
            if boxes:
                filtered.append((img_path, label_path))
        return filtered

    def __len__(self) -> int:
        return len(self.items)

    def _load_raw(self, idx: int):
        img_path, label_path = self.items[idx]
        im_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        arr = im_rgb.astype(np.float32) / 255.0
        raw_boxes = _read_label(label_path, self.class_to_idx)
        boxes: List[Sequence[float]] = []
        labels: List[int] = []
        for cid, cx, cy, w, h in raw_boxes:
            boxes.append([cx, cy, w, h])
            labels.append(cid)
        boxes_np = np.array(boxes, dtype=np.float32)
        labels_np = np.array(labels, dtype=np.int64)
        return arr, boxes_np, labels_np, img_path

    def sample_random(self):
        ridx = random.randrange(len(self.items))
        return self._load_raw(ridx)

    def resize_image(self, img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        return resize_image_numpy(img, size=size, mode=self.resize_mode)

    def __getitem__(self, idx: int):
        max_retry = 10
        last_sample = None
        for attempt in range(max_retry):
            cur_idx = idx if attempt == 0 else random.randrange(len(self.items))
            arr, boxes_np, labels_np, img_path = self._load_raw(cur_idx)
            h0, w0 = arr.shape[:2]
            if self.pipeline_pre:
                img_np, boxes_np, labels_np = self.pipeline_pre(arr, boxes_np, labels_np)
            else:
                img_np = arr
            if img_np.shape[:2] != (self.img_h, self.img_w):
                img_np = self.resize_image(img_np, (self.img_h, self.img_w))
            if self.pipeline_post:
                img_np, boxes_np, labels_np = self.pipeline_post(img_np, boxes_np, labels_np)

            if self.output_yuv422:
                img_np = rgb_to_yuyv422(img_np)
            elif self.output_y_bin:
                img_np = y_to_bin(rgb_to_y(img_np))
            elif self.output_y_tri:
                img_np = y_to_tri(rgb_to_y(img_np))
            elif self.output_y_only:
                img_np = rgb_to_y(img_np)

            img_np = np.ascontiguousarray(img_np)
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            last_sample = (img_tensor, boxes_np, labels_np, img_path, (h0, w0))
            if boxes_np.size > 0:
                target = {
                    "boxes": torch.tensor(boxes_np, dtype=torch.float32),
                    "labels": torch.tensor(labels_np, dtype=torch.long),
                    "image_id": img_path,
                    "orig_size": (h0, w0),
                }
                return img_tensor, target

        # fallback: return last attempt even if empty to avoid infinite loop
        if last_sample is None:
            raise ValueError("Dataset is empty.")
        img_tensor, boxes_np, labels_np, img_path, orig_size = last_sample
        target = {
            "boxes": torch.tensor(boxes_np, dtype=torch.float32),
            "labels": torch.tensor(labels_np, dtype=torch.long),
            "image_id": img_path,
            "orig_size": orig_size,
        }
        return img_tensor, target


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
