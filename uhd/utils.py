import os
import random
from typing import Any, Dict

import numpy as np
import torch


TRUST_CKPT_ENV = "UHD_TRUST_CHECKPOINTS"


def safe_torch_load(path: str, map_location: Any = "cpu") -> Any:
    """torch.load with weights_only=True (no arbitrary pickle code execution).

    UHD checkpoints only hold tensors and plain Python metadata, so they load fine this way.
    For a checkpoint you trust that needs full unpickling (e.g. some third-party backbone
    files), set UHD_TRUST_CHECKPOINTS=1 to fall back to weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # UnpicklingError / RuntimeError depending on torch version
        if os.environ.get(TRUST_CKPT_ENV) == "1":
            print(f"[WARN] {path}: weights_only load failed ({type(exc).__name__}); "
                  f"{TRUST_CKPT_ENV}=1 set, loading with full unpickling.")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"Refusing to fully unpickle checkpoint '{path}' (weights_only load failed: {exc}). "
            f"Only if you trust this file, re-run with {TRUST_CKPT_ENV}=1."
        ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def move_targets(targets, device):
    moved = []
    for t in targets:
        moved.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t.get("labels", torch.zeros(len(t["boxes"]), dtype=torch.long)).to(device),
                "image_id": t["image_id"],
                "orig_size": t.get("orig_size"),
            }
        )
    return moved
