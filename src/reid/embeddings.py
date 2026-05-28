"""
Person re-identification embeddings.

Tries OSNet (a CNN purpose-built for person re-ID) via the `boxmot` package
first; falls back to torchvision's ResNet18 (ImageNet-trained, general-purpose)
if boxmot isn't installed. The active model name is exposed as MODEL_NAME so
callers can tag persons in the DB with which model produced their embedding —
embeddings from different models can't be compared (different vector spaces).

OSNet vs ResNet18, in plain terms:
  ResNet18 was trained on photos of cars/cats/chairs/people. It's a general
  feature extractor — OK at telling people apart but not specialized.
  OSNet was trained specifically on millions of person photos with the
  "same person or different person?" task. Significantly better at recognizing
  the same customer across days, different lighting, different angles.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

_resnet_model = None
_resnet_transform = None
_boxmot_reid = None
_device: torch.device | None = None

# Try OSNet via boxmot (v19+ API)
try:
    from boxmot.reid.core import ReID as _BoxmotReID  # type: ignore
    _OSNET_AVAILABLE = True
except Exception:
    _OSNET_AVAILABLE = False

# OSNet model size — x1_0 is the full-size model (~10x more parameters than
# the tiny x0_25 we started with). Materially more accurate at recognizing the
# same person across days, lighting, and clothing changes. CPU cost is ~2-3x.
# Override via env var REID_MODEL if you want to experiment, e.g.
#   REID_MODEL=osnet_x0_25_msmt17    (tiny, fast, less accurate — original default)
#   REID_MODEL=osnet_x0_5_msmt17     (medium)
#   REID_MODEL=osnet_x1_0_msmt17     (full size — current default)
#   REID_MODEL=osnet_ain_x1_0_msmt17 (full size with instance norm — even more robust)
import os as _os
_REID_MODEL = _os.environ.get("REID_MODEL", "osnet_x1_0_msmt17")
MODEL_NAME = _REID_MODEL if _OSNET_AVAILABLE else "resnet18_imagenet"
EMBEDDING_DIM = 512  # all OSNet variants and ResNet18 (head stripped) produce 512-dim


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_osnet() -> None:
    """Load OSNet via boxmot. Auto-downloads weights on first use."""
    global _boxmot_reid, _device
    from pathlib import Path as _Path
    _device = _pick_device()
    # Build the weights path from MODEL_NAME so boxmot picks the right model.
    weights_path = _Path.home() / ".cache" / "boxmot" / "weights" / f"{MODEL_NAME}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    _boxmot_reid = _BoxmotReID(
        path=weights_path,
        device=str(_device),
        half=False,
    )
    print(f"[reid] loaded {MODEL_NAME} via boxmot (purpose-built for person re-ID) on device={_device}")


def _load_resnet18() -> None:
    global _resnet_model, _resnet_transform, _device
    import torchvision.models as models
    import torchvision.transforms as T

    _device = _pick_device()
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone.eval().to(_device)
    _resnet_model = backbone
    _resnet_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"[reid] loaded ResNet18 (general-purpose fallback) on device={_device}")


def _load() -> None:
    if _resnet_model is not None or _boxmot_reid is not None:
        return
    if _OSNET_AVAILABLE:
        try:
            _load_osnet()
            return
        except Exception as e:
            print(f"[reid] OSNet load failed ({e}); falling back to ResNet18")
            globals()["MODEL_NAME"] = "resnet18_imagenet"
    _load_resnet18()


@torch.no_grad()
def embed_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Embed a BGR person crop into a 512-dim L2-normalized vector."""
    _load()
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 4:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    if _boxmot_reid is not None:
        # boxmot.reid.core.ReID is callable; pass the BGR crop directly.
        # It handles resize, normalization, and L2-norms internally.
        features = _boxmot_reid(crop_bgr)
        feat = np.asarray(features[0], dtype=np.float32).flatten()
    else:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = _resnet_transform(rgb).unsqueeze(0).to(_device)
        feat = _resnet_model(tensor).squeeze(0).cpu().numpy().astype(np.float32)

    norm = float(np.linalg.norm(feat))
    if norm > 0:
        feat = feat / norm
    return feat
