"""Extract UNI patch features from whole-slide images (offline tool).

Run this on a machine that has the slides; copy only the resulting ``.npy``
files (~4 MB each) to the training node, whose 30 GB disk cannot hold TCGA-BRCA
slides (~600 GB).

    python -m hill.data.uni_features \
        --wsi-dir /data/wsi --out-dir data/processed/uni_features \
        --batch-size 256 --max-patches 2048

Requires ``openslide-python``, ``timm``, ``huggingface-hub`` and ``Pillow`` (see
the commented block in environment.yml), plus access to the gated UNI weights on
the Hugging Face Hub (``MahmoodLab/UNI``).  Output: float32 ``(n_patches, 1024)``
arrays named ``<patient_id>.npy``, where the patient id is the first 12
characters of the slide filename (the TCGA barcode).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from hill.utils.logging import get_logger

log = get_logger("data.uni")

PATCH_SIZE = 256
TARGET_MAG = 20
UNI_INPUT = 224
TISSUE_FRACTION = 0.30      # minimum non-background fraction to keep a patch
BRIGHTNESS_CUTOFF = 220     # mean RGB above this is background


def _load_uni(device: str):
    import timm
    import torch
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin")
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5,
        num_classes=0, dynamic_img_size=True,
    )
    model.load_state_dict(torch.load(weights, map_location="cpu"), strict=True)
    return model.eval().to(device)


def _tissue_patches(slide_path: Path, max_patches: int, rng: np.random.Generator) -> list:
    """Grid-sample tissue patches at the target magnification."""
    import openslide
    from PIL import Image

    slide = openslide.OpenSlide(str(slide_path))
    native_mag = float(slide.properties.get("openslide.objective-power", 40))
    downsample = max(native_mag / TARGET_MAG, 1.0)
    patch_l0 = int(PATCH_SIZE * downsample)
    width, height = slide.dimensions

    coords = [
        (x, y)
        for y in range(0, height - patch_l0, patch_l0)
        for x in range(0, width - patch_l0, patch_l0)
    ]
    rng.shuffle(coords)

    patches: list[Image.Image] = []
    for x, y in coords:
        if len(patches) >= max_patches:
            break
        tile = slide.read_region((x, y), 0, (patch_l0, patch_l0)).convert("RGB")
        arr = np.asarray(tile.resize((PATCH_SIZE, PATCH_SIZE)))
        tissue = float((arr.mean(axis=2) < BRIGHTNESS_CUTOFF).mean())
        if tissue >= TISSUE_FRACTION:
            patches.append(tile.resize((UNI_INPUT, UNI_INPUT)))
    slide.close()
    return patches


def extract_slide(slide_path: Path, model, device: str, batch_size: int, max_patches: int,
                  seed: int = 0) -> np.ndarray | None:
    import torch
    from torchvision import transforms

    rng = np.random.default_rng(seed)
    patches = _tissue_patches(slide_path, max_patches, rng)
    if not patches:
        log.warning("no tissue patches found in %s", slide_path.name)
        return None
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    feats = []
    with torch.inference_mode():
        for i in range(0, len(patches), batch_size):
            batch = torch.stack([tf(p) for p in patches[i : i + batch_size]]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                feats.append(model(batch).float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract UNI patch features from WSIs")
    ap.add_argument("--wsi-dir", required=True)
    ap.add_argument("--out-dir", default="data/processed/uni_features")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-patches", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pattern", default="*.svs")
    args = ap.parse_args(argv)

    from importlib.util import find_spec

    missing = [m for m in ("openslide", "timm", "huggingface_hub", "PIL") if find_spec(m) is None]
    if missing:
        log.error(
            "missing dependencies %s — install openslide-python, timm, huggingface-hub and Pillow "
            "(see the commented block in environment.yml).", missing,
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slides = sorted(Path(args.wsi_dir).rglob(args.pattern))
    if not slides:
        log.error("no slides matching %s under %s", args.pattern, args.wsi_dir)
        return 1
    log.info("found %d slides", len(slides))

    model = _load_uni(args.device)
    done, failed = 0, 0
    for i, slide in enumerate(slides, 1):
        patient_id = slide.stem[:12]
        out_path = out_dir / f"{patient_id}.npy"
        if out_path.exists():
            continue
        t0 = time.time()
        try:
            feats = extract_slide(slide, model, args.device, args.batch_size, args.max_patches)
        except Exception as exc:  # noqa: BLE001 - one bad slide must not stop the batch
            log.error("%s failed: %s", slide.name, exc)
            failed += 1
            continue
        if feats is None:
            failed += 1
            continue
        np.save(out_path, feats)
        done += 1
        log.info("[%d/%d] %s -> %s (%.0fs)", i, len(slides), slide.name, feats.shape, time.time() - t0)

    log.info("extracted %d slides, %d failed", done, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
