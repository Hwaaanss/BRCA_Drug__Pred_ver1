#!/usr/bin/env python3
"""Extract UNI (ViT-Large) patch features from TCGA-BRCA WSI slides.

Optimized pipeline:
  - Multiprocess patch extraction (CPU-bound) with prefetching
  - Large-batch GPU inference (GPU-bound)
  - Overlap CPU and GPU work via concurrent.futures

Usage:
  python 07_extract_uni_features.py [--batch_size 512] [--extract_workers 8]
"""
import os
import sys
import csv
import time
import argparse
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import openslide
import timm
from huggingface_hub import hf_hub_download

# ─── Config ───────────────────────────────────────────────────────────
WSI_DIR = "./data/05_morphology/wsi"
TARGET_CSV = "./data/05_morphology/wsi_target_3modal.csv"
OUT_DIR = "./data/05_morphology/features"
LOG_FILE = "./data/logs/uni_feature_extraction.log"

PATCH_SIZE = 256
TARGET_MAG = 20
UNI_INPUT = 224
TISSUE_THRESH = 0.7
BRIGHTNESS_THRESH = 220
COMPLETE_RATIO = 0.99


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def is_complete_file(path, expected_mb):
    if not os.path.exists(path):
        return False
    actual_mb = os.path.getsize(path) / (1024**2)
    return actual_mb >= expected_mb * COMPLETE_RATIO


# ─── Patch extraction (runs in worker processes) ─────────────────────
def extract_patches_from_slide(svs_path: str):
    """Extract tissue patches from a single SVS file. Runs in subprocess."""
    try:
        slide = openslide.OpenSlide(svs_path)
    except Exception as e:
        return []

    native_mag = float(slide.properties.get("openslide.objective-power", 40))
    downsample = native_mag / TARGET_MAG
    patch_size_l0 = int(PATCH_SIZE * downsample)

    if slide.level_count > 1 and slide.level_downsamples[1] <= 4.5:
        read_level = 1
    else:
        read_level = 0
    level_ds = slide.level_downsamples[read_level]
    read_size = int(patch_size_l0 / level_ds)

    w, h = slide.dimensions

    # Thumbnail pre-screening
    thumb = slide.get_thumbnail((512, 512))
    thumb_arr = np.array(thumb.convert("RGB"))
    thumb_gray = np.mean(thumb_arr, axis=2)
    scale_x = thumb_arr.shape[1] / w
    scale_y = thumb_arr.shape[0] / h

    patches = []
    for y in range(0, h - patch_size_l0 + 1, patch_size_l0):
        for x in range(0, w - patch_size_l0 + 1, patch_size_l0):
            tx = int(x * scale_x)
            ty = int(y * scale_y)
            tw = max(1, int(patch_size_l0 * scale_x))
            th = max(1, int(patch_size_l0 * scale_y))
            tx2 = min(tx + tw, thumb_gray.shape[1])
            ty2 = min(ty + th, thumb_gray.shape[0])
            thumb_region = thumb_gray[ty:ty2, tx:tx2]
            if thumb_region.size == 0 or np.mean(thumb_region < BRIGHTNESS_THRESH) < 0.5:
                continue

            region = slide.read_region((x, y), read_level, (read_size, read_size))
            region = region.convert("RGB")
            if region.size != (PATCH_SIZE, PATCH_SIZE):
                region = region.resize((PATCH_SIZE, PATCH_SIZE), Image.LANCZOS)

            arr = np.array(region)
            if np.mean(np.mean(arr, axis=2) < BRIGHTNESS_THRESH) > TISSUE_THRESH:
                patches.append(region)

    slide.close()
    return patches


def extract_patient_patches(slides):
    """Extract patches from all slides of one patient (single process)."""
    all_patches = []
    for svs_path in slides:
        all_patches.extend(extract_patches_from_slide(svs_path))
    return all_patches


# ─── GPU inference ────────────────────────────────────────────────────
class PatchTensorDataset(Dataset):
    """Pre-transformed tensor dataset for fast GPU loading."""
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--extract_workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=3,
                        help="Number of patients to prefetch patches for")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device} | batch_size: {args.batch_size} | "
        f"extract_workers: {args.extract_workers} | prefetch: {args.prefetch}")

    # ── Load target list and keep only complete patient slide sets ──
    with open(TARGET_CSV) as f:
        rows = list(csv.DictReader(f))

    patient_rows = defaultdict(list)
    for r in rows:
        patient_rows[r["patient_id"]].append(r)

    patient_slides = {}
    incomplete = []
    for pid, slide_rows in patient_rows.items():
        complete_slides = []
        for r in slide_rows:
            svs_path = os.path.join(WSI_DIR, r["file_name"])
            if is_complete_file(svs_path, float(r["file_size_MB"])):
                complete_slides.append(svs_path)

        if len(complete_slides) == len(slide_rows):
            patient_slides[pid] = complete_slides
        else:
            incomplete.append((pid, len(complete_slides), len(slide_rows)))

    done = {f.replace(".pt", "") for f in os.listdir(OUT_DIR) if f.endswith(".pt")}
    todo_items = [(pid, slides) for pid, slides in patient_slides.items() if pid not in done]
    log(f"Target patients: {len(patient_rows)}, complete slide sets: {len(patient_slides)}, "
        f"incomplete/skipped: {len(incomplete)}, done: {len(done)}, remaining: {len(todo_items)}")
    if incomplete:
        examples = ", ".join(f"{pid}({have}/{total})" for pid, have, total in incomplete[:5])
        log(f"Incomplete examples: {examples}")

    if not todo_items:
        log("Nothing to do!")
        return

    # ── Load UNI model ──
    log("Loading UNI model...")
    weights_path = hf_hub_download("MahmoodLab/uni", filename="pytorch_model.bin")
    model = timm.create_model(
        "vit_large_patch16_224", init_values=1.0, num_classes=0,
        dynamic_img_size=True, pretrained=False,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    # Use torch.compile for faster inference if available
    try:
        model = torch.compile(model, mode="reduce-overhead")
        log("UNI model loaded + torch.compile enabled")
    except Exception:
        log("UNI model loaded (torch.compile not available)")

    # ── Transform ──
    transform = transforms.Compose([
        transforms.Resize(UNI_INPUT, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(UNI_INPUT),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Prefetch + GPU pipeline ──
    # Use ProcessPoolExecutor for CPU-bound patch extraction
    # While GPU processes current patient, CPU extracts next patients
    start_time = time.time()
    completed = 0

    def transform_patches(patches):
        """Apply transforms to patches on CPU (in main process)."""
        return torch.stack([transform(p) for p in patches])

    def gpu_inference(tensor_batch):
        """Run UNI on a pre-transformed tensor batch."""
        features = []
        for i in range(0, len(tensor_batch), args.batch_size):
            batch = tensor_batch[i:i + args.batch_size].to(device, non_blocking=True)
            with torch.no_grad():
                feat = model(batch)
            features.append(feat.cpu())
        return torch.cat(features, dim=0)

    with ProcessPoolExecutor(max_workers=args.extract_workers) as pool:
        # Submit initial prefetch batch
        prefetch_size = min(args.prefetch, len(todo_items))
        futures = {}
        next_submit = 0

        # Submit first batch of prefetch jobs
        for idx in range(prefetch_size):
            pid, slides = todo_items[idx]
            future = pool.submit(extract_patient_patches, slides)
            futures[future] = (idx, pid, slides)
            next_submit = idx + 1

        # Process results as they complete, keep submitting new jobs
        while futures:
            # Wait for any future to complete
            done_futures = []
            for future in as_completed(futures):
                done_futures.append(future)
                break  # Process one at a time to maintain order for logging

            for future in done_futures:
                idx, pid, slides = futures.pop(future)
                patches = future.result()

                # Submit next job immediately to keep workers busy
                if next_submit < len(todo_items):
                    next_pid, next_slides = todo_items[next_submit]
                    next_future = pool.submit(extract_patient_patches, next_slides)
                    futures[next_future] = (next_submit, next_pid, next_slides)
                    next_submit += 1

                t0 = time.time()

                if len(patches) == 0:
                    completed += 1
                    log(f"[{completed}/{len(todo_items)}] {pid} | 0 patches — SKIPPED")
                    continue

                # Transform + GPU inference
                tensor_batch = transform_patches(patches)
                features = gpu_inference(tensor_batch)

                # Save
                torch.save(features, os.path.join(OUT_DIR, f"{pid}.pt"))
                completed += 1

                elapsed = time.time() - t0
                total_elapsed = time.time() - start_time
                avg = total_elapsed / completed
                eta_min = avg * (len(todo_items) - completed) / 60

                if completed % 5 == 0 or completed <= 3:
                    log(f"[{completed}/{len(todo_items)}] {pid} | {len(slides)} slides | "
                        f"{len(patches)} patches | {features.shape} | "
                        f"{elapsed:.0f}s | ETA: {eta_min:.0f}min")

                del patches, tensor_batch, features

    total_done = len([f for f in os.listdir(OUT_DIR) if f.endswith(".pt")])
    total_time = (time.time() - start_time) / 3600
    log(f"\nDone! {total_done} patient feature files in {OUT_DIR}")
    log(f"Total time: {total_time:.1f} hours")


if __name__ == "__main__":
    main()
