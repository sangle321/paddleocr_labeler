"""
Turns the boxes saved by the GUI into the folder layout PaddleOCR /
PaddleX's fine-tuning scripts expect:

  * Detection: Label.txt itself (see label_store.py) is already in the
    right format; split_detection_labels() just divides it into a
    train/val pair.
  * Recognition: recognition training needs one cropped image per text
    box plus a "<crop path>\t<text>" ground-truth file. export_recognition_dataset()
    perspective-crops every box out of its source image and writes that.
  * build_char_dict() collects every character that appears in the
    transcriptions, so you can extend PaddleOCR's recognition
    dictionary to cover them (important for languages/symbols not in
    the default dictionaries).
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .shapes import Shape

def imwrite(filename: str, img: np.ndarray, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        return False


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        return None



def get_rotate_crop_image(img: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    """Perspective-crop the quadrilateral `points` out of `img` into an
    upright rectangle. `points` must be ordered top-left, top-right,
    bottom-right, bottom-left (the order PaddleOCR/this app both use).
    Tall/narrow crops (vertical text) are rotated 90 degrees, matching
    the convention PaddleOCR's own recognition training data uses."""
    pts = np.array(points, dtype=np.float32)
    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3]),
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2]),
    ))
    width, height = max(width, 1), max(height, 1)
    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(
        img, matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    if height * 1.0 / max(width, 1) >= 1.5:
        crop = np.rot90(crop)
    return crop


def export_recognition_dataset(
    image_dir: str,
    labels: Dict[str, List[Shape]],
    out_dir: str,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, int]:
    """Crop every non-difficult, non-empty box out of every labeled image
    and write crop_img/*.jpg + rec_gt_train.txt + rec_gt_val.txt into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    crop_dir = os.path.join(out_dir, "crop_img")
    os.makedirs(crop_dir, exist_ok=True)

    entries: List[Tuple[str, str]] = []  # (crop_relpath, text)
    skipped_images = 0

    for relpath, shapes in labels.items():
        if not shapes:
            continue
        img_path = os.path.join(image_dir, relpath)
        img = imread(img_path)
        if img is None:
            skipped_images += 1
            continue
        base = os.path.splitext(os.path.basename(relpath))[0]
        base = "".join(c if c.isalnum() or c in "-_." else "_" for c in base)
        for i, shape in enumerate(shapes):
            if shape.difficult or not shape.transcription.strip():
                continue
            crop = get_rotate_crop_image(img, shape.points)
            if crop.size == 0:
                continue
            crop_name = f"{base}_{i:03d}.jpg"
            imwrite(os.path.join(crop_dir, crop_name), crop)
            entries.append((f"crop_img/{crop_name}", shape.transcription))

    rng = random.Random(seed)
    rng.shuffle(entries)
    n_val = int(len(entries) * val_ratio)
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    def _write(path: str, rows: List[Tuple[str, str]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rel, text in rows:
                f.write(f"{rel}\t{text}\n")

    _write(os.path.join(out_dir, "rec_gt_train.txt"), train_entries)
    _write(os.path.join(out_dir, "rec_gt_val.txt"), val_entries)

    return {
        "total_boxes": len(entries),
        "train": len(train_entries),
        "val": len(val_entries),
        "images_skipped": skipped_images,
    }


def split_detection_labels(
    labels: Dict[str, List[Shape]],
    out_dir: str,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, int]:
    """Write Label_train.txt / Label_val.txt splits (same format as
    Label.txt) for detection fine-tuning."""
    items = list(labels.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = int(len(items) * val_ratio)
    val_items = items[:n_val]
    train_items = items[n_val:]

    def _write(path: str, rows: List[Tuple[str, List[Shape]]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for rel, shapes in rows:
                payload = [s.to_dict() for s in shapes]
                f.write(f"{rel}\t{json.dumps(payload, ensure_ascii=False)}\n")

    os.makedirs(out_dir, exist_ok=True)
    _write(os.path.join(out_dir, "Label_train.txt"), train_items)
    _write(os.path.join(out_dir, "Label_val.txt"), val_items)

    return {"total_images": len(items), "train": len(train_items), "val": len(val_items)}


def build_char_dict(labels: Dict[str, List[Shape]], out_path: str) -> int:
    """Collect every unique character used across all (non-difficult)
    transcriptions and write one PaddleOCR-compatible dictionary file
    (one character per line, UTF-8). Use this to extend/replace the
    recognition model's character dictionary so training doesn't drop
    characters your text actually uses."""
    chars = set()
    for shapes in labels.values():
        for s in shapes:
            if s.difficult:
                continue
            chars.update(s.transcription)
    chars.discard("\n")
    chars.discard("\r")
    chars.discard("\t")
    ordered = sorted(chars)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in ordered:
            f.write(c + "\n")
    return len(ordered)
