"""
IDRiD Dataset Preparation
Extracts A. Segmentation.zip and organises it into:

dataset/
├── train/
│   ├── images/        ← IDRiD_01.jpg ... IDRiD_54.jpg  (preprocessed, 512x512)
│   └── masks/         ← IDRiD_01_mask.png ... (combined MA+HE+EX+SE binary mask)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/        ← IDRiD_55.jpg ... IDRiD_81.jpg  (official test set)
    └── masks/

Split: Training 54 images → 80% train (43) / 20% val (11)
       Testing 27 images  → kept as test set (masks available for all)

Run once:  python prepare_dataset.py
"""

import zipfile, cv2, numpy as np, json
from pathlib import Path
from tqdm import tqdm

ZIP_PATH   = r"C:\Users\RAHUL R\Downloads\A. Segmentation.zip"
OUT_DIR    = Path(__file__).parent.parent / "data"
IMG_SIZE   = 512   # final size saved to disk

LESION_SUFFIXES = {
    "1. Microaneurysms": "MA",
    "2. Haemorrhages":   "HE",
    "3. Hard Exudates":  "EX",
    "4. Soft Exudates":  "SE",
    # Optic Disc excluded — not a lesion
}

TRAIN_IMG_PREFIX  = "A. Segmentation/1. Original Images/a. Training Set/"
TEST_IMG_PREFIX   = "A. Segmentation/1. Original Images/b. Testing Set/"
TRAIN_MASK_PREFIX = "A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/"
TEST_MASK_PREFIX  = "A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/"


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(img_bgr, size=IMG_SIZE):
    """Crop black borders → CLAHE → resize."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop retinal disc (remove black padding)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = 8
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(img_rgb.shape[1], x + w + pad)
        y2 = min(img_rgb.shape[0], y + h + pad)
        img_rgb = img_rgb[y1:y2, x1:x2]

    # CLAHE on L channel
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img_rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    return cv2.resize(img_rgb, (size, size))


def build_combined_mask(zf, all_names, img_id, mask_prefix, size=IMG_SIZE):
    """Merge MA + HE + EX + SE masks into one binary mask."""
    h, w = size, size
    combined = np.zeros((h, w), dtype=np.uint8)

    for folder, suffix in LESION_SUFFIXES.items():
        entry = f"{mask_prefix}{folder}/{img_id}_{suffix}.tif"
        if entry in all_names:
            with zf.open(entry) as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                m = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
                    combined = np.maximum(combined, m)

    # Binary: any lesion pixel → 255
    combined = (combined > 0).astype(np.uint8) * 255
    return combined


def process_split(zf, all_names, img_prefix, mask_prefix, out_img_dir, out_mask_dir):
    """Extract, preprocess, and save one split (train/val/test)."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    img_entries = sorted([n for n in all_names
                          if n.startswith(img_prefix) and n.endswith('.jpg')])

    stats = {"total": len(img_entries), "with_lesions": 0, "no_lesions": 0}

    for entry in tqdm(img_entries, desc=f"  {out_img_dir.parent.name}"):
        img_id = Path(entry).stem   # e.g. IDRiD_01

        # Read image
        with zf.open(entry) as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        # Preprocess image
        img_rgb = preprocess_image(img_bgr, IMG_SIZE)

        # Build combined mask
        mask = build_combined_mask(zf, all_names, img_id, mask_prefix, IMG_SIZE)

        if mask.max() > 0:
            stats["with_lesions"] += 1
        else:
            stats["no_lesions"] += 1

        # Save
        cv2.imwrite(str(out_img_dir / f"{img_id}.png"),
                    cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_mask_dir / f"{img_id}_mask.png"), mask)

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  IDRiD DATASET PREPARATION")
    print("="*60)
    print(f"  Source : {ZIP_PATH}")
    print(f"  Output : {OUT_DIR.resolve()}")
    print(f"  Size   : {IMG_SIZE}x{IMG_SIZE}")

    print("\n  Opening zip...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        all_names = set(zf.namelist())
        print(f"  Total zip entries: {len(all_names)}")

        # ── Step 1: Process official training images (54) ─────────────────────
        print("\n[1/3] Processing training images (IDRiD_01–54)...")
        train_entries = sorted([n for n in all_names
                                 if n.startswith(TRAIN_IMG_PREFIX) and n.endswith('.jpg')])
        all_ids = [Path(e).stem for e in train_entries]   # IDRiD_01 ... IDRiD_54

        # Shuffle with fixed seed for reproducibility
        np.random.seed(42)
        shuffled = np.random.permutation(all_ids)
        n_train  = int(len(shuffled) * 0.80)   # 43 train
        train_ids = list(shuffled[:n_train])
        val_ids   = list(shuffled[n_train:])

        print(f"  Train: {len(train_ids)} | Val: {len(val_ids)}")

        # Write train
        (OUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "train" / "masks").mkdir(parents=True, exist_ok=True)
        tr_stats = {"total": len(train_ids), "with_lesions": 0, "no_lesions": 0}
        for img_id in tqdm(train_ids, desc="  train"):
            entry = f"{TRAIN_IMG_PREFIX}{img_id}.jpg"
            with zf.open(entry) as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_rgb = preprocess_image(img_bgr, IMG_SIZE)
            mask    = build_combined_mask(zf, all_names, img_id, TRAIN_MASK_PREFIX, IMG_SIZE)
            cv2.imwrite(str(OUT_DIR / "train" / "images" / f"{img_id}.png"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT_DIR / "train" / "masks" / f"{img_id}_mask.png"), mask)
            if mask.max() > 0: tr_stats["with_lesions"] += 1
            else:              tr_stats["no_lesions"]   += 1

        # Write val
        (OUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "val" / "masks").mkdir(parents=True, exist_ok=True)
        va_stats = {"total": len(val_ids), "with_lesions": 0, "no_lesions": 0}
        for img_id in tqdm(val_ids, desc="  val  "):
            entry = f"{TRAIN_IMG_PREFIX}{img_id}.jpg"
            with zf.open(entry) as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_rgb = preprocess_image(img_bgr, IMG_SIZE)
            mask    = build_combined_mask(zf, all_names, img_id, TRAIN_MASK_PREFIX, IMG_SIZE)
            cv2.imwrite(str(OUT_DIR / "val" / "images" / f"{img_id}.png"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT_DIR / "val" / "masks" / f"{img_id}_mask.png"), mask)
            if mask.max() > 0: va_stats["with_lesions"] += 1
            else:              va_stats["no_lesions"]   += 1

        # ── Step 2: Process official test images (27) ─────────────────────────
        print("\n[2/3] Processing test images (IDRiD_55–81)...")
        (OUT_DIR / "test" / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "test" / "masks").mkdir(parents=True, exist_ok=True)
        test_entries = sorted([n for n in all_names
                                if n.startswith(TEST_IMG_PREFIX) and n.endswith('.jpg')])
        te_stats = {"total": len(test_entries), "with_lesions": 0, "no_lesions": 0}
        for entry in tqdm(test_entries, desc="  test "):
            img_id = Path(entry).stem
            with zf.open(entry) as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            img_rgb = preprocess_image(img_bgr, IMG_SIZE)
            mask    = build_combined_mask(zf, all_names, img_id, TEST_MASK_PREFIX, IMG_SIZE)
            cv2.imwrite(str(OUT_DIR / "test" / "images" / f"{img_id}.png"),
                        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(OUT_DIR / "test" / "masks" / f"{img_id}_mask.png"), mask)
            if mask.max() > 0: te_stats["with_lesions"] += 1
            else:              te_stats["no_lesions"]   += 1

    # ── Step 3: Save split manifest ───────────────────────────────────────────
    print("\n[3/3] Saving dataset manifest...")
    manifest = {
        "image_size": IMG_SIZE,
        "lesion_types_combined": list(LESION_SUFFIXES.values()),
        "splits": {
            "train": {**tr_stats, "ids": train_ids},
            "val":   {**va_stats, "ids": val_ids},
            "test":  {**te_stats,
                      "ids": [Path(e).stem for e in test_entries]}
        }
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  DATASET READY")
    print("="*60)
    print(f"  train/images  : {tr_stats['total']} images  "
          f"({tr_stats['with_lesions']} with lesions, {tr_stats['no_lesions']} clean)")
    print(f"  val/images    : {va_stats['total']} images  "
          f"({va_stats['with_lesions']} with lesions, {va_stats['no_lesions']} clean)")
    print(f"  test/images   : {te_stats['total']} images  "
          f"({te_stats['with_lesions']} with lesions, {te_stats['no_lesions']} clean)")
    print(f"\n  Saved to: {OUT_DIR.resolve()}")
    print(f"  Manifest: {OUT_DIR / 'manifest.json'}")
    print("="*60)
    print("\n  Next step → run:  python train_idrid.py")


if __name__ == "__main__":
    main()
