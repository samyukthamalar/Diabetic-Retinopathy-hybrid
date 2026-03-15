"""
IDRiD Patch-Based Training Pipeline
- Extracts 256x256 patches centred on lesion regions (lesion-focused sampling)
- Turns 43 images into 400+ training patches
- Lightweight U-Net (1.2M params) suited for small dataset
- Weighted Focal + Dice loss for 97.7% background imbalance
Run: python train_idrid.py
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np, cv2, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings('ignore')

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATCH_SIZE  = 256
PATCHES_PER_IMG = 10   # lesion-centred patches per image → 43×10 = 430 train patches
BATCH_SIZE  = 8
EPOCHS      = 80
LR          = 1e-3
POS_WEIGHT  = 43.0     # (1 - 0.0228) / 0.0228
RESULTS_DIR = Path(__file__).parent.parent / 'models'
DATA_DIR    = Path(__file__).parent.parent / 'data'
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Device: {DEVICE} | Patch: {PATCH_SIZE} | Epochs: {EPOCHS}")


# ── Patch extraction ──────────────────────────────────────────────────────────
def extract_patches(img, mask, n_patches=PATCHES_PER_IMG, patch_size=PATCH_SIZE):
    """
    Extract n_patches from an image.
    - 70% centred on a random lesion pixel (lesion-focused)
    - 30% random location (context diversity)
    """
    H, W = img.shape[:2]
    half = patch_size // 2
    patches_img, patches_msk = [], []

    lesion_yx = np.argwhere(mask > 0)   # all lesion pixel coords

    for i in range(n_patches):
        if len(lesion_yx) > 0 and np.random.rand() < 0.70:
            # Centre on a random lesion pixel
            cy, cx = lesion_yx[np.random.randint(len(lesion_yx))]
        else:
            # Random location
            cy = np.random.randint(half, H - half)
            cx = np.random.randint(half, W - half)

        # Clamp so patch stays inside image
        cy = int(np.clip(cy, half, H - half))
        cx = int(np.clip(cx, half, W - half))

        p_img = img [cy-half:cy+half, cx-half:cx+half]
        p_msk = mask[cy-half:cy+half, cx-half:cx+half]

        patches_img.append(p_img)
        patches_msk.append(p_msk)

    return patches_img, patches_msk


# ── Dataset ───────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, split='train', augment=False):
        self.augment = augment
        img_dir  = DATA_DIR / split / 'images'
        mask_dir = DATA_DIR / split / 'masks'

        img_paths  = sorted(img_dir.glob('*.png'))
        mask_index = {p.stem.replace('_mask', ''): p
                      for p in mask_dir.glob('*_mask.png')}

        self.patches_img = []
        self.patches_msk = []

        n_per = PATCHES_PER_IMG if split == 'train' else 6

        for img_path in img_paths:
            img_id = img_path.stem
            img  = cv2.imread(str(img_path))
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mp = mask_index.get(img_id)
            if mp and mp.exists():
                msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            else:
                msk = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            pi, pm = extract_patches(img, msk, n_patches=n_per)
            self.patches_img.extend(pi)
            self.patches_msk.extend(pm)

        print(f"  [{split}] {len(img_paths)} images → {len(self.patches_img)} patches")

    def __len__(self): return len(self.patches_img)

    def __getitem__(self, idx):
        img = self.patches_img[idx].astype(np.float32) / 255.0
        msk = (self.patches_msk[idx] > 0).astype(np.float32)

        if self.augment:
            if np.random.rand() > 0.5: img = np.fliplr(img).copy(); msk = np.fliplr(msk).copy()
            if np.random.rand() > 0.5: img = np.flipud(img).copy(); msk = np.flipud(msk).copy()
            k = np.random.randint(0, 4)
            img = np.rot90(img, k).copy(); msk = np.rot90(msk, k).copy()
            img = np.clip(img * np.random.uniform(0.80, 1.25), 0, 1)
            img = np.clip(img + np.random.uniform(-0.06, 0.06), 0, 1)
            img = np.clip(img + np.random.normal(0, 0.01, img.shape), 0, 1)

        img_t = torch.from_numpy(img).permute(2, 0, 1).float()
        msk_t = torch.from_numpy(msk).unsqueeze(0).float()
        return img_t, msk_t


# ── Lightweight U-Net (1.2M params — right-sized for 43 images) ───────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class LightUNet(nn.Module):
    """
    Lightweight U-Net with Squeeze-Excitation bottleneck.
    ~1.2M parameters — appropriate for 43-image dataset with patch training.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBnRelu(3,  16)
        self.enc2 = ConvBnRelu(16, 32)
        self.enc3 = ConvBnRelu(32, 64)
        self.enc4 = ConvBnRelu(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.2)
        # Bottleneck + SE
        self.bottleneck = ConvBnRelu(128, 256)
        self.se_fc1 = nn.Linear(256, 16)
        self.se_fc2 = nn.Linear(16, 256)
        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec4 = ConvBnRelu(256, 128)
        self.up3 = nn.ConvTranspose2d(128,  64, 2, stride=2); self.dec3 = ConvBnRelu(128,  64)
        self.up2 = nn.ConvTranspose2d( 64,  32, 2, stride=2); self.dec2 = ConvBnRelu( 64,  32)
        self.up1 = nn.ConvTranspose2d( 32,  16, 2, stride=2); self.dec1 = ConvBnRelu( 32,  16)
        self.out_conv = nn.Conv2d(16, 1, 1)

    def squeeze_excite(self, x):
        b, c, _, _ = x.shape
        s = torch.relu(self.se_fc1(x.mean(dim=[2, 3])))
        return x * torch.sigmoid(self.se_fc2(s)).view(b, c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bn = self.drop(self.squeeze_excite(self.bottleneck(self.pool(e4))))
        d4 = self.dec4(torch.cat([self.up4(bn), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out_conv(d1))


# ── Weighted Focal + Dice Loss ────────────────────────────────────────────────
class FocalDiceLoss(nn.Module):
    def __init__(self, pos_weight=POS_WEIGHT, gamma=2.0, smooth=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma      = gamma
        self.smooth     = smooth

    def weighted_focal(self, pred, target):
        bce = -(self.pos_weight * target * torch.log(pred + 1e-8)
                + (1 - target) * torch.log(1 - pred + 1e-8))
        pt      = torch.where(target == 1, pred, 1 - pred)
        focal_w = (1 - pt) ** self.gamma
        return (focal_w * bce).mean()

    def dice_loss(self, pred, target):
        p = pred.view(-1); t = target.view(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, pred, target):
        return 0.4 * self.weighted_focal(pred, target) + 0.6 * self.dice_loss(pred, target)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred, target, thresh=0.5):
    p  = (pred > thresh).float(); t = target.float()
    tp = (p * t).sum().item();    fp = (p * (1-t)).sum().item()
    fn = ((1-p) * t).sum().item(); tn = ((1-p) * (1-t)).sum().item()
    dice = (2*tp) / (2*tp + fp + fn + 1e-8)
    iou  = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2*prec*rec / (prec + rec + 1e-8)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return dict(dice=dice, iou=iou, precision=prec, recall=rec, f1=f1, accuracy=acc)


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train(); total = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


def eval_epoch(model, loader, criterion):
    model.eval(); total = 0
    all_m = {k: [] for k in ['dice','iou','precision','recall','f1','accuracy']}
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            out = model(imgs)
            total += criterion(out, masks).item()
            for k, v in compute_metrics(out, masks).items():
                all_m[k].append(v)
    return total / len(loader), {k: float(np.mean(v)) for k, v in all_m.items()}


# ── Full-image evaluation (for final test metrics) ────────────────────────────
def eval_full_images(model, split='test', thresh=0.5):
    """Run inference on full 512x512 images using sliding window patches."""
    model.eval()
    img_dir  = DATA_DIR / split / 'images'
    mask_dir = DATA_DIR / split / 'masks'
    img_paths = sorted(img_dir.glob('*.png'))

    all_m = {k: [] for k in ['dice','iou','precision','recall','f1','accuracy']}

    with torch.no_grad():
        for img_path in img_paths:
            img_id = img_path.stem
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            mp = mask_dir / f"{img_id}_mask.png"
            msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
            if msk is None:
                continue
            msk = (msk > 0).astype(np.float32)

            # Sliding window inference
            H, W = img.shape[:2]
            P = PATCH_SIZE
            pred_full  = np.zeros((H, W), dtype=np.float32)
            count_full = np.zeros((H, W), dtype=np.float32)

            for y in range(0, H - P + 1, P // 2):
                for x in range(0, W - P + 1, P // 2):
                    patch = img[y:y+P, x:x+P]
                    t = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
                    out = model(t).squeeze().cpu().numpy()
                    pred_full[y:y+P, x:x+P] += out
                    count_full[y:y+P, x:x+P] += 1

            count_full = np.maximum(count_full, 1)
            pred_full /= count_full

            pred_t = torch.from_numpy(pred_full).unsqueeze(0).unsqueeze(0)
            msk_t  = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
            for k, v in compute_metrics(pred_t, msk_t, thresh).items():
                all_m[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_m.items()}


# ── Visualise predictions ─────────────────────────────────────────────────────
def save_predictions(model, n=6):
    """Run sliding window on full test images and save visual results."""
    model.eval()
    img_dir  = DATA_DIR / 'test' / 'images'
    mask_dir = DATA_DIR / 'test' / 'masks'
    img_paths = sorted(img_dir.glob('*.png'))[:n]

    fig, axes = plt.subplots(n, 4, figsize=(16, n*4))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('IDRiD — Lesion Segmentation Results (Real Data)',
                 color='white', fontsize=13, fontweight='bold')

    with torch.no_grad():
        for row, img_path in enumerate(img_paths):
            img_id = img_path.stem
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_f = img.astype(np.float32) / 255.0

            mp  = mask_dir / f"{img_id}_mask.png"
            msk = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else np.zeros(img.shape[:2], np.uint8)
            msk = (msk > 0).astype(np.float32)

            # Sliding window
            H, W = img_f.shape[:2]; P = PATCH_SIZE
            pred_full  = np.zeros((H, W), np.float32)
            count_full = np.zeros((H, W), np.float32)
            for y in range(0, H - P + 1, P // 2):
                for x in range(0, W - P + 1, P // 2):
                    patch = img_f[y:y+P, x:x+P]
                    t = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
                    out = model(t).squeeze().cpu().numpy()
                    pred_full[y:y+P, x:x+P] += out
                    count_full[y:y+P, x:x+P] += 1
            pred_full /= np.maximum(count_full, 1)

            overlay  = img.copy()
            pred_bin = (pred_full > 0.5).astype(np.uint8)
            overlay[pred_bin == 1] = np.clip(
                overlay[pred_bin == 1].astype(int) * 0.4 + np.array([255, 50, 50]) * 0.6, 0, 255
            ).astype(np.uint8)

            for col, (im, title, cmap) in enumerate([
                (img,       f'{img_id}\nInput',  None),
                (msk,       'Ground Truth',      'Reds'),
                (pred_full, 'Prediction',        'Reds'),
                (overlay,   'Overlay',           None),
            ]):
                ax = axes[row, col]
                ax.set_facecolor('#161b22')
                ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
                ax.set_title(title, color='white', fontsize=9)
                ax.axis('off')

    plt.tight_layout()
    out = RESULTS_DIR / 'idrid_predictions.png'
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"  Saved: {out}")


def save_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Training History — IDRiD Patch-Based Training', fontsize=13, fontweight='bold')
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss (Focal+Dice)'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history['val_dice'], label='Dice'); axes[1].plot(history['val_iou'], label='IoU')
    axes[1].set_title('Dice & IoU'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].plot(history['val_precision'], label='Precision')
    axes[2].plot(history['val_recall'],    label='Recall')
    axes[2].plot(history['val_f1'],        label='F1')
    axes[2].set_title('Precision / Recall / F1'); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / 'training_curves.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  IDRiD PATCH-BASED TRAINING — Lightweight U-Net + SE")
    print("="*60)

    if not (DATA_DIR / 'train' / 'images').exists():
        print("\n  ERROR: dataset/ not found. Run:  python prepare_dataset.py")
        return

    # Datasets
    print("\n[1/4] Building patch datasets...")
    train_ds = PatchDataset('train', augment=True)
    val_ds   = PatchDataset('val',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    print("\n[2/4] Building model...")
    model  = LightUNet().to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")

    criterion = FocalDiceLoss(pos_weight=POS_WEIGHT, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Train
    print(f"\n[3/4] Training for {EPOCHS} epochs on patches...")
    history   = {k: [] for k in ['train_loss','val_loss','val_dice','val_iou',
                                   'val_precision','val_recall','val_f1']}
    best_dice = 0.0
    best_path = RESULTS_DIR / 'best_model.pth'

    for epoch in range(1, EPOCHS + 1):
        tr_loss       = train_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_m = eval_epoch(model, val_loader,   criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        for k in ['dice','iou','precision','recall','f1']:
            history[f'val_{k}'].append(vl_m[k])

        if vl_m['dice'] > best_dice:
            best_dice = vl_m['dice']
            torch.save(model.state_dict(), best_path)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
              f"Dice {vl_m['dice']:.4f} | "
              f"IoU {vl_m['iou']:.4f} | "
              f"F1 {vl_m['f1']:.4f}")

    print(f"\n  Best Val Dice: {best_dice:.4f}")

    # Final evaluation on full test images (sliding window)
    print("\n[4/4] Final evaluation on full test images (sliding window)...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    test_m = eval_full_images(model, split='test')

    print("\n  ┌──────────────────────────────────────┐")
    print("  │   FINAL TEST RESULTS — IDRiD Real   │")
    print("  ├──────────────────────────────────────┤")
    for k, v in test_m.items():
        print(f"  │  {k.capitalize():<12}: {v:.4f}                 │")
    print("  └──────────────────────────────────────┘")

    save_curves(history)
    save_predictions(model)

    report = {
        "model": "Lightweight U-Net + SE (Patch-Based, IDRiD Real Data)",
        "dataset": "IDRiD Segmentation — 54 train + 27 test images",
        "patch_size": PATCH_SIZE,
        "patches_per_image": PATCHES_PER_IMG,
        "epochs_trained": EPOCHS,
        "best_val_dice": round(best_dice, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_m.items()},
        "model_saved": str(best_path)
    }
    with open(RESULTS_DIR / 'final_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: final_results/final_report.json")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  Dice: {test_m['dice']:.4f} | IoU: {test_m['iou']:.4f} | F1: {test_m['f1']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f} | Recall: {test_m['recall']:.4f}")
    print(f"  Accuracy:  {test_m['accuracy']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
