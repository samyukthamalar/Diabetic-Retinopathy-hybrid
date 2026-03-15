"""
System Architecture for Multimodal DR Lesion Segmentation
──────────────────────────────────────────────────────────
STAGE 1 │ INPUT          : Fundus (IDRiD Dataset) + OCT Images
STAGE 2 │ PREPROCESSING  : Resize → Denoise → Normalize → Augment
STAGE 3 │ FEATURE EXTRACT: CNN encoder (LightUNet encoder path)
STAGE 4 │ HYBRID MODEL   : U-Net Segmentation + Severity Classifier
STAGE 5 │ OUTPUT         : Lesion Segmentation (MA/HE/EX) + DR Severity → Diagnosis

Usage:
  Fundus : python src/predict.py --fundus path/to/fundus.jpg
  OCT    : python src/predict.py --oct    path/to/oct.jpg
  Both   : python src/predict.py --fundus path/to/fundus.jpg --oct path/to/oct.jpg
"""

import torch, torch.nn as nn
import numpy as np, cv2, argparse, sys, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
warnings.filterwarnings('ignore')

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH  = Path(__file__).parent.parent / 'models' / 'best_model.pth'
REPORT_PATH = Path(__file__).parent.parent / 'models' / 'final_report.json'
PATCH_SIZE  = 256
OUT_DIR     = Path(__file__).parent.parent / 'outputs'

DR_STAGES = [
    (0.08,  2,  "No DR",            "#27ae60", "NO DR",    "No diabetic retinopathy detected."),
    (0.40,  10, "Mild NPDR",        "#f1c40f", "MILD",     "Mild NPDR — Microaneurysms present."),
    (1.20,  30, "Moderate NPDR",    "#e67e22", "MODERATE", "Moderate NPDR — Hemorrhages and exudates detected."),
    (3.00,  80, "Severe NPDR",      "#e74c3c", "SEVERE",   "Severe NPDR — Extensive lesions. Urgent referral."),
    (1e9,  1e9, "Proliferative DR", "#8e44ad", "PDR",      "Proliferative DR — Immediate treatment required."),
]


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID DEEP LEARNING MODEL — LightUNet + Squeeze-Excitation
# Matches exactly the weights saved in models/best_model.pth
# ═══════════════════════════════════════════════════════════════════════════════
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
    Hybrid U-Net with Squeeze-Excitation bottleneck.
    Encoder path = CNN Feature Extraction (Stage 3 in architecture).
    Decoder path = U-Net Segmentation     (Stage 4 in architecture).
    """
    def __init__(self):
        super().__init__()
        # ── CNN Encoder (Feature Extraction) ──────────────────────────────────
        self.enc1 = ConvBnRelu(3,   16)   # 512 → 512
        self.enc2 = ConvBnRelu(16,  32)   # 256 → 256
        self.enc3 = ConvBnRelu(32,  64)   # 128 → 128
        self.enc4 = ConvBnRelu(64, 128)   #  64 →  64
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.2)
        # ── Bottleneck + Squeeze-Excitation ───────────────────────────────────
        self.bottleneck = ConvBnRelu(128, 256)
        self.se_fc1 = nn.Linear(256, 16)
        self.se_fc2 = nn.Linear(16, 256)
        # ── U-Net Decoder (Segmentation) ──────────────────────────────────────
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec4 = ConvBnRelu(256, 128)
        self.up3 = nn.ConvTranspose2d(128,  64, 2, stride=2); self.dec3 = ConvBnRelu(128,  64)
        self.up2 = nn.ConvTranspose2d( 64,  32, 2, stride=2); self.dec2 = ConvBnRelu( 64,  32)
        self.up1 = nn.ConvTranspose2d( 32,  16, 2, stride=2); self.dec1 = ConvBnRelu( 32,  16)
        self.out_conv = nn.Conv2d(16, 1, 1)   # → lesion probability map

    def squeeze_excite(self, x):
        b, c, _, _ = x.shape
        s = torch.relu(self.se_fc1(x.mean(dim=[2, 3])))
        return x * torch.sigmoid(self.se_fc2(s)).view(b, c, 1, 1)

    def forward(self, x):
        # CNN feature extraction
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Hybrid bottleneck (SE attention)
        bn = self.drop(self.squeeze_excite(self.bottleneck(self.pool(e4))))
        # U-Net segmentation decoder
        d4 = self.dec4(torch.cat([self.up4(bn), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out_conv(d1))


def load_model():
    m = LightUNet().to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    return m

def load_metrics():
    try:
        with open(REPORT_PATH) as f: return json.load(f).get('test_metrics', {})
    except: return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — IMAGE PREPROCESSING
# Resize, Denoise, Normalize, (Augment handled at training time)
# ═══════════════════════════════════════════════════════════════════════════════
def preprocess_fundus(path):
    """Fundus: crop black border → denoise → CLAHE → normalize → 512×512."""
    img = cv2.imread(str(path))
    if img is None: print(f"ERROR: Cannot read '{path}'"); sys.exit(1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig = img.copy()

    # Resize: crop retinal disc, remove black padding
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thr = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        p = 8
        img = img[max(0,y-p):min(img.shape[0],y+h+p),
                  max(0,x-p):min(img.shape[1],x+w+p)]

    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, h=5, hColor=5,
                                          templateWindowSize=7, searchWindowSize=21)
    # Normalize (CLAHE on L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    # Resize to model input size
    prep = cv2.resize(img, (512, 512))
    return orig, prep


def preprocess_oct(path):
    """OCT: denoise → CLAHE normalize → resize to 512×512."""
    img = cv2.imread(str(path))
    if img is None: print(f"ERROR: Cannot read '{path}'"); sys.exit(1)
    orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    den = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    # Normalize (CLAHE)
    enh = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(den)
    # Resize
    gray_rgb = cv2.resize(cv2.cvtColor(den, cv2.COLOR_GRAY2RGB), (512, 512))
    enh_rs   = cv2.resize(enh, (512, 512))
    return orig, gray_rgb, enh_rs


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3+4 — CNN FEATURE EXTRACTION → HYBRID U-NET SEGMENTATION
# Sliding-window patch inference over the full 512×512 image
# ═══════════════════════════════════════════════════════════════════════════════
def cnn_extract_and_segment(model, img_rgb):
    """
    Passes image through the full Hybrid Model pipeline:
      CNN encoder → SE bottleneck → U-Net decoder → lesion probability map
    Uses overlapping 256×256 patches with 50% stride for full-image coverage.
    """
    H, W  = img_rgb.shape[:2]
    imgf  = img_rgb.astype(np.float32) / 255.0   # pixel normalisation [0,1]
    pred  = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    step  = PATCH_SIZE // 2                        # 50% overlap

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, step):
            for x in range(0, W - PATCH_SIZE + 1, step):
                patch = imgf[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                out = model(t).squeeze().cpu().numpy()
                pred [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += out
                count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    return pred / np.maximum(count, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — SEVERITY CLASSIFIER
# Analyses segmentation map → DR stage (Mild / Moderate / Severe / PDR)
# Also identifies lesion types: Microaneurysms, Hemorrhages, Exudates
# ═══════════════════════════════════════════════════════════════════════════════
def classify_severity_fundus(pred_map, thresh=0.5):
    """
    Severity classification from U-Net output:
    - Lesion burden (% pixels) → stage threshold
    - Connected component count → lesion count
    - Lesion size distribution → type estimation (MA / HE / EX)
    """
    binary    = (pred_map > thresh).astype(np.uint8)
    burden    = float(binary.mean() * 100)
    n_labels, label_map = cv2.connectedComponents(binary)
    n_lesions = n_labels - 1

    # Estimate lesion types by connected-component area
    lesion_types = []
    if n_lesions > 0:
        areas = [int((label_map == i).sum()) for i in range(1, n_labels)]
        small  = sum(1 for a in areas if a < 50)    # microaneurysms (tiny dots)
        medium = sum(1 for a in areas if 50 <= a < 500)  # hemorrhages
        large  = sum(1 for a in areas if a >= 500)  # hard exudates / patches
        if small  > 0: lesion_types.append(f"Microaneurysms ({small})")
        if medium > 0: lesion_types.append(f"Hemorrhages ({medium})")
        if large  > 0: lesion_types.append(f"Exudates ({large})")

    # Map to DR stage
    for blim, clim, name, color, short, desc in DR_STAGES:
        if burden < blim or n_lesions < clim:
            return name, color, short, desc, burden, n_lesions, lesion_types
    s = DR_STAGES[-1]
    return s[2], s[3], s[4], s[5], burden, n_lesions, lesion_types


def classify_severity_oct(enh):
    """
    OCT severity classification via biomarker analysis:
    - Bright regions → Drusen / Hard Exudates
    - Dark regions   → Subretinal / Intraretinal Fluid
    - Thickness profile → retinal layer disruption score
    """
    H, W = enh.shape
    mv, sv = float(enh.mean()), float(enh.std())

    _, bm = cv2.threshold(enh, min(255, int(mv + 2*sv)),   255, cv2.THRESH_BINARY)
    _, dm = cv2.threshold(enh, max(0,   int(mv - 1.5*sv)), 255, cv2.THRESH_BINARY_INV)

    # Remove border noise
    mg = int(H * 0.15)
    bm[:mg, :] = 0; bm[H-mg:, :] = 0
    dm[:mg, :] = 0; dm[H-mg:, :] = 0

    # Remove tiny specks
    min_area = H * W * 0.0005
    for mask in [bm, dm]:
        nc, lbl = cv2.connectedComponents(mask)
        for l in range(1, nc):
            if (lbl == l).sum() < min_area: mask[lbl == l] = 0

    bright_pct = float((bm > 0).sum()) / (H * W) * 100
    dark_pct   = float((dm > 0).sum()) / (H * W) * 100

    # Retinal thickness profile per column
    th = np.array([
        float(np.where(enh[:, c] > mv*0.5)[0][-1] - np.where(enh[:, c] > mv*0.5)[0][0]) / H * 100
        if len(np.where(enh[:, c] > mv*0.5)[0]) > 5 else 0
        for c in range(W)
    ], dtype=np.float32)

    score = (min(bright_pct/2, 2) + min(dark_pct/1.5, 2.5) +
             min(th.mean()/15, 2) + min(th.std()/5, 1.5))
    idx = 0 if score < 1 else 1 if score < 2.5 else 2 if score < 4 else 3 if score < 6 else 4
    s   = DR_STAGES[idx]
    conf = round(min(100, 50 + (score/8)*50 if idx > 0 else 50 + ((1-score)/1)*50), 1)
    return s[2], s[3], s[4], s[5], round(dark_pct, 3), round(bright_pct, 3), conf, bm, dm, th


def confidence_score(pred_map, thresh=0.5):
    lp = pred_map[pred_map > thresh]
    return round(float(lp.mean()) * 100, 2) if len(lp) > 0 \
           else round(float((1 - pred_map).mean()) * 100, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — DIAGNOSIS RESULT CARD
# Lesion Segmentation panel + DR Severity Classification + Diagnosis Results
# ═══════════════════════════════════════════════════════════════════════════════
def make_overlay(img, pred_map):
    ov = img.copy()
    pb = (pred_map > 0.5).astype(np.uint8)
    if pb.max() > 0:
        ov[pb == 1] = np.clip(
            ov[pb == 1].astype(int) * 0.35 + np.array([255, 50, 50]) * 0.65, 0, 255
        ).astype(np.uint8)
    return ov

def make_heatmap(pred_map):
    hm = cv2.applyColorMap((pred_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)

def _card(ax, label, value, val_color, bg='#161b22', border='#30363d', bw=1.2):
    ax.set_facecolor(bg); ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(bw)
    ax.text(0.08, 0.75, label, transform=ax.transAxes, fontsize=7.5,
            color='#8b949e', va='center', fontfamily='monospace', fontweight='bold')
    ax.text(0.08, 0.28, value, transform=ax.transAxes, fontsize=15,
            color=val_color, va='center', fontfamily='monospace', fontweight='bold')

def _img_panel(ax, im, title, cmap=None, border='#30363d'):
    ax.set_facecolor('#161b22')
    ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
    ax.set_title(title, color='#8b949e', fontsize=8.5, pad=4, fontfamily='monospace')
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(1.2)


def save_result(fundus_data, oct_data, metrics, out_name):
    OUT_DIR.mkdir(exist_ok=True)
    BG    = '#0d1117'
    has_f = fundus_data is not None
    has_o = oct_data    is not None

    # Final diagnosis — fundus takes priority when both present
    if has_f:
        _, _, _, stage, color, short, desc, burden, n_les, conf, lesion_types = fundus_data
    else:
        _, _, _, _, stage, color, short, desc, fluid, drusen, conf, bm, dm, th = oct_data

    # Layout: image rows + diagnosis row + model metrics row
    row_heights = []
    if has_f: row_heights.append(3.5)
    if has_o: row_heights.append(3.5)
    row_heights += [1.1, 1.0]   # diagnosis + model metrics

    fig = plt.figure(figsize=(22, sum(row_heights) * 1.55 + 1.5), facecolor=BG)
    gs  = fig.add_gridspec(len(row_heights), 4,
                           height_ratios=row_heights,
                           hspace=0.12, wspace=0.06,
                           left=0.02, right=0.98,
                           top=0.91,  bottom=0.02)
    cur = 0

    # ── FUNDUS ROW: Input | Preprocessed | U-Net Segmentation | Lesion Overlay ──
    if has_f:
        _, prep, pred, _, col_f, _, _, burden, n_les, conf, lesion_types = fundus_data
        overlay = make_overlay(prep, pred)
        for c, (im, title, cmap) in enumerate([
            (fundus_data[0], 'Fundus Input (IDRiD)',    None),
            (prep,           'Preprocessed — CLAHE',    None),
            (pred,           'U-Net Segmentation',      'hot'),
            (overlay,        'Lesion Overlay',          None),
        ]):
            _img_panel(fig.add_subplot(gs[cur, c]), im, title, cmap,
                       border=col_f if c == 2 else '#30363d')
        cur += 1

    # ── OCT ROW: Input | Drusen | Fluid | Combined ────────────────────────────
    if has_o:
        _, orig_r, gray_r, _, _, _, _, _, fluid, drusen, conf_o, bm, dm, _ = oct_data
        ov_bright = gray_r.copy(); ov_bright[bm > 0] = [255, 220, 50]
        ov_dark   = gray_r.copy(); ov_dark  [dm > 0] = [50,  150, 255]
        combined  = gray_r.copy()
        combined[bm > 0] = [255, 220, 50]; combined[dm > 0] = [50, 150, 255]
        for c, (im, title) in enumerate([
            (orig_r,    'OCT Input'),
            (ov_bright, 'Drusen / Exudates'),
            (ov_dark,   'Fluid Regions'),
            (combined,  'Combined Lesion Map'),
        ]):
            _img_panel(fig.add_subplot(gs[cur, c]), im, title)
        cur += 1

    # ── DIAGNOSIS ROW ─────────────────────────────────────────────────────────
    # Col 0: Diagnosis badge
    ax_d = fig.add_subplot(gs[cur, 0])
    ax_d.set_facecolor('#161b22'); ax_d.axis('off')
    for sp in ax_d.spines.values():
        sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(2.5)
    ax_d.text(0.5, 0.80, 'DR SEVERITY CLASSIFICATION',
              transform=ax_d.transAxes, fontsize=7, color='#8b949e',
              ha='center', fontfamily='monospace', fontweight='bold')
    ax_d.text(0.5, 0.35, short,
              transform=ax_d.transAxes, fontsize=20, color=color,
              ha='center', fontfamily='monospace', fontweight='bold')

    # Cols 1-3: prediction metrics
    if has_f:
        lesion_str = ', '.join(lesion_types) if lesion_types else 'None detected'
        pred_cards = [
            ('LESION SEGMENTATION', lesion_str[:28], '#58a6ff'),
            ('LESION BURDEN',       f"{burden:.3f}%", '#58a6ff'),
            ('CONFIDENCE',          f"{conf:.1f}%",   '#3fb950'),
        ]
    else:
        pred_cards = [
            ('FLUID AREA',  f"{fluid:.3f}%",  '#58a6ff'),
            ('DRUSEN AREA', f"{drusen:.3f}%", '#f1c40f'),
            ('CONFIDENCE',  f"{conf_o:.1f}%", '#3fb950'),
        ]
    for c, (lbl, val, mc) in enumerate(pred_cards, start=1):
        _card(fig.add_subplot(gs[cur, c]), lbl, val, mc)
    cur += 1

    # ── MODEL METRICS ROW ─────────────────────────────────────────────────────
    acc_cards = [
        ('MODEL ACCURACY',   f"{metrics.get('accuracy',  0)*100:.2f}%", '#e3b341'),
        ('DICE COEFFICIENT', f"{metrics.get('dice',       0):.4f}",     '#e3b341'),
        ('PRECISION',        f"{metrics.get('precision',  0)*100:.2f}%",'#e3b341'),
        ('RECALL',           f"{metrics.get('recall',     0)*100:.2f}%",'#e3b341'),
    ]
    for c, (lbl, val, mc) in enumerate(acc_cards):
        ax = fig.add_subplot(gs[cur, c])
        ax.set_facecolor('#0d1f0d'); ax.axis('off')
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('#238636'); sp.set_linewidth(1.2)
        ax.text(0.08, 0.75, lbl, transform=ax.transAxes, fontsize=7.5,
                color='#8b949e', va='center', fontfamily='monospace', fontweight='bold')
        ax.text(0.08, 0.28, val, transform=ax.transAxes, fontsize=15,
                color=mc, va='center', fontfamily='monospace', fontweight='bold')

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965,
             'System Architecture for Multimodal DR Lesion Segmentation',
             ha='center', color='white', fontsize=13,
             fontweight='bold', fontfamily='monospace')
    fig.text(0.5, 0.938,
             f"Diagnosis: {stage}   |   {desc}",
             ha='center', color=color, fontsize=10, fontfamily='monospace')

    out_img = OUT_DIR / f"{out_name}_result.png"
    plt.savefig(out_img, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    return out_img


# ═══════════════════════════════════════════════════════════════════════════════
# JSON REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def save_json_report(fundus_data, oct_data, metrics, out_name):
    OUT_DIR.mkdir(exist_ok=True)
    report = {
        "timestamp":    datetime.now().isoformat(),
        "architecture": "Hybrid U-Net + Squeeze-Excitation (LightUNet, IDRiD)",
        "pipeline": [
            "Stage 1: Input — Fundus (IDRiD) + OCT",
            "Stage 2: Preprocessing — Resize, Denoise, Normalize",
            "Stage 3: CNN Feature Extraction (encoder path)",
            "Stage 4: Hybrid Model — U-Net Segmentation + Severity Classifier",
            "Stage 5: Output — Lesion Segmentation + DR Severity + Diagnosis",
        ],
        "model_metrics": metrics,
    }
    if fundus_data is not None:
        _, _, _, stage, _, short, desc, burden, n_les, conf, lesion_types = fundus_data
        report["fundus"] = {
            "diagnosis":        stage,
            "severity":         short,
            "description":      desc,
            "lesion_types":     lesion_types,
            "lesion_burden_pct": round(burden, 4),
            "lesion_count":     n_les,
            "confidence_pct":   conf,
        }
    if oct_data is not None:
        _, _, _, _, stage_o, _, short_o, desc_o, fluid, drusen, conf_o, bm, dm, th = oct_data
        report["oct"] = {
            "diagnosis":      stage_o,
            "severity":       short_o,
            "description":    desc_o,
            "fluid_area_pct":  fluid,
            "drusen_area_pct": drusen,
            "confidence_pct":  conf_o,
        }
    out_json = OUT_DIR / f"{out_name}_report.json"
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2)
    return out_json


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Full Pipeline Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="System Architecture for Multimodal DR Lesion Segmentation")
    ap.add_argument('--fundus', type=str, default=None, help='Path to fundus image (IDRiD)')
    ap.add_argument('--oct',    type=str, default=None, help='Path to OCT image')
    args = ap.parse_args()

    if not args.fundus and not args.oct:
        ap.print_help(); sys.exit(1)

    parts    = []
    if args.fundus: parts.append(Path(args.fundus).stem)
    if args.oct:    parts.append(Path(args.oct).stem)
    out_name = '_'.join(parts) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f"\n{'='*62}")
    print("  System Architecture — Multimodal DR Lesion Segmentation")
    print(f"{'='*62}")

    # ── Stage 4 setup: load Hybrid Model ─────────────────────────────────────
    print("\n[Stage 4] Loading Hybrid Deep Learning Model...")
    model   = load_model()
    metrics = load_metrics()
    print(f"          Device   : {DEVICE}")
    print(f"          Accuracy : {metrics.get('accuracy',0)*100:.2f}%  "
          f"| Dice: {metrics.get('dice',0):.4f}")

    fundus_data = None
    oct_data    = None

    # ── Stage 1→2→3→4: Fundus pipeline ───────────────────────────────────────
    if args.fundus:
        print(f"\n[Stage 1] Fundus input  : {args.fundus}")
        print( "[Stage 2] Preprocessing : Resize → Denoise → Normalize...")
        orig, prep = preprocess_fundus(args.fundus)

        print( "[Stage 3] CNN Feature Extraction → encoder path...")
        print( "[Stage 4] Hybrid U-Net Segmentation (sliding window)...")
        pred = cnn_extract_and_segment(model, prep)

        print( "[Stage 4] Severity Classifier...")
        stage, color, short, desc, burden, n_les, lesion_types = classify_severity_fundus(pred)
        conf = confidence_score(pred)
        fundus_data = (orig, prep, pred, stage, color, short, desc, burden, n_les, conf, lesion_types)

        print(f"\n          Diagnosis      : {stage}")
        print(f"          Lesion Types   : {', '.join(lesion_types) if lesion_types else 'None'}")
        print(f"          Lesion Burden  : {burden:.3f}%  | Count: {n_les}  | Confidence: {conf:.1f}%")

    # ── Stage 1→2→4: OCT pipeline ────────────────────────────────────────────
    if args.oct:
        print(f"\n[Stage 1] OCT input     : {args.oct}")
        print( "[Stage 2] Preprocessing : Denoise → CLAHE Normalize → Resize...")
        orig_r, gray_r, enh = preprocess_oct(args.oct)
        enh_gray = cv2.cvtColor(enh, cv2.COLOR_RGB2GRAY) if enh.ndim == 3 else enh

        print( "[Stage 4] Severity Classifier — OCT biomarker analysis...")
        stage_o, color_o, short_o, desc_o, fluid, drusen, conf_o, bm, dm, th = \
            classify_severity_oct(enh_gray)
        oct_data = (args.oct, orig_r, gray_r, enh_gray,
                    stage_o, color_o, short_o, desc_o,
                    fluid, drusen, conf_o, bm, dm, th)

        print(f"\n          Diagnosis      : {stage_o}")
        print(f"          Fluid Area     : {fluid:.3f}%  | Drusen: {drusen:.3f}%  | Confidence: {conf_o:.1f}%")

    # ── Stage 5: Diagnosis Results ────────────────────────────────────────────
    print(f"\n[Stage 5] Generating Diagnosis Results...")
    out_img  = save_result(fundus_data, oct_data, metrics, out_name)
    out_json = save_json_report(fundus_data, oct_data, metrics, out_name)

    print(f"\n{'='*62}")
    print(f"  Result image : {out_img}")
    print(f"  JSON report  : {out_json}")
    print(f"{'='*62}\n")


if __name__ == '__main__':
    main()
