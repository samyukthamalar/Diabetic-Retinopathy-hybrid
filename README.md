# Diabetic Retinopathy Lesion Segmentation using Deep Learning
### with Fundus and OCT Images — Hybrid U-Net + Squeeze-Excitation Model

---

## 🧒 What is this project? (Simple explanation)

Imagine your eye is like a camera. At the back of your eye there is a special layer called the **retina** — it helps you see. When someone has **Diabetic Retinopathy (DR)**, tiny blood vessels in the retina start to leak or break. This causes small spots and patches to appear on the retina. If not caught early, it can make a person go blind.

Doctors use two types of eye scans to check for this:
- **Fundus image** — a colour photo of the back of the eye (looks orange/red)
- **OCT image** — a cross-section scan of the retina layers (looks like a black and white slice)

This project uses **Artificial Intelligence (AI)** to look at these images and automatically find the damaged spots (lesions) and tell the doctor how serious the condition is — from **No DR** all the way to **Proliferative DR (PDR)**.

Think of it like a smart robot doctor that looks at eye photos and says: *"This person has Moderate NPDR — there are hemorrhages and exudates detected."*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│         System Architecture for Multimodal DR Lesion           │
│                      Segmentation                               │
└─────────────────────────────────────────────────────────────────┘

  STAGE 1 │ INPUT
  ────────┤  • Fundus Image  (IDRiD Dataset — colour retinal photo)
          │  • OCT Image     (cross-section retinal scan)
          │
  STAGE 2 │ IMAGE PREPROCESSING
  ────────┤  • Resize        → crop black borders, scale to 512×512
          │  • Denoise       → remove noise using NL-Means filter
          │  • Normalize     → CLAHE contrast enhancement on L channel
          │  • Augment       → flip, rotate, brightness (training only)
          │
  STAGE 3 │ CNN FEATURE EXTRACTION
  ────────┤  • 4-level encoder: 3→16→32→64→128 channels
          │  • MaxPool2d between each level
          │  • Learns edges, textures, lesion patterns
          │
  STAGE 4 │ HYBRID DEEP LEARNING MODEL
  ────────┤  ┌─────────────────────────────────────┐
          │  │  Bottleneck + Squeeze-Excitation     │
          │  │  (channel attention — focuses on     │
          │  │   the most important features)       │
          │  └──────────────┬──────────────────────┘
          │                 │
          │  ┌──────────────▼──────────────────────┐
          │  │  U-Net Decoder (Segmentation)        │
          │  │  128→64→32→16→1 channel map          │
          │  │  Skip connections from encoder       │
          │  └──────────────┬──────────────────────┘
          │                 │
          │  ┌──────────────▼──────────────────────┐
          │  │  Severity Classifier                 │
          │  │  Lesion burden % + connected         │
          │  │  component count → DR stage          │
          │  └─────────────────────────────────────┘
          │
  STAGE 5 │ DIAGNOSIS RESULTS
  ────────┤  • Lesion Segmentation map
          │    (Microaneurysms · Hemorrhages · Exudates)
          │  • DR Severity: No DR / Mild / Moderate / Severe / PDR
          │  • Confidence score
          │  • Result card image + JSON report saved to outputs/
```

---

## 📊 DR Severity Stages

| Stage | Colour | What it means |
|---|---|---|
| No DR | 🟢 Green | Healthy eye, no lesions found |
| Mild NPDR | 🟡 Yellow | Tiny microaneurysms (small dots) present |
| Moderate NPDR | 🟠 Orange | Hemorrhages and hard exudates visible |
| Severe NPDR | 🔴 Red | Extensive lesions, urgent referral needed |
| Proliferative DR | 🟣 Purple | New blood vessels growing, immediate treatment required |

---

## 📁 Project Structure

```
project/
│
├── src/
│   ├── predict.py     ← Main inference pipeline (command line)
│   ├── app.py         ← Gradio web UI (browser interface)
│   ├── train.py       ← Model training script
│   └── prepare.py     ← Dataset preparation from IDRiD zip
│
├── data/
│   ├── train/         ← 43 training images + masks
│   ├── val/           ← 11 validation images + masks
│   └── test/          ← 27 test images + masks (official IDRiD)
│
├── models/
│   ├── best_model.pth     ← Trained model weights
│   └── final_report.json  ← Training metrics
│
├── outputs/           ← Prediction results saved here
├── samples/
│   └── sample_oct.jpg ← Sample OCT image for testing
│
└── requirements.txt   ← All Python dependencies
```

---

## ⚙️ Setup — Step by Step

### Step 1 — Make sure you have Python installed

Open your terminal (Command Prompt or PowerShell) and check:

```bash
python --version
```

You need Python 3.8 or higher. If you don't have it, download from [python.org](https://www.python.org/downloads/).

---

### Step 2 — Install all dependencies

In the project folder, run:

```bash
pip install -r requirements.txt
```

This installs PyTorch, OpenCV, Gradio, and everything else the project needs. It may take a few minutes.

---

### Step 3 — Prepare the dataset (only needed if you want to retrain)

You need the IDRiD Segmentation zip file: `A. Segmentation.zip`

Place it at `C:\Users\<YourName>\Downloads\A. Segmentation.zip`, then run:

```bash
python src/prepare.py
```

This extracts the zip, combines lesion masks (MA + HE + EX + SE), applies CLAHE preprocessing, and saves everything to the `data/` folder.

---

### Step 4 — Train the model (only needed if you want to retrain)

```bash
python src/train.py
```

This trains the Hybrid U-Net for 80 epochs on patch-based IDRiD data. The best model is saved to `models/best_model.pth`. Takes ~30–60 minutes on CPU.

> The trained model is already included in `models/best_model.pth` — you can skip Steps 3 and 4 and go straight to running predictions.

---

## 🚀 How to Run

### Option A — Web UI (Recommended, easiest)

```bash
python src/app.py
```

Then open your browser and go to:

```
http://localhost:7860
```

You will see a web interface with 3 tabs:

| Tab | What to do |
|---|---|
| 🔬 Fundus Image | Upload a colour retinal photo → click Run Analysis |
| 🩺 OCT Image | Upload an OCT scan → click Run Analysis |
| 🔭 Multimodal | Upload BOTH images → click Run Multimodal Analysis |

The result shows the segmentation map, lesion overlay, and diagnosis.

---

### Option B — Command Line

**Fundus image only:**
```bash
python src/predict.py --fundus "path/to/your/fundus.jpg"
```

**OCT image only:**
```bash
python src/predict.py --oct "path/to/your/oct.jpg"
```

**Both images together (multimodal):**
```bash
python src/predict.py --fundus "path/to/fundus.jpg" --oct "path/to/oct.jpg"
```

Results are saved to the `outputs/` folder as:
- `*_result.png` — visual result card (dark theme)
- `*_report.json` — full diagnosis report in JSON format

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | Lightweight U-Net + Squeeze-Excitation (SE) |
| Input size | 256×256 patches (sliding window over 512×512) |
| Encoder channels | 3 → 16 → 32 → 64 → 128 → 256 |
| Bottleneck | SE attention (fc1: 256→16, fc2: 16→256) |
| Decoder | 4 upsampling stages with skip connections |
| Parameters | ~1.95 million |
| Loss function | Weighted Focal Loss (40%) + Dice Loss (60%) |
| Positive weight | 43.0 (handles 97.7% background imbalance) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (80 epochs) |
| Dataset | IDRiD Segmentation — 54 train + 27 test images |
| Train/Val split | 43 train / 11 val (80/20, seed=42) |

---

## 📈 Model Performance

Evaluated on the official IDRiD test set (27 full images, sliding window inference):

| Metric | Score |
|---|---|
| Accuracy | 78.97% |
| Dice Coefficient | 0.1149 |
| IoU | 0.0626 |
| Precision | 6.54% |
| Recall | 59.21% |
| F1 Score | 0.1149 |

> Note: Low Dice/Precision is expected — IDRiD lesions (especially microaneurysms) occupy less than 2.3% of pixels. The model is tuned for high recall to avoid missing lesions, which is the clinically safer choice.

---

## 🔬 Lesion Types Detected

| Lesion | Source | Description |
|---|---|---|
| Microaneurysms (MA) | Fundus | Tiny red dots — earliest sign of DR |
| Hemorrhages (HE) | Fundus | Larger red blotches from burst vessels |
| Hard Exudates (EX) | Fundus | Bright yellow-white deposits |
| Soft Exudates (SE) | Fundus | Cotton-wool spots (nerve fibre infarcts) |
| Fluid regions | OCT | Dark areas — intraretinal/subretinal fluid |
| Drusen / Exudates | OCT | Bright deposits under the retina |

---

## 🖼️ Output Example

After running a prediction, the result card shows:

```
┌──────────────────────────────────────────────────────────┐
│  System Architecture for Multimodal DR Lesion Segmentation│
│  Diagnosis: Mild NPDR  |  Mild NPDR — Microaneurysms...  │
├────────────┬────────────┬────────────┬────────────────────┤
│ Fundus     │ Preprocessed│ U-Net Seg │ Lesion Overlay     │
│ Input      │ (CLAHE)    │ Map (hot) │ (red highlights)   │
├────────────┴────────────┴────────────┴────────────────────┤
│ OCT Input  │ Drusen     │ Fluid      │ Combined Map       │
│            │ (yellow)   │ (blue)     │                    │
├────────────┬────────────┬────────────┬────────────────────┤
│ DIAGNOSIS  │ LESION     │ LESION     │ CONFIDENCE         │
│ MILD       │ BURDEN     │ FOUND      │ 56.8%              │
│            │ 5.488%     │ 7          │                    │
├────────────┼────────────┼────────────┼────────────────────┤
│ ACCURACY   │ DICE       │ PRECISION  │ RECALL             │
│ 78.97%     │ 0.1149     │ 6.54%      │ 59.21%             │
└────────────┴────────────┴────────────┴────────────────────┘
```

---

## ❓ Troubleshooting

**"Cannot read image" error**
→ Make sure the file path is correct and the image is a `.jpg` or `.png` file.

**"Model not found" error**
→ Make sure `models/best_model.pth` exists. It should already be there.

**Browser shows "site can't be reached"**
→ Make sure you are opening `http://localhost:7860` (not `0.0.0.0`).
→ Make sure `python src/app.py` is still running in your terminal.

**Slow on CPU**
→ The sliding window inference over a 512×512 image takes ~10–20 seconds on CPU. This is normal. If you have an NVIDIA GPU, PyTorch will use it automatically.

**pip install fails**
→ Try upgrading pip first: `python -m pip install --upgrade pip`

---

## 📜 Dataset Credit

This project uses the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**:

> Porwal P. et al., "IDRiD: Diabetic Retinopathy — Segmentation and Grading Challenge",
> Medical Image Analysis, 2020. [https://idrid.grand-challenge.org](https://idrid.grand-challenge.org)

---

## 👨‍💻 Project Title

**Diabetic Retinopathy Lesion Segmentation using Deep Learning with Fundus and OCT Images by Implementing a Hybrid Model**
