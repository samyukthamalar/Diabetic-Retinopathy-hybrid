"""
Multimodal DR Lesion Segmentation — Gradio Web UI
Wraps the existing predict.py pipeline with an image upload interface.

Run: python src/app.py
"""

import sys, json, tempfile, warnings
from pathlib import Path
import numpy as np
import cv2
import gradio as gr

warnings.filterwarnings('ignore')

# Make sure src/ is on path so we can import predict
sys.path.insert(0, str(Path(__file__).parent))

from predict import (
    load_model, load_metrics,
    preprocess_fundus, preprocess_oct,
    cnn_extract_and_segment,
    classify_severity_fundus, classify_severity_oct,
    confidence_score, make_overlay, make_heatmap,
    save_result, save_json_report,
    OUT_DIR,
)
from datetime import datetime

# Load once at startup
print("Loading model...")
MODEL   = load_model()
METRICS = load_metrics()
print("Model ready.")


# ── Core inference functions ──────────────────────────────────────────────────
def run_fundus(pil_img):
    """Takes a PIL image from Gradio, runs full fundus pipeline."""
    # Save to temp file so preprocess_fundus can read it
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        pil_img.save(f.name)
        tmp = f.name

    orig, prep = preprocess_fundus(tmp)
    pred       = cnn_extract_and_segment(MODEL, prep)
    stage, color, short, desc, burden, n_les, lesion_types = classify_severity_fundus(pred)
    conf       = confidence_score(pred)
    overlay    = make_overlay(prep, pred)
    heatmap    = make_heatmap(pred)

    return (orig, prep, pred, stage, color, short, desc,
            burden, n_les, conf, lesion_types), overlay, heatmap, pred


def run_oct(pil_img):
    """Takes a PIL image from Gradio, runs full OCT pipeline."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        pil_img.save(f.name)
        tmp = f.name

    orig_r, gray_r, enh = preprocess_oct(tmp)
    enh_gray = cv2.cvtColor(enh, cv2.COLOR_RGB2GRAY) if enh.ndim == 3 else enh
    stage_o, color_o, short_o, desc_o, fluid, drusen, conf_o, bm, dm, th = \
        classify_severity_oct(enh_gray)

    ov_bright = gray_r.copy(); ov_bright[bm > 0] = [255, 220, 50]
    ov_dark   = gray_r.copy(); ov_dark  [dm > 0] = [50,  150, 255]
    combined  = gray_r.copy()
    combined[bm > 0] = [255, 220, 50]; combined[dm > 0] = [50, 150, 255]

    return (tmp, orig_r, gray_r, enh_gray, stage_o, color_o, short_o,
            desc_o, fluid, drusen, conf_o, bm, dm, th), combined


# ── Gradio handler functions ──────────────────────────────────────────────────
def predict_fundus(fundus_img):
    if fundus_img is None:
        return None, None, None, "⚠️ Please upload a fundus image."

    fundus_data, overlay, heatmap, pred = run_fundus(fundus_img)
    _, prep, _, stage, color, short, desc, burden, n_les, conf, lesion_types = fundus_data

    # Save result card + JSON
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"fundus_{ts}"
    save_result(fundus_data, None, METRICS, out_name)
    save_json_report(fundus_data, None, METRICS, out_name)

    lesion_str = ', '.join(lesion_types) if lesion_types else 'None detected'
    summary = (
        f"### {short} — {stage}\n\n"
        f"**{desc}**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Lesion Types | {lesion_str} |\n"
        f"| Lesion Burden | {burden:.3f}% |\n"
        f"| Lesions Found | {n_les} |\n"
        f"| Confidence | {conf:.1f}% |\n"
        f"| Model Accuracy | {METRICS.get('accuracy',0)*100:.2f}% |\n"
        f"| Dice Score | {METRICS.get('dice',0):.4f} |\n"
        f"| Precision | {METRICS.get('precision',0)*100:.2f}% |\n"
        f"| Recall | {METRICS.get('recall',0)*100:.2f}% |"
    )
    return overlay, heatmap, prep, summary


def predict_oct(oct_img):
    if oct_img is None:
        return None, None, "⚠️ Please upload an OCT image."

    oct_data, combined = run_oct(oct_img)
    _, orig_r, gray_r, _, stage_o, _, short_o, desc_o, fluid, drusen, conf_o, bm, dm, _ = oct_data

    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"oct_{ts}"
    save_result(None, oct_data, METRICS, out_name)
    save_json_report(None, oct_data, METRICS, out_name)

    summary = (
        f"### {short_o} — {stage_o}\n\n"
        f"**{desc_o}**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Fluid Area | {fluid:.3f}% |\n"
        f"| Drusen / Exudates | {drusen:.3f}% |\n"
        f"| Confidence | {conf_o:.1f}% |\n"
        f"| Model Accuracy | {METRICS.get('accuracy',0)*100:.2f}% |\n"
        f"| Dice Score | {METRICS.get('dice',0):.4f} |"
    )
    return orig_r, combined, summary


DR_STAGE_ORDER = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

def fuse_diagnosis(fundus_data, oct_data):
    """
    Merge fundus + OCT into a single unified diagnosis.
    Takes the more severe stage from both modalities.
    Returns: stage, color, short, desc, summary_dict
    """
    from predict import DR_STAGES

    f_stage = fundus_data[3] if fundus_data else None
    o_stage = oct_data[4]    if oct_data    else None

    # Pick the worse (higher index) stage
    f_idx = DR_STAGE_ORDER.index(f_stage) if f_stage in DR_STAGE_ORDER else -1
    o_idx = DR_STAGE_ORDER.index(o_stage) if o_stage in DR_STAGE_ORDER else -1
    chosen_idx = max(f_idx, o_idx)
    s = DR_STAGES[chosen_idx]
    stage, color, short, desc = s[2], s[3], s[4], s[5]

    summary = {"stage": stage, "color": color, "short": short, "desc": desc}
    if fundus_data:
        _, _, _, _, _, _, _, burden, n_les, conf, lesion_types = fundus_data
        summary["lesion_types"]  = lesion_types
        summary["lesion_burden"] = burden
        summary["lesion_count"]  = n_les
        summary["fundus_conf"]   = conf
    if oct_data:
        _, _, _, _, _, _, _, _, fluid, drusen, conf_o, *_ = oct_data
        summary["fluid"]     = fluid
        summary["drusen"]    = drusen
        summary["oct_conf"]  = conf_o

    return stage, color, short, desc, summary


def predict_both(fundus_img, oct_img):
    if fundus_img is None or oct_img is None:
        return None, "⚠️ Please upload BOTH a fundus image and an OCT image."

    fundus_data, overlay, heatmap, pred = run_fundus(fundus_img)
    oct_data, oct_combined              = run_oct(oct_img)

    # Fuse into single diagnosis
    stage, color, short, desc, info = fuse_diagnosis(fundus_data, oct_data)

    # Save unified result card (both rows) + JSON
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"multimodal_{ts}"
    result_path = save_result(fundus_data, oct_data, METRICS, out_name)
    save_json_report(fundus_data, oct_data, METRICS, out_name)

    # Build single unified diagnosis text
    lesion_str = ', '.join(info.get('lesion_types', [])) or 'None detected'
    summary = (
        f"## {short} — {stage}\n\n"
        f"**{desc}**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Lesion Types (Fundus) | {lesion_str} |\n"
        f"| Lesion Burden | {info.get('lesion_burden', 0):.3f}% |\n"
        f"| Lesions Found | {info.get('lesion_count', 0)} |\n"
        f"| Fundus Confidence | {info.get('fundus_conf', 0):.1f}% |\n"
        f"| Fluid Area (OCT) | {info.get('fluid', 0):.3f}% |\n"
        f"| Drusen Area (OCT) | {info.get('drusen', 0):.3f}% |\n"
        f"| OCT Confidence | {info.get('oct_conf', 0):.1f}% |\n"
        f"| Model Accuracy | {METRICS.get('accuracy',0)*100:.2f}% |\n"
        f"| Dice Score | {METRICS.get('dice',0):.4f} |\n"
        f"| Precision | {METRICS.get('precision',0)*100:.2f}% |\n"
        f"| Recall | {METRICS.get('recall',0)*100:.2f}% |"
    )

    # Return the saved result card image as the single output
    result_img = cv2.imread(str(result_path))
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img, summary


# ── UI Layout ─────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#0d1117",
    body_text_color="#c9d1d9",
    block_background_fill="#161b22",
    block_border_color="#30363d",
    block_title_text_color="#8b949e",
    input_background_fill="#0d1117",
    button_primary_background_fill="#1f6feb",
    button_primary_text_color="white",
)

with gr.Blocks(theme=THEME, title="DR Lesion Segmentation") as demo:

    gr.Markdown(
        """
        # System Architecture for Multimodal DR Lesion Segmentation
        **Diabetic Retinopathy Lesion Segmentation using Deep Learning — Hybrid U-Net + EfficientNet (SE)**
        > Pipeline: Input → Preprocessing → CNN Feature Extraction → Hybrid U-Net → Severity Classification → Diagnosis
        """
    )

    with gr.Tabs():

        # ── Tab 1: Fundus ─────────────────────────────────────────────────────
        with gr.TabItem("🔬 Fundus Image (IDRiD)"):
            gr.Markdown("Upload a **fundus retinal image**. The model will segment lesions and classify DR severity.")
            with gr.Row():
                with gr.Column(scale=1):
                    f_input   = gr.Image(type="pil", label="Upload Fundus Image", height=300)
                    f_btn     = gr.Button("Run Analysis", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        f_prep    = gr.Image(label="Preprocessed (CLAHE)", height=220)
                        f_heatmap = gr.Image(label="U-Net Segmentation Map", height=220)
                        f_overlay = gr.Image(label="Lesion Overlay", height=220)
                    f_result  = gr.Markdown(label="Diagnosis")

            f_btn.click(
                fn=predict_fundus,
                inputs=[f_input],
                outputs=[f_overlay, f_heatmap, f_prep, f_result],
            )

        # ── Tab 2: OCT ────────────────────────────────────────────────────────
        with gr.TabItem("🩺 OCT Image"):
            gr.Markdown("Upload an **OCT scan**. Fluid regions and drusen/exudates will be detected.")
            with gr.Row():
                with gr.Column(scale=1):
                    o_input   = gr.Image(type="pil", label="Upload OCT Image", height=300)
                    o_btn     = gr.Button("Run Analysis", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        o_orig    = gr.Image(label="OCT Input", height=220)
                        o_combined= gr.Image(label="Lesion Map (Fluid=blue, Drusen=yellow)", height=220)
                    o_result  = gr.Markdown(label="Diagnosis")

            o_btn.click(
                fn=predict_oct,
                inputs=[o_input],
                outputs=[o_orig, o_combined, o_result],
            )

        # ── Tab 3: Multimodal ─────────────────────────────────────────────────
        with gr.TabItem("🔭 Multimodal (Fundus + OCT)"):
            gr.Markdown("Upload **both** a fundus image and an OCT scan. The model fuses both modalities into a **single unified diagnosis**.")
            with gr.Row():
                with gr.Column(scale=1):
                    m_fundus = gr.Image(type="pil", label="Fundus Image (IDRiD)", height=260)
                    m_oct    = gr.Image(type="pil", label="OCT Image",     height=240)
                    m_btn     = gr.Button("Run Multimodal Analysis", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        m_overlay  = gr.Image(label="Fundus Lesion Overlay", height=200)
                        m_heatmap  = gr.Image(label="Segmentation Map",      height=200)
                    with gr.Row():
                        m_oct_map  = gr.Image(label="OCT Lesion Map",        height=200)
                        m_prep     = gr.Image(label="Preprocessed Fundus",   height=200)
                    m_result   = gr.Markdown(label="Diagnosis")

            m_btn.click(
                fn=predict_both,
                inputs=[m_fundus, m_oct],
                outputs=[m_overlay, m_heatmap, m_oct_map, m_prep, m_result],
            )

    gr.Markdown(
        """
        ---
        **Model**: LightUNet + Squeeze-Excitation &nbsp;|&nbsp;
        **Dataset**: IDRiD Segmentation &nbsp;|&nbsp;
        **Lesion Types**: Microaneurysms · Hemorrhages · Exudates
        """
    )


if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
