import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ─────────────────────────────────────────
# 1. Load Model & Class Names
# ─────────────────────────────────────────
model = tf.keras.models.load_model("best_model_phase2.keras")

with open("class_names.json") as f:
    class_names = json.load(f)

# ─────────────────────────────────────────
# 2. Disease Information
# ─────────────────────────────────────────
DISEASE_INFO = {
    "Pepper bell Bacterial spot": {"desc": "Bacterial disease causing dark, water-soaked spots.", "treatment": "Apply copper-based bactericides.", "severity": "Medium"},
    "Pepper bell healthy": {"desc": "Your plant looks healthy!", "treatment": "Maintain watering and sunlight.", "severity": "None"},
    "Potato Early blight": {"desc": "Fungal disease causing brown concentric ring spots.", "treatment": "Apply fungicides like chlorothalonil.", "severity": "Medium"},
    "Potato Late blight": {"desc": "Severe fungal disease capable of destroying crops.", "treatment": "Apply systemic fungicides immediately.", "severity": "High"},
    "Potato healthy": {"desc": "The plant shows no visible disease symptoms.", "treatment": "Continue good care practices.", "severity": "None"},
    "Tomato Bacterial spot": {"desc": "Small dark water-soaked spots on leaves.", "treatment": "Apply copper sprays.", "severity": "Medium"},
    "Tomato Early blight": {"desc": "Brown spots with target-like rings.", "treatment": "Remove infected leaves.", "severity": "Medium"},
    "Tomato Late blight": {"desc": "Dark lesions with white mold underneath.", "treatment": "Apply fungicide immediately.", "severity": "High"},
    "Tomato Leaf Mold": {"desc": "Yellow patches with olive mold underneath.", "treatment": "Reduce humidity.", "severity": "Medium"},
    "Tomato Septoria leaf spot": {"desc": "Small circular spots with dark edges.", "treatment": "Remove infected leaves.", "severity": "Medium"},
    "Tomato Spider mites Two spotted spider mite": {"desc": "Tiny mites causing yellowing and webbing.", "treatment": "Apply neem oil.", "severity": "Medium"},
    "Tomato Target Spot": {"desc": "Fungal disease causing concentric rings.", "treatment": "Improve airflow.", "severity": "Medium"},
    "Tomato Tomato YellowLeaf Curl Virus": {"desc": "Viral infection causing leaf curling.", "treatment": "Control whiteflies.", "severity": "High"},
    "Tomato Tomato mosaic virus": {"desc": "Mosaic pattern and leaf distortion.", "treatment": "Remove infected plants.", "severity": "High"},
    "Tomato healthy": {"desc": "The plant appears healthy.", "treatment": "Maintain sunlight.", "severity": "None"},
}

SEVERITY_COLOR = {"None": "#16a34a", "Medium": "#f59e0b", "High": "#ef4444"}

# ─────────────────────────────────────────
# 3. Prediction Function
# ─────────────────────────────────────────
def predict(img):
    if img is None:
        return "<div class='diagnosis-card' style='display:flex; align-items:center; justify-content:center; opacity:0.6;'>Upload a leaf image to start</div>"

    img_res = Image.fromarray(img).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img_res) / 255.0, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    top3_idx = preds.argsort()[-3:][::-1]

    top_name = class_names[top3_idx[0]].replace("_", " ")
    confidence = preds[top3_idx[0]] * 100
    info = DISEASE_INFO.get(top_name, {"desc": "N/A", "treatment": "N/A", "severity": "Medium"})
    color = SEVERITY_COLOR[info["severity"]]

    bars = ""
    for idx in top3_idx:
        name = class_names[idx].replace("_", " ")
        conf = preds[idx] * 100
        bars += f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:2px;">
                <span>{name}</span><b>{conf:.1f}%</b>
            </div>
            <div style="background:var(--border-color-primary);height:6px;border-radius:10px;">
                <div style="width:{conf:.1f}%;background:#16a34a;height:6px;border-radius:10px;"></div>
            </div>
        </div>"""

    return f"""
    <div class="diagnosis-card">
        <h3 style="margin:0 0 5px 0;font-size:11px;text-transform:uppercase;opacity:0.6;">Diagnosis</h3>
        <div style="font-size:22px;font-weight:800;margin-bottom:5px;">{top_name}</div>
        <div style="font-size:14px;margin-bottom:12px;opacity:0.8;">Confidence: <b>{confidence:.1f}%</b></div>
        <div style="background:{color}33;color:{color};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:bold;margin-bottom:15px;display:inline-block;">
            Severity: {info["severity"]}
        </div>
        <hr style="opacity:0.1; margin:15px 0;">
        <h4 style="color:#16a34a; font-size:14px; margin:0 0 5px 0;">Description</h4>
        <p style="font-size:13px; line-height:1.4; opacity:0.9; margin-bottom:15px;">{info["desc"]}</p>
        <h4 style="color:#16a34a; font-size:14px; margin:0 0 5px 0;">Treatment</h4>
        <p style="font-size:13px; line-height:1.4; opacity:0.9; margin-bottom:15px;">{info["treatment"]}</p>
        <hr style="opacity:0.1; margin:15px 0;">
        <h4 style="font-size:11px; opacity:0.6; margin-bottom:8px;">Top Predictions</h4>
        {bars}
    </div>"""

# ─────────────────────────────────────────
# 4. Custom CSS (Strict Fix for Scrollbars)
# ─────────────────────────────────────────
custom_css = """
/* Sembunyikan semua scrollbar melayang yang muncul di HuggingFace/Gradio */
::-webkit-scrollbar { display: none !important; }
* { -ms-overflow-style: none !important; scrollbar-width: none !important; }
.diagnosis-card {
    background: var(--block-background-fill);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid var(--border-color-primary);
    color: var(--body-text-color);
    min-height: 410px;
}
/* Hilangkan padding ekstra yang sering memicu scrollbar melayang */
.gradio-container { padding: 0 !important; }
"""

# ─────────────────────────────────────────
# 5. UI Setup
# ─────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "system-ui"]
)

with gr.Blocks(theme=theme, css=custom_css, title="PlantWise AI") as demo:
    gr.Markdown("# 🌿 PlantWise AI\n### Deep Learning Plant Disease Detection")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Leaf Image", type="numpy", height=350)
            analyze_btn = gr.Button("Analyze Plant 🌱", variant="primary")
            
        with gr.Column(scale=1):
            result_output = gr.HTML(
                "<div class='diagnosis-card' style='display:flex; align-items:center; justify-content:center; opacity:0.5;'>Upload a leaf image to start</div>"
            )

    # ─────────────────────────────────────────
    # LOGIC FIX: Cuma jalan pas di-klik tombol
    # ─────────────────────────────────────────
    analyze_btn.click(predict, inputs=image_input, outputs=result_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
