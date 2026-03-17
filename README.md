# 🌿 PlantWise AI — Plant Disease Classifier

A deep learning web application for detecting plant diseases from leaf images, built with MobileNetV2 and deployed on Hugging Face Spaces using Gradio.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Latest-F97316?style=flat&logo=gradio&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-4ade80?style=flat)
![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-FFD21E?style=flat&logo=huggingface&logoColor=black)

🔗 **Live Demo:** [huggingface.co/spaces/unvariablehuman/plantwise-ai](https://huggingface.co/spaces/unvariablehuman/plantwise-ai)

---

## 📋 Overview

PlantWise AI classifies plant leaf images into 15 categories — healthy or diseased — using a fine-tuned MobileNetV2 model trained on the PlantVillage dataset. The app provides diagnosis, confidence scores, disease description, and treatment recommendations.

**Supported Plants:** Pepper (Bell), Potato, Tomato

---

## 🌱 Classes

| # | Class | Type |
|---|-------|------|
| 1 | Pepper Bell — Bacterial Spot | Disease |
| 2 | Pepper Bell — Healthy | Healthy |
| 3 | Potato — Early Blight | Disease |
| 4 | Potato — Late Blight | Disease |
| 5 | Potato — Healthy | Healthy |
| 6 | Tomato — Bacterial Spot | Disease |
| 7 | Tomato — Early Blight | Disease |
| 8 | Tomato — Late Blight | Disease |
| 9 | Tomato — Leaf Mold | Disease |
| 10 | Tomato — Septoria Leaf Spot | Disease |
| 11 | Tomato — Spider Mites | Disease |
| 12 | Tomato — Target Spot | Disease |
| 13 | Tomato — Yellow Leaf Curl Virus | Disease |
| 14 | Tomato — Mosaic Virus | Disease |
| 15 | Tomato — Healthy | Healthy |

---

## 🧠 Model Architecture

Transfer learning with **MobileNetV2** as the backbone.

```
Input (224×224×3)
    ↓
preprocess_input (normalization)
    ↓
MobileNetV2 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.5)
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(15, Softmax)
```

### Training Strategy

**Phase 1 — Feature Extraction**
- Base model frozen, only classification head trained
- Learning rate: `0.001`
- Best Val Accuracy: **90.33%**

**Phase 2 — Fine-Tuning**
- First 100 layers frozen, remaining 54 unfrozen
- Learning rate: `0.00001`
- Best Val Accuracy: **96.10%** ✅

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **96%** |
| Macro Avg Precision | 96% |
| Macro Avg Recall | 96% |
| Macro Avg F1-Score | 96% |

**Per-Class Performance:** 13/15 classes above 90% recall. Two lowest:
- `Tomato Early Blight`: 83% — visually similar to Target Spot
- `Tomato Spider Mites`: 87% — visually similar to other spot diseases

---

## 🚀 Run Locally

```bash
git clone https://github.com/unvariablehuman/plant-disease-classifier.git
cd plant-disease-classifier
pip install -r requirements.txt
python app.py
```

---

## 📁 Project Structure

```
plant-disease-classifier/
│
├── app.py                      # Gradio web app
├── requirements.txt            # Python dependencies
├── best_model_phase2.keras     # Trained model (96% accuracy)
├── class_names.json            # Class label mapping
└── README.md                   # This file
```

---

## 📦 Dataset

**PlantVillage Dataset**
- Total images: 20,669
- Training: 80% (16,541 images)
- Validation: 20% (4,128 images)
- Image size: 224×224 pixels

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **Base Model:** MobileNetV2 (ImageNet pretrained)
- **Web App:** Gradio
- **Deployment:** Hugging Face Spaces
- **Training:** Google Colab (GPU T4)
- **Dataset:** PlantVillage

---

## 📝 Notes

- Use clear, close-up photos of single leaves for best results
- Best with well-lit images, avoid shadows
- Accepts JPG, JPEG, PNG formats

---

*Built as a Computer Vision course final project.*
