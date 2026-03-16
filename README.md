# 🌿 Plant Disease Classifier

A deep learning web application for detecting plant diseases from leaf images, built with MobileNetV2 and deployed using Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-4ade80?style=flat)

---

## 📋 Overview

This application classifies plant leaf images into 15 categories — healthy or diseased — using a fine-tuned MobileNetV2 model trained on the PlantVillage dataset.

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
- Base model frozen
- Only classification head trained
- Learning rate: `0.001`
- Epochs: up to 50 (EarlyStopping patience=10)
- Best Val Accuracy: **90.33%**

**Phase 2 — Fine-Tuning**
- First 100 layers frozen, remaining 54 unfrozen
- Learning rate: `0.00001`
- Epochs: up to 50 (EarlyStopping patience=10)
- Best Val Accuracy: **96.10%** ✅

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **96%** |
| Macro Avg Precision | 96% |
| Macro Avg Recall | 96% |
| Macro Avg F1-Score | 96% |

### Per-Class Performance
Most classes achieve **>90% recall**. The two lowest:
- `Tomato Early Blight`: 83% — visually similar to Target Spot
- `Tomato Spider Mites`: 87% — visually similar to other spot diseases

---

## 🚀 Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
plant-disease-classifier/
│
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── best_model_phase2.keras     # Trained model (96% accuracy)
├── class_names.json            # Class label mapping
└── README.md                   # This file
```

---

## 📦 Dataset

**PlantVillage Dataset**
- Total images: 20,669
- Training set: 80% (16,541 images)
- Validation set: 20% (4,128 images)
- Image size: 224×224 pixels

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **Base Model:** MobileNetV2 (ImageNet pretrained)
- **Web App:** Streamlit
- **Training:** Google Colab (GPU T4)
- **Dataset:** PlantVillage

---

## 📝 Notes

- Input images should be clear photos of plant leaves
- Best results with well-lit, single-leaf images
- Model accepts JPG, JPEG, PNG formats

---

*Built as a Computer Vision course final project.*
