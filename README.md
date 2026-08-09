<div align="center">

# 🦠 MalariaDetection AI

### Intelligent Malaria Cell Detection using Deep Learning & Transfer Learning

<p>An AI-powered web application that detects malaria-infected blood smear images using MobileNetV2 and Streamlit.</p>

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-MobileNetV2-D00000?style=for-the-badge&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit)

</div>

---

# 📖 Overview

MalariaVision AI is a deep learning-powered healthcare application that automatically detects **malaria-infected blood cells** from microscopic blood smear images.

Built using **MobileNetV2 Transfer Learning**, the application performs fast and reliable binary classification while providing an intuitive **Streamlit** interface for real-time predictions.

Demo Link: https://huggingface.co/spaces/Natzi21/malaria-cell-detection


---

# ✨ Features

| 🚀 Feature | Description |
|------------|-------------|
| 🦠 Binary Classification | Detects Parasitized and Uninfected cells |
| 🧠 Transfer Learning | MobileNetV2 backbone |
| ⚡ Real-Time Prediction | Instant AI inference |
| 📊 Confidence Score | Displays prediction probability |
| 📈 Dashboard | Interactive Streamlit UI |
| 📉 Training Visualizations | Accuracy & Loss Curves |

---

# 🧠 AI Workflow

```text
Blood Smear Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 MobileNetV2
        │
        ▼
 Feature Extraction
        │
        ▼
 Dense Layers
        │
        ▼
 Prediction
        │
 ┌──────┴───────┐
 ▼              ▼
Parasitized  Uninfected
```

---

# 🏗️ Project Architecture

```text
User Upload
     │
     ▼
Blood Smear Image
     │
     ▼
Preprocessing
     │
     ▼
MobileNetV2
     │
     ▼
Classification Head
     │
     ▼
Prediction + Confidence
     │
     ▼
Streamlit Dashboard
```

---

# 📂 Project Structure

```text
MalariaVision-AI/
│
├── app.py
├── malaria_detection.py
├── requirements.txt
├── runtime.txt
├── training_history.png
├── evaluation_plots.png
└── README.md
```

---

# 📊 Dataset

- **Dataset:** NIH Malaria Cell Images Dataset
- **Total Images:** 27,558
- **Classes:** Parasitized & Uninfected
- **Training Split:** 80%
- **Validation Split:** 20%

---

# 🧬 Model

| Component | Details |
|-----------|---------|
| Backbone | MobileNetV2 |
| Framework | TensorFlow / Keras |
| Learning | Transfer Learning |
| Task | Binary Classification |
| Deployment | Streamlit |

---

# 📈 Training Strategy

- Phase 1: Frozen MobileNetV2 feature extractor
- Phase 2: Fine-tuning last layers
- Adam Optimizer
- Early Stopping
- ReduceLROnPlateau
- Model Checkpointing

---

# ⚙️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, Pillow |
| Framework | Streamlit |
| Visualization | Matplotlib |

---

# 📸 Screenshots

> Replace these placeholders with your application screenshots.

| Home | Prediction |
|------|------------|
| ![](screenshots/home.png) | ![](screenshots/prediction.png) |

---

# 🚀 Installation

```bash
git clone https://github.com/yourusername/MalariaVision-AI.git
cd MalariaVision-AI
pip install -r requirements.txt
streamlit run app.py
```

---

# 💻 Usage

1. Launch the application.
2. Upload a blood smear image.
3. Wait for AI prediction.
4. View confidence score.
5. Analyze the result.

---

# 📈 Results

| Metric | Value |
|---------|------:|
| Accuracy | ~94% |
| AUC | ~0.98 |
| Model | MobileNetV2 |
| Classes | 2 |

---

# 🔮 Future Improvements

- Grad-CAM Explainability
- Cloud Deployment
- REST API
- Mobile Application
- Vision Transformer (ViT)
- Multi-class Parasite Detection

---

# 🤝 Contributing

Contributions are welcome! Fork the repository, create a feature branch, commit your changes, and open a Pull Request.

---

# 📜 Disclaimer

This project is intended for **educational and research purposes only** and should not be used as a replacement for professional medical diagnosis.

---

# 👨‍💻 Authors

**Limnisha Changkakati**
**Natalia Mathews**
**Prema Malipatil**

---

<div align="center">

### ⭐ If you found this project useful, consider starring the repository!



</div>
