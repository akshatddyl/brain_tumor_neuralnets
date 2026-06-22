<div align="center">

# 🧠 Brain Tumor Detection AI

A full end-to-end deep learning pipeline that classifies brain MRI scans as **tumor / no tumor** — trained in Google Colab and a web app interface.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

</div>

---

## 🧩 About The Project

This project is a complete deep learning pipeline for **binary classification of brain MRI scans** — determining whether a scan indicates the presence of a brain tumor.

The workflow starts in a **Google Colab notebook** where a CNN is trained on labelled MRI imagery, and the resulting weights are exported as `brain_tumor_model.h5`. That model is then served through an interactive **Streamlit web app** where any user can upload their own scan and receive an instant AI prediction with a confidence score — no code required.

> ⚠️ **Medical Disclaimer:** This tool is built for **educational and research purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional.

---

## 📸 Screenshots
🚨 Tumor Detected<img width="2016" height="1139" alt="Screenshot From 2026-06-15 14-25-51" src="https://github.com/user-attachments/assets/cc862f74-d0f1-464e-b8c7-45fbe5bdc451"/>
✅ No Tumor Detected<img width="2016" height="1139" alt="Screenshot From 2026-06-15 14-25-37" src="https://github.com/user-attachments/assets/803d1096-f6eb-40f2-8a7e-91ce2e0c88f8" />
>

---

## ✨ Features

- **🔬 Real-Time MRI Analysis** — Upload any `.jpg`, `.jpeg`, or `.png` brain MRI scan and get an instant binary prediction.
- **🧠 Custom CNN Architecture** — A 3-block Convolutional Neural Network designed, trained, and evaluated from scratch on real MRI data.
- **📊 Confidence Scoring** — Displays the model's raw sigmoid output as a human-readable percentage, for both positive and negative predictions.
- **🌐 Zero-Code Web UI** — A clean, intuitive Streamlit interface
- **📉 Training Transparency** — Accuracy and loss curves are plotted per epoch in the notebook so you can inspect overfitting and convergence at a glance.
- **🧮 Bonus: ML from Scratch** — A pure NumPy implementation of Lasso (L1) Linear Regression, demonstrating mathematical fluency beyond library calls.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | TensorFlow 2.x / Keras |
| **Image Processing** | OpenCV, Pillow |
| **Web App Framework** | Streamlit |
| **Numerical Computing** | NumPy |
| **Data & Splitting** | Pandas, Scikit-learn |
| **Visualization** | Matplotlib |
| **Training Environment** | Google Colab (GPU) |
| **Language** | Python 3.9+ |

---

## 📁 Project Structure

```
brain-tumor-detection/
│
├── app.py                    # 🌐 Streamlit web app — main entry point for users
├── brain_tumor_cnn.ipynb     # 📓 Colab notebook: preprocessing, CNN training, export
├── linear_reg.py             # 🧮 Linear Regression + L1 regularization from scratch
├── brain_tumor_model.h5      # 🤖 Trained model weights (generated — not tracked by Git)
├── .gitignore                # 🙈 Ignores venv/, __pycache__, and .pyc files
└── README.md                 # 📖 You are here
```

> **Note:** `brain_tumor_model.h5` is excluded from version control. You must generate it yourself by running the Colab notebook first (see [Usage](#-usage)).

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- `pip` package manager

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/brain-tumor-detection.git
cd brain_tumor_neuralnets
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install streamlit tensorflow opencv-python pillow numpy scikit-learn matplotlib
```

---

## 💻 Usage

### Step 1 — Train the Model in Google Colab

> Do this first. The web app requires the `.h5` model file to run.

1. Open `brain_tumor_cnn.ipynb` in **[Google Colab](https://colab.research.google.com/)**.
2. Set the runtime to **GPU** (`Runtime → Change runtime type → T4 GPU`).
3. Run all cells. The notebook will automatically:
   - Clone the MRI dataset from GitHub
   - Resize and normalize all images to `150×150 px`
   - Train the CNN for **15 epochs**
   - Plot accuracy & loss curves
   - Save the model as **`brain_tumor_model.h5`**
4. **Download `brain_tumor_model.h5`** from the Colab file panel and drop it in the **root of this project folder**.

### Step 2 — Run the Web App

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. Use the uploader to drop in any brain MRI image — the model will return a prediction and confidence score within seconds.

### Step 3 — (Optional) Run the Linear Regression Demo

```bash
python linear_reg.py
```

Trains the custom Lasso regression on synthetic data (`y = 3x + 4 + noise`) and saves a plot as `linear_regression_plot.png`.

---
## Results

<img width="990" height="451" alt="colab" src="https://github.com/user-attachments/assets/0372c052-faba-4ec3-952b-8fb946a93a7b" />

---

## 🏗️ Model Architecture

The CNN is a sequential model with **3 convolutional blocks** that progressively extract spatial features, followed by a fully connected classifier head.
<img width="1280" height="720" alt="convnet_architecture" src="https://github.com/user-attachments/assets/5edf1b0f-ae9e-4cf5-af4f-c29c8b5ac061" />


```
Input → (150 × 150 × 3 RGB Image)
│
├── Conv2D(32 filters, 3×3, ReLU)  →  MaxPooling2D(2×2)
├── Conv2D(64 filters, 3×3, ReLU)  →  MaxPooling2D(2×2)
├── Conv2D(128 filters, 3×3, ReLU) →  MaxPooling2D(2×2)
│
├── Flatten()
├── Dense(128, ReLU)
├── Dropout(0.5)         ← Regularisation against overfitting
│
└── Dense(1, Sigmoid)    ← Output: P(tumor)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Metric | Accuracy |
| Input Resolution | 150 × 150 × 3 |
| Epochs | 15 |
| Batch Size | 32 |
| Dropout Rate | 0.5 |
| Train / Test Split | 80% / 20% |

> The sigmoid output is thresholded at **0.5**: above means tumor detected, below means healthy.

---

## 🧮 Bonus: Linear Regression from Scratch

`linear_reg.py` implements **Lasso (L1) Linear Regression** using pure NumPy and Gradient Descent — zero scikit-learn for the core algorithm.

**Key concepts implemented:**

- **Bias augmentation** — a column of ones is prepended to `X` so the bias is learned as `weights[0]`, keeping the gradient update unified.
- **MSE Gradient** — `dw = (1/n) · Xᵀ · (ŷ − y)`
- **L1 Penalty** — `λ · sign(w)` is added to the gradient at each step, driving less important weights toward zero. The bias term is explicitly excluded from regularisation.
- **Validation** — trains on `y = 3x + 4 + noise` and recovers the ground-truth coefficients to within a few thousandths.
- **Result**

  <img width="640" height="480" alt="linear_regression_plot" src="https://github.com/user-attachments/assets/33555faa-1653-42f8-9956-efcf3ccb8da2" />

---

## 📊 Dataset

| Detail | Info |
|---|---|
| **Source** | [mohamedalihabib/Brain-Tumor-Detection](https://github.com/mohamedalihabib/Brain-Tumor-Detection) |
| **Classes** | `yes/` (tumor present) · `no/` (healthy) |
| **Preprocessing** | Resized to `150×150 px`, pixel values normalised to `[0, 1]` |
| **Split Strategy** | `train_test_split` · `test_size=0.2` · `random_state=42` |

---
