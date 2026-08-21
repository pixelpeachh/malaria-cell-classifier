# 🩸 Malaria Cell Classifier

A deep learning image classification project that classifies malaria blood-cell images as **Parasitized** or **Uninfected** using a trained **ResNet18** model.

The trained model is integrated into a **Streamlit** web application where users can upload a cell image and receive a prediction with the model's confidence score.

> ⚠️ **Educational & Research Use Only:** This application is not a medical diagnostic tool and should not be used for diagnosis, treatment, or clinical decision-making.

---

## 🧠 Model

- **Architecture:** ResNet18
- **Classes:** Parasitized, Uninfected
- **Input Size:** 128 × 128 RGB images
- **Training Epochs:** 5
- **Test Accuracy:** **95.74%**

### Preprocessing

Images are:

1. Converted to RGB
2. Resized to 128 × 128
3. Converted to tensors
4. Normalized using statistics calculated from the training dataset

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Pillow**
- **Streamlit**
- **Jupyter Notebook**
- **Git & GitHub**

---

## 🔄 Workflow

```text
Cell Image
    ↓
Preprocessing
    ↓
ResNet18
    ↓
Classification
    ↓
Softmax
    ↓
Prediction + Confidence
    ↓
Streamlit UI
```

## 📊 Results

| Metric | Parasitized | Uninfected |
|---|---:|---:|
| Precision | 0.95 | 0.97 |
| Recall | 0.97 | 0.95 |
| F1-Score | 0.96 | 0.96 |
| Support | 2042 | 2093 |

**Test Accuracy:** `95.74%`

### Confusion Matrix

```text
[[1980   62]
 [ 114 1979]]

* Correctly classified 1980 Parasitized images
* Correctly classified 1979 Uninfected images
* 62 Parasitized images were classified as Uninfected
* 114 Uninfected images were classified as Parasitized
```

⸻

📁 Project Structure
malaria-classifier/
│
├── app.py
├── malaria_resnet18.pth
├── notebook.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── config.toml

The dataset and virtual environment are excluded from the repository.

⸻

🚀 Run Locally

Clone the repository:
git clone <repository-url>
cd malaria-classifier

Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

⸻

⚠️ Disclaimer
This project is intended for educational and research purposes only. Model predictions may be incorrect and should not replace professional medical evaluation.

