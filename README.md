# SecureLens 🔒
### Privacy-Preserving Medical Image Diagnostics using Fully Homomorphic Encryption

![Python](https://img.shields.io/badge/Python-3.13-blue)
![TenSEAL](https://img.shields.io/badge/TenSEAL-0.3.14-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-red)
![Flask](https://img.shields.io/badge/Flask-3.1-lightgrey)
![Accuracy](https://img.shields.io/badge/Accuracy-89.42%25-brightgreen)

---

## Overview

SecureLens is a privacy-preserving medical image diagnostic system
that uses **Fully Homomorphic Encryption (FHE)** to classify chest
X-rays as Normal or Pneumonia — without the cloud server ever
seeing the raw patient data.

The AI model runs inference **directly on encrypted data** using
the **CKKS scheme** (Cheon-Kim-Kim-Song) implemented via TenSEAL.

---

## Pipeline
Client                    Crypto Layer              Cloud Server
│                            │                         │
│  Upload Chest X-Ray        │                         │
│──────────────────────────► │                         │
│                            │  Encrypt via CKKS       │
│                            │─────────────────────────►
│                            │                         │
│                            │                    Run Linear
│                            │                    Evaluation
│                            │                    on Ciphertext
│                            │                         │
│                            │  Encrypted Prediction   │
│                            │◄─────────────────────────
│  Decrypt Result            │                         │
│◄───────────────────────────│                         │
│                            │                         │
Render Diagnosis

---

## Features

- **CKKS Homomorphic Encryption** — 128-bit security
- **ResNet-18 Transfer Learning** — 89.42% test accuracy
- **Zero plaintext exposure** — server never sees pixel data
- **Real-time inference** — results in seconds
- **Interactive UI** — drag-and-drop chest X-ray upload
- **Pipeline visualization** — shows each encryption step

---

## Project Structure
SecureLens/
├── app.py                          # Entry point
├── requirements.txt
├── README.md
│
├── crypto_layer/
│   └── ckks_engine.py             # CKKS encryption/decryption
│
├── cloud_server/
│   ├── server.py                  # Flask API
│   ├── train_model.py             # Model training
│   ├── models/                    # Saved weights
│   └── encrypted_inference/
│       └── he_inference.py        # HE inference engine
│
├── client/
│   ├── templates/index.html       # Frontend UI
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
├── utils/
│   └── prepare_dataset.py
│
├── data/
│   └── chest_xray/               # Kaggle dataset
│       ├── train/
│       ├── val/
│       └── test/
│
└── tests/
├── test_ckks.py
├── test_inference.py
└── test_api.py

---

## Installation

```bash
# Clone or download project
cd SecureLens

# Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1     # Windows
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Download from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Place in `data/chest_xray/` with structure:
data/chest_xray/train/NORMAL/
data/chest_xray/train/PNEUMONIA/
data/chest_xray/val/NORMAL/
data/chest_xray/val/PNEUMONIA/
data/chest_xray/test/NORMAL/
data/chest_xray/test/PNEUMONIA/

---

## Training

```bash
python utils/prepare_dataset.py
python cloud_server/train_model.py
```

**Results:**
| Metric | Value |
|--------|-------|
| Train Accuracy | 97.68% |
| Val Accuracy | 97.39% |
| Test Accuracy | **89.42%** |
| Val/Test Gap | 7.97% |
| Epochs | 20 |

---

## Running the App

```bash
python app.py
```

Open browser at: `http://127.0.0.1:5000`

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Encryption Details

| Parameter | Value |
|-----------|-------|
| Scheme | CKKS (Cheon-Kim-Kim-Song) |
| Library | TenSEAL 0.3.14 |
| Poly Modulus Degree | 8192 |
| Security Level | 128-bit |
| Global Scale | 2^40 |
| Ciphertext Size | ~326 KB |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Encryption | TenSEAL (CKKS) |
| Deep Learning | PyTorch + ResNet-18 |
| Backend | Flask |
| Frontend | HTML + CSS + JavaScript |
| Dataset | Kaggle Chest X-Ray (5,856 images) |

---

## Results

- Normal X-ray → correctly classified as **Normal**
- Pneumonia X-ray → correctly classified as **Pneumonia**
- All inference runs on **encrypted data**
- Server has **zero access** to plaintext pixels

---

## Author

**Amartya** — Final Year Student
Thesis: Privacy-Preserving Medical Image Diagnostics
         using Fully Homomorphic Encryption

