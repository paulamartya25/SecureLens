---
title: SecureLens
emoji: 🔒
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
# SecureLens ðŸ”’
---
title: SecureLens
emoji: ðŸ”’
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---
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
X-rays as Normal or Pneumonia â€” without the cloud server ever
seeing the raw patient data.

The AI model runs inference **directly on encrypted data** using
the **CKKS scheme** (Cheon-Kim-Kim-Song) implemented via TenSEAL.

---

## Pipeline
Client                    Crypto Layer              Cloud Server
â”‚                            â”‚                         â”‚
â”‚  Upload Chest X-Ray        â”‚                         â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º â”‚                         â”‚
â”‚                            â”‚  Encrypt via CKKS       â”‚
â”‚                            â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º
â”‚                            â”‚                         â”‚
â”‚                            â”‚                    Run Linear
â”‚                            â”‚                    Evaluation
â”‚                            â”‚                    on Ciphertext
â”‚                            â”‚                         â”‚
â”‚                            â”‚  Encrypted Prediction   â”‚
â”‚                            â”‚â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
â”‚  Decrypt Result            â”‚                         â”‚
â”‚â—„â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚                         â”‚
â”‚                            â”‚                         â”‚
Render Diagnosis

---

## Features

- **CKKS Homomorphic Encryption** â€” 128-bit security
- **ResNet-18 Transfer Learning** â€” 89.42% test accuracy
- **Zero plaintext exposure** â€” server never sees pixel data
- **Real-time inference** â€” results in seconds
- **Interactive UI** â€” drag-and-drop chest X-ray upload
- **Pipeline visualization** â€” shows each encryption step

---

## Project Structure
SecureLens/
â”œâ”€â”€ app.py                          # Entry point
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ README.md
â”‚
â”œâ”€â”€ crypto_layer/
â”‚   â””â”€â”€ ckks_engine.py             # CKKS encryption/decryption
â”‚
â”œâ”€â”€ cloud_server/
â”‚   â”œâ”€â”€ server.py                  # Flask API
â”‚   â”œâ”€â”€ train_model.py             # Model training
â”‚   â”œâ”€â”€ models/                    # Saved weights
â”‚   â””â”€â”€ encrypted_inference/
â”‚       â””â”€â”€ he_inference.py        # HE inference engine
â”‚
â”œâ”€â”€ client/
â”‚   â”œâ”€â”€ templates/index.html       # Frontend UI
â”‚   â””â”€â”€ static/
â”‚       â”œâ”€â”€ css/style.css
â”‚       â””â”€â”€ js/main.js
â”‚
â”œâ”€â”€ utils/
â”‚   â””â”€â”€ prepare_dataset.py
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ chest_xray/               # Kaggle dataset
â”‚       â”œâ”€â”€ train/
â”‚       â”œâ”€â”€ val/
â”‚       â””â”€â”€ test/
â”‚
â””â”€â”€ tests/
â”œâ”€â”€ test_ckks.py
â”œâ”€â”€ test_inference.py
â””â”€â”€ test_api.py

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

- Normal X-ray â†’ correctly classified as **Normal**
- Pneumonia X-ray â†’ correctly classified as **Pneumonia**
- All inference runs on **encrypted data**
- Server has **zero access** to plaintext pixels

---

## Author

**Amartya** â€” Final Year Student
Thesis: Privacy-Preserving Medical Image Diagnostics
         using Fully Homomorphic Encryption


