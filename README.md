---
title: SecureLens
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: TRUE FHE Privacy-Preserving Pneumonia Detection (CKKS)
---

<div align="center">

# 🔐 SecureLens
### TRUE Fully Homomorphic Encryption for Medical AI

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/paulamartya25/SecureLens)
[![FHE Server](https://img.shields.io/badge/Render-FHE%20Server%20Live-46E3B7?style=for-the-badge&logo=render)](https://securelens-1-m5kt.onrender.com/health)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS%20128bit-7B2FBE?style=for-the-badge)](https://github.com/OpenMined/TenSEAL)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Privacy-Preserving Pneumonia Detection — the server computes on encrypted data and is mathematically unable to decrypt it.**

</div>

---

## 🏗️ TRUE FHE Architecture — Two Physical Machines

```
┌──────────────────────────────────┐    HTTPS — ciphertext only    ┌──────────────────────────────────┐
│  CLIENT — HuggingFace Spaces     │  ─────────────────────────►   │  SERVER — Render.com             │
│  paulamartya25/SecureLens        │                               │  securelens-1-m5kt.onrender.com  │
│                                  │                               │                                  │
│  ✅ ResNet-18 extracts features   │                               │  ✅ W1 @ enc(x) + b1  [HE math]  │
│  ✅ CKKS encrypt → 326 KB cipher  │  ◄─────────────────────────   │  ✅ W2 @ enc(h) + b2  [HE math]  │
│  ✅ Secret key — HERE ONLY        │    encrypted logits           │  ❌ NO secret key                │
│  ✅ Decrypt result locally        │                               │  ❌ NEVER decrypts               │
│                                  │                               │  ❌ Cannot see plaintext         │
│  Machine A (USA)                 │                               │  Machine B (Singapore)           │
└──────────────────────────────────┘                               └──────────────────────────────────┘

         Two physically separate machines — different companies — TRUE FHE privacy guarantee
```

### Why This Matters

| Approach | Server sees patient data? | Privacy guaranteed by? |
|----------|--------------------------|------------------------|
| Traditional ML | ✅ Yes — raw image | ❌ Nothing |
| Encrypted Transport (HTTPS) | ✅ Yes — after decryption | ❌ Trust only |
| **SecureLens TRUE FHE** | **❌ Never — only ciphertext** | **✅ Mathematics** |

---

## ✨ Features

### 5-Tab Gradio Interface

| Tab | Description |
|-----|-------------|
| 🔒 **TRUE FHE Classification** | Upload X-ray → encrypt locally → server computes on ciphertext → local decrypt → diagnosis |
| ⚔️ **Attack Demo** | Adversarial robustness testing — noise, blur, brightness, contrast, FGSM, combined |
| 📊 **FHE vs Traditional** | Side-by-side performance & timing comparison |
| 🧠 **GradCAM** | Explainable AI — gradient-weighted class activation maps |
| 📊 **Model Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |

### Security & Reliability
- 🔐 **128-bit CKKS encryption** via TenSEAL
- 🔑 **Secret key physically isolated** on client machine
- ⚡ **Wake Server button** — shows live server status before demo
- 🔄 **Background keep-alive** — pings server every 10 min
- 🤖 **UptimeRobot** — external monitor pings every 5 min
- ✅ **Auto fallback** — gracefully falls back to in-process if server unreachable

---

## 🔬 Technical Architecture

### Model Pipeline

```
Input X-Ray Image (224×224×3)
         │
         ▼
┌─────────────────────────────────┐
│  ResNet-18 Backbone             │  ← Pretrained on ImageNet
│  (feature extractor)            │
│  Global Average Pooling         │
│  Output: 512-dim feature vector │
└──────────────┬──────────────────┘
               │  Plaintext features (client only)
               ▼
┌─────────────────────────────────┐
│  CKKS Encryption (CLIENT SIDE)  │  ← Secret key generated here
│  enc(features) = 326 KB cipher  │  ← Never leaves this machine
└──────────────┬──────────────────┘
               │  Ciphertext sent over HTTPS
               ▼
┌─────────────────────────────────┐
│  FHE Server (Render.com)        │  ← Different physical machine
│                                 │
│  Layer 1: W1 @ enc(x) + b1     │  ← Pure homomorphic math
│  Layer 2: W2 @ enc(h) + b2     │  ← Still on ciphertext
│                                 │
│  Returns: enc(logits)           │  ← Never decrypted
└──────────────┬──────────────────┘
               │  Encrypted logits returned
               ▼
┌─────────────────────────────────┐
│  Client Decryption              │  ← Uses secret key (client only)
│  Softmax → Prediction           │
│  Display: Normal / Pneumonia    │
└─────────────────────────────────┘
```

### Why No ReLU in the FHE Head?
ReLU requires value comparison on ciphertext — computationally intractable in CKKS. The linear head (no non-linearity) is the standard approach for FHE-compatible neural networks. Accuracy is preserved because the ResNet-18 backbone handles all non-linear feature extraction.

### CKKS Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Polynomial modulus degree | 8192 | 128-bit security |
| Coefficient modulus bits | [60, 40, 40, 60] | 2 multiplication levels |
| Global scale | 2⁴⁰ | Precision balance |
| Security level | 128-bit | SEAL library standard |
| Ciphertext size | ~326 KB | 512-dim vector |

---

## 📁 Project Structure

```
SecureLens/
├── app.py                                   # HF Spaces entry point
├── app_gradio_enhanced_FOR_HF.py            # Main 5-tab Gradio interface
│                                            # ← fhe_server_infer() calls Render over HTTP
│                                            # ← warm_up_server() for Wake button
│                                            # ← _keep_alive_loop() background thread
├── requirements.txt                         # Client dependencies
├── Dockerfile                               # Client Docker config
│
├── server/                                  # 🔐 TRUE FHE Server (Render.com)
│   ├── server_fhe.py                        # Flask API — HE inference only, no decrypt
│   ├── feature_weights.json                 # W1 matrix (256×512)
│   ├── linear_weights.json                  # W2 matrix (2×256)
│   ├── requirements.txt                     # flask, tenseal, numpy only (no torch!)
│   └── Dockerfile                           # Render.com Docker config
│
├── crypto_layer/
│   └── ckks_engine.py                       # CKKS engine — encrypt/decrypt (CLIENT ONLY)
│
├── cloud_server/
│   ├── server.py                            # Flask server for local deployment
│   ├── client_pipeline.py                   # Native Python client (true FHE locally)
│   ├── train_model_fhe_compatible.py        # SecureLensNetFHE — FHE-compatible model
│   ├── models/
│   │   ├── best_model.pth                   # Trained weights (Git LFS)
│   │   ├── feature_weights.json             # Exported W1 (Git LFS)
│   │   └── linear_weights.json             # Exported W2
│   └── encrypted_inference/
│       └── he_inference.py                  # HE engine — NEVER calls .decrypt()
│
└── client/
    └── templates/                           # HTML templates (Flask web UI version)
```

---

## 🚀 Run It Yourself

### Option 1 — Live Demo (No Setup)
👉 **[huggingface.co/spaces/paulamartya25/SecureLens](https://huggingface.co/spaces/paulamartya25/SecureLens)**

### Option 2 — Local (True Physical Separation)

```bash
# Clone repo
git clone https://github.com/paulamartya25/SecureLens-.git
cd SecureLens-

# Terminal 1 — Run the FHE server (Machine B)
cd server/
pip install -r requirements.txt
python server_fhe.py
# → Running on http://localhost:10000

# Terminal 2 — Run the Gradio client (Machine A)
cd ..
pip install -r requirements.txt
FHE_SERVER_URL=http://localhost:10000 python app.py
# → Open http://localhost:7860
```

### Option 3 — Docker

```bash
# Server
docker build -t securelens-server ./server/
docker run -p 10000:10000 securelens-server

# Client (in another terminal)
docker build -t securelens-client .
docker run -p 7860:7860 -e FHE_SERVER_URL=http://host.docker.internal:10000 securelens-client
```

### Option 4 — Native Python Client (Full True FHE)

```bash
# Uses client_pipeline.py — secret key truly local
python cloud_server/client_pipeline.py cloud_server/models/best_model.pth your_xray.jpg
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| FHE | [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS) | Homomorphic encryption |
| Deep Learning | PyTorch 2.0.1, ResNet-18 | Feature extraction |
| Web Interface | Gradio 3.50.2 | 5-tab demo UI |
| FHE Server | Flask + Gunicorn | HE inference API |
| Client Hosting | HuggingFace Spaces (Docker) | Public demo |
| Server Hosting | Render.com (Docker) | Separate FHE server |
| Uptime | UptimeRobot | Keep-alive monitoring |

---

## 📖 How CKKS Homomorphic Encryption Works

```python
# Standard ML inference (NOT private):
features = model(xray)          # plaintext
logits = W @ features + b       # plaintext
prediction = softmax(logits)

# SecureLens FHE inference (PRIVATE):
features = backbone(xray)       # plaintext — CLIENT ONLY
enc_x = ckks.encrypt(features) # 326 KB ciphertext — client

# Server receives enc_x — cannot see features
enc_h = enc_x.dot(W1) + b1     # homomorphic dot product on ciphertext
enc_logits = enc_h.dot(W2) + b2 # still encrypted

# Client receives enc_logits — server never saw plaintext
logits = ckks.decrypt(enc_logits)  # secret key on client only
prediction = softmax(logits)        # Normal / Pneumonia
```

---

## ⚠️ Disclaimer

Research prototype demonstrating FHE in medical AI. **Not intended for clinical use.**

---

<div align="center">

**SecureLens** — Proving that AI can be both intelligent and private.

*CKKS · TenSEAL · PyTorch · ResNet-18 · 128-bit Security*

Built by [Amartya Paul](https://github.com/paulamartya25)

</div>
