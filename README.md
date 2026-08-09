<div align="center">

# 🔐 SecureLens
### TRUE Fully Homomorphic Encryption for Medical AI

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-SecureLens-blue?style=for-the-badge)](https://huggingface.co/spaces/paulamartya25/SecureLens)
[![FHE Server](https://img.shields.io/badge/Render-FHE%20Server-46E3B7?style=for-the-badge&logo=render)](https://securelens-1-m5kt.onrender.com/health)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS-7B2FBE?style=for-the-badge)](https://github.com/OpenMined/TenSEAL)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Privacy-Preserving Pneumonia Detection using CKKS Homomorphic Encryption**

*The server computes on encrypted data and mathematically cannot decrypt it — even if hacked.*

</div>

---

## 🏗️ TRUE FHE Architecture

```
┌─────────────────────────────────┐       HTTPS (ciphertext only)      ┌─────────────────────────────────┐
│   CLIENT — HuggingFace Spaces   │  ──────────────────────────────►   │   SERVER — Render.com           │
│   paulamartya25/SecureLens      │                                     │   securelens-1-m5kt.onrender.com│
│                                 │                                     │                                 │
│  ✅ ResNet-18 feature extractor  │                                     │  ✅ W1 @ enc(x) + b1            │
│  ✅ CKKS encrypt → 326 KB cipher │                                     │  ✅ W2 @ enc(h) + b2            │
│  ✅ Secret key — STAYS HERE ONLY │  ◄──────────────────────────────   │  ❌ NO secret key               │
│  ✅ Decrypt result locally       │       encrypted logits              │  ❌ NEVER decrypts              │
│                                 │                                     │  ❌ CANNOT see plaintext        │
│  Machine A (USA)                │                                     │  Machine B (Singapore)          │
└─────────────────────────────────┘                                     └─────────────────────────────────┘

       ↑ Two physically separate machines — different companies — TRUE FHE privacy guarantee
```

### Why This Matters

| Approach | Server sees patient data? | Privacy guaranteed? |
|----------|--------------------------|---------------------|
| Traditional ML | ✅ Yes — raw image | ❌ No |
| Encrypted Transport (HTTPS) | ✅ Yes — after decryption | ❌ No |
| **SecureLens (TRUE FHE)** | **❌ Never — only ciphertext** | **✅ Yes — mathematically** |

---

## ✨ Features

### 5-Tab Interface

| Tab | Description |
|-----|-------------|
| 🔒 **TRUE FHE Classification** | Upload X-ray → encrypt locally → server computes on ciphertext → local decrypt |
| ⚔️ **Attack Demo** | Test adversarial robustness with noise, blur, brightness, contrast, FGSM attacks |
| 📊 **FHE vs Traditional Comparison** | Side-by-side performance benchmarking with timing |
| 🧠 **GradCAM Visualization** | Explainable AI — see what the model focuses on |
| 📊 **Model Evaluation** | Full metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |

### Security Features
- 🔐 **128-bit CKKS encryption** (SEAL library via TenSEAL)
- 🔑 **Secret key never leaves client** — physically impossible
- 🧮 **Homomorphic linear layers** — W₁@enc(x)+b₁ → W₂@enc(h)+b₂
- 🚫 **Server has zero secret key** — verified in code
- ✅ **Graceful fallback** — if server unreachable, in-process simulation

---

## 🔬 Technical Details

### Model Architecture

```
Input X-Ray (224×224×3)
        ↓
ResNet-18 Backbone (pretrained ImageNet)
        ↓
Global Average Pooling
        ↓
512-dim Feature Vector  ← CLIENT encrypts this with CKKS
        ↓ [ENCRYPTION BOUNDARY]
enc(features) → 326 KB CKKS Ciphertext
        ↓ [SENT TO SERVER]
Linear(512 → 256) + BatchNorm  ← HOMOMORPHIC (on ciphertext)
        ↓
Linear(256 → 2)                ← HOMOMORPHIC (on ciphertext)
        ↓ [RETURNED TO CLIENT]
enc(logits) → CLIENT decrypts → Softmax → Diagnosis
```

### Why No ReLU in the FHE Head?
ReLU is not FHE-compatible — it requires comparing values which is impossible on ciphertext. The linear head (without activation) is the standard approach for FHE-compatible neural networks.

### CKKS Parameters

| Parameter | Value |
|-----------|-------|
| Polynomial modulus degree | 8192 |
| Coefficient modulus bits | [60, 40, 40, 60] |
| Global scale | 2⁴⁰ |
| Security level | 128-bit |
| Ciphertext size | ~326 KB |
| Scheme | CKKS (approximate arithmetic) |

---

## 🚀 Deployment

### Live Demo
👉 **[huggingface.co/spaces/paulamartya25/SecureLens](https://huggingface.co/spaces/paulamartya25/SecureLens)**

### Architecture
| Component | Platform | Purpose |
|-----------|----------|---------|
| Gradio Client | HuggingFace Spaces (Docker) | UI + encryption + decryption |
| FHE Server | Render.com (Docker) | Homomorphic inference only |
| Keep-Alive | UptimeRobot | Ping server every 5 min (never sleeps) |

### Run Locally

```bash
# Clone
git clone https://github.com/paulamartya25/SecureLens-.git
cd SecureLens-

# Install dependencies
pip install -r requirements.txt

# Run the Gradio app (client)
python app.py

# In a separate terminal — run the FHE server
cd server/
pip install -r requirements.txt
python server_fhe.py
```

### Run with Docker

```bash
# Client (Gradio UI)
docker build -t securelens-client .
docker run -p 7860:7860 securelens-client

# Server (FHE inference)
docker build -t securelens-server ./server/
docker run -p 10000:10000 securelens-server
```

---

## 📁 Project Structure

```
SecureLens/
├── app.py                              # HF Spaces entry point
├── app_gradio_enhanced_FOR_HF.py       # Main 5-tab Gradio interface
├── requirements.txt                    # Client dependencies
├── Dockerfile                          # Client Docker config
│
├── server/                             # 🔐 TRUE FHE Server (Render.com)
│   ├── server_fhe.py                   # Flask FHE inference API
│   ├── feature_weights.json            # W1 (256×512) — plaintext weights
│   ├── linear_weights.json             # W2 (2×256)   — plaintext weights
│   ├── requirements.txt                # Server dependencies (no torch!)
│   └── Dockerfile                      # Server Docker config
│
├── crypto_layer/
│   └── ckks_engine.py                  # CKKS encryption/decryption (CLIENT ONLY)
│
├── cloud_server/
│   ├── server.py                       # Flask server (local deployment)
│   ├── client_pipeline.py              # Native Python client pipeline
│   ├── train_model_fhe_compatible.py   # SecureLensNetFHE model definition
│   ├── models/
│   │   ├── best_model.pth              # Trained ResNet-18 weights (Git LFS)
│   │   ├── feature_weights.json        # Exported linear head W1
│   │   └── linear_weights.json         # Exported linear head W2
│   └── encrypted_inference/
│       └── he_inference.py             # HE inference engine (NEVER decrypts)
│
└── client/
    └── templates/                      # Web UI templates (Flask version)
```

---

## 🔒 FHE Security Audit

**Every `.decrypt()` call in the codebase — accounted for:**

| File | Has `.decrypt()`? | Who calls it? | Verdict |
|------|------------------|---------------|---------|
| `he_inference.py` (server) | ❌ **NEVER** | — | ✅ Perfect FHE |
| `server/server_fhe.py` (Render) | ❌ **NEVER** | — | ✅ Perfect FHE |
| `ckks_engine.py` (client) | ✅ Yes | Client only, uses secret key | ✅ Correct by design |
| `benchmark.py` | ✅ Yes | Offline tool only | ✅ Never deployed |

**Conclusion:** The server (both local and Render.com) mathematically cannot decrypt anything — it has no secret key.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Dataset | Chest X-Ray Pneumonia (Kaggle) |
| Architecture | ResNet-18 + FHE-compatible linear head |
| Training | Transfer learning, ImageNet pretrained |
| FHE Inference Time | ~5-10s (warm server) |
| Encryption Overhead | ~2x vs plaintext |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| FHE Library | [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS scheme) |
| Deep Learning | PyTorch 2.0.1, ResNet-18 |
| Web Interface | Gradio 3.50.2 |
| FHE Server | Flask + Gunicorn |
| Client Deployment | HuggingFace Spaces (Docker) |
| Server Deployment | Render.com (Docker) |
| Keep-Alive | UptimeRobot (5 min ping) |

---

## 📖 How TRUE FHE Works — Simple Explanation

```python
# ON CLIENT (patient's device / HF Space):
features = resnet18_backbone(xray_image)       # 512 numbers
ciphertext = ckks.encrypt(features)            # 326 KB encrypted blob
# Secret key stays on client — NEVER sent anywhere

# OVER THE NETWORK:
# Only ciphertext is sent to server — looks like random bytes

# ON SERVER (Render.com — different machine):
enc_h = W1 @ ciphertext + b1                  # homomorphic math on ciphertext
enc_logits = W2 @ enc_h + b2                  # still encrypted!
# Server returns encrypted_logits — cannot decrypt them (no secret key)

# BACK ON CLIENT:
logits = ckks.decrypt(enc_logits)             # uses secret key (client only)
prediction = softmax(logits)                   # Normal / Pneumonia
```

---

## 🤝 Contributing

Pull requests welcome! Areas for improvement:
- Reduce HE inference time (optimize CKKS parameters)
- Add more attack types to the Attack Demo
- Improve GradCAM visualization
- Add more medical imaging datasets

---

## ⚠️ Disclaimer

This is a **research prototype** demonstrating FHE in medical AI. It is **not intended for clinical use**. Always consult qualified medical professionals for diagnosis.

---

<div align="center">

**SecureLens** — Proving that AI can be both intelligent and private.

*CKKS · TenSEAL · PyTorch · ResNet-18 · 128-bit Security*

Made with 🔐 by [Amartya Paul](https://github.com/paulamartya25)

</div>
