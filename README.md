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

# 🔐 SecureLens — TRUE Fully Homomorphic Encryption

**Privacy-Preserving Pneumonia Detection**

The server computes on **encrypted** chest X-rays and is **mathematically unable to decrypt** — even if compromised.

## Architecture

```
HuggingFace (Client)          Render.com (Server)
────────────────────          ────────────────────
ResNet-18 features            W1@enc(x)+b1 (HE)
CKKS encrypt → 326KB   ──►   W2@enc(h)+b2 (HE)
Secret key stays here  ◄──   Encrypted logits
Local decrypt & show          NO secret key ever
```

## Features
- 🔒 **TRUE FHE Classification** — CKKS encrypted inference on a separate physical server
- ⚔️ **Attack Demo** — Adversarial robustness testing  
- 📊 **FHE vs Traditional** — Performance comparison
- 🧠 **GradCAM** — Explainable AI visualization
- 📊 **Model Evaluation** — Full metrics (Accuracy, F1, ROC-AUC)

## Tech Stack
- **FHE**: TenSEAL (CKKS, 128-bit security)
- **ML**: PyTorch, ResNet-18
- **UI**: Gradio
- **Server**: Flask on Render.com (separate machine)

## Full Documentation
👉 [GitHub Repository](https://github.com/paulamartya25/SecureLens-)
