---
title: SecureLens
emoji: 🔐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🔐 SecureLens - Privacy-Preserving Medical AI

**TRUE Fully Homomorphic Encryption for Pneumonia Detection**

Secure medical diagnosis using CKKS encryption - analyze X-rays without exposing patient data.

## ✨ Features

- 🔒 **TRUE FHE Classification** - Full homomorphic encryption using CKKS
- ⚔️ **Attack Demo** - Adversarial robustness testing
- 📊 **FHE vs Traditional Comparison** - Performance benchmarking
- 🧠 **GradCAM Visualization** - Explainable AI for medical decisions
- 📊 **Model Evaluation** - Comprehensive metrics on test set

## 🚀 Deployment

SecureLens can be deployed to multiple platforms:

### Hugging Face Spaces (Current)
✅ Already deployed - Live demo available

### Render (Recommended)
Quick deployment with one command:
```bash
.\deploy_to_render.ps1
```
See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed instructions.

### Other Platforms
- Railway.app
- Fly.io
- Google Cloud Run
- Azure Container Instances

See [DEPLOYMENT_OPTIONS.md](DEPLOYMENT_OPTIONS.md) for complete comparison.

## 🛠️ Technology Stack

- **Encryption**: TenSEAL (CKKS) - 128-bit security
- **Deep Learning**: PyTorch, ResNet-18
- **Interface**: Gradio 4.44.1
- **Model**: FHE-compatible neural network
- **Dataset**: Chest X-Ray Pneumonia Detection
