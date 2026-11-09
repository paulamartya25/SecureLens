---
title: SecureLens
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Encrypted (CKKS) pneumonia detection prototype
---

# SecureLens

### Encrypted inference for chest X-ray classification using CKKS homomorphic encryption

[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/paulamartya25/SecureLens)
[![FHE Server](https://img.shields.io/badge/Render-FHE%20Server%20Live-46E3B7?style=for-the-badge&logo=render)](https://securelens-1-m5kt.onrender.com/health)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![TenSEAL](https://img.shields.io/badge/TenSEAL-CKKS-7B2FBE?style=for-the-badge)](https://github.com/OpenMined/TenSEAL)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

SecureLens is a research prototype that classifies chest X-rays as Normal or Pneumonia while the server only ever sees **encrypted** data. A ResNet-18 backbone extracts features on the client. The features are encrypted with the CKKS scheme (via TenSEAL), and a server computes the final classification layers directly on the ciphertext. Only the holder of the secret key can read the result.

This is the author's undergraduate thesis project. It demonstrates the idea end to end, but it has clear limits, which are stated below. Please read **Threat Model and Limitations** before relying on any privacy claim.

**Not intended for clinical use.**

---

## What is and is not encrypted

| Stage | Where it runs | Protected by encryption? |
| --- | --- | --- |
| X-ray image to 512-dim feature vector (ResNet-18) | Client | No, plaintext on the client |
| Final classification layers (512 → 256 → 2) | Server | Yes, computed on ciphertext |
| Prediction (decrypted logits, softmax) | Client | Decrypted locally with the secret key |

Only the final linear layers are encrypted. The deep network is not.

---

## How it works

```
 CLIENT (holds the secret key)                      FHE SERVER (no secret key)
 ─────────────────────────────                      ──────────────────────────
 X-ray (224x224x3)
   → ResNet-18 → 512-dim features (plaintext)
   → CKKS encrypt (~326 KB ciphertext)  ─ ciphertext ─►  enc(h) = W1 · enc(x) + b1
                                                         enc(y) = W2 · enc(h) + b2
 decrypt logits  ◄──────── encrypted logits ─────────  (never decrypts anything)
   → softmax → Normal / Pneumonia
```

The server stores the model weights (W1: 256×512, W2: 2×256) and performs only homomorphic additions and multiplications. It has no access to the secret key.

---

## Threat Model and Limitations

**Assumed attacker:** an honest-but-curious inference server. It follows the protocol but would read patient data if it could.

**What the design protects:** the feature vector sent to the server and the resulting logits are visible to the server only as ciphertext. The security of this rests on the CKKS scheme (RLWE assumption) with parameters at the 128-bit level (see below).

**What it does not protect, or has not been shown to protect:**

1. **The hosted demo's "client" is itself a remote server.** The Gradio app runs on HuggingFace Spaces, so in the live demo the plaintext image and the secret key exist on HuggingFace's infrastructure, not on the user's device. The intended separation, where the key never leaves the user's machine, only holds when you run the native client (Option 4 below) locally.
2. **Only the classification head is encrypted.** The ResNet-18 backbone runs on plaintext. Encrypting a deep network is far more expensive and is not attempted here.
3. **The model is not private.** The server holds the weights in plaintext. This protects the user's input, not the model owner's intellectual property.
4. **No integrity guarantee.** A malicious server could return wrong results and the client cannot tell. Verifiable computation is out of scope.
5. **Metadata leaks.** The server can see ciphertext size, request timing, and request frequency.
6. **CKKS is approximate.** Decrypted logits carry small numerical error compared with plaintext inference.
7. **No formal end-to-end security proof.** The claim is "the server sees only CKKS ciphertexts under standard parameters," not a proof for the whole system.
8. **Encrypted-vs-plaintext agreement is not yet reported systematically.** The accuracy figures below are from plaintext evaluation of the model.

---

## Results

### Classification (plaintext evaluation, 624-image test set: 234 Normal, 390 Pneumonia)

| Metric | Value |
| --- | --- |
| Accuracy | 91.19% |
| Recall (sensitivity) | 99.74% |
| Specificity | 76.92% |
| Precision | 87.81% |
| ROC-AUC | 0.9656 |

Confusion matrix: TN = 180, FP = 54, FN = 1, TP = 389.

The model almost never misses pneumonia (1 false negative) but flags 54 of 234 normal images as pneumonia. It is tuned toward sensitivity, and the lower specificity is a known weakness.

### Cost of encryption (from the demo's FHE benchmark)

| Measure | Value |
| --- | --- |
| Encryption time | 5.9 ms |
| Homomorphic inference time | 14.6 ms |
| Ciphertext size | ~326 KB |
| Overhead vs plaintext head | ~597x |

These numbers cover the cryptographic steps only. End-to-end latency in the hosted demo is higher because of network transfer and server cold starts.

---

## Design Choices

### CKKS parameters

| Parameter | Value | Reason |
| --- | --- | --- |
| Polynomial modulus degree | 8192 | Standard size for the 128-bit security level |
| Coefficient modulus bits | [60, 40, 40, 60] | Total 200 bits, within the 218-bit limit for 128-bit security at this degree; supports 2 multiplication levels |
| Global scale | 2^40 | Balance between precision and noise growth |
| Security level | 128-bit | Per the Homomorphic Encryption Standard tables |

### Why there is no ReLU in the encrypted head

CKKS only supports addition and multiplication. ReLU needs a comparison, which is not a polynomial, so using it would require a polynomial approximation that consumes extra multiplication levels.

**Consequence:** the encrypted head has no activation function, so its two layers mathematically collapse into one linear layer, `(W2·W1)x + (W2·b1 + b2)`. In practice this is a linear classifier on top of ResNet-18 features. All non-linear feature learning happens in the plaintext backbone.

---

## Features

The Gradio interface has five tabs:

| Tab | Description |
| --- | --- |
| FHE Classification | Upload an X-ray, encrypt locally, server computes on ciphertext, decrypt and show the diagnosis |
| Attack Demo | Robustness testing against noise, blur, brightness, contrast, FGSM, and combinations |
| FHE vs Traditional | Side-by-side timing comparison of encrypted and plaintext inference |
| GradCAM | Gradient-weighted class activation maps showing where the model looks |
| Model Evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |

Reliability features: a "Wake Server" button for cold starts, a background keep-alive ping, and an automatic fallback to in-process inference if the FHE server is unreachable. In that fallback case the data is **not** sent to the remote server, so the privacy separation does not apply.

---

## Project Structure

```
SecureLens/
├── app.py                              # HuggingFace Spaces entry point
├── app_gradio_enhanced_FOR_HF.py       # Main 5-tab Gradio interface (client)
├── requirements.txt                    # Client dependencies
├── Dockerfile                          # Client Docker config
│
├── server/                             # FHE inference server (deployed on Render)
│   ├── server_fhe.py                   # Flask API, homomorphic inference only
│   ├── feature_weights.json            # W1 (256x512)
│   ├── linear_weights.json             # W2 (2x256)
│   ├── requirements.txt                # flask, tenseal, numpy (no torch)
│   └── Dockerfile
│
├── crypto_layer/
│   └── ckks_engine.py                  # CKKS encrypt/decrypt (client side only)
│
├── cloud_server/
│   ├── server.py                       # Flask server for local deployment
│   ├── client_pipeline.py              # Native Python client (key stays local)
│   ├── train_model_fhe_compatible.py   # FHE-compatible model definition and training
│   ├── models/                         # Trained weights (Git LFS)
│   └── encrypted_inference/
│       └── he_inference.py             # Homomorphic engine, never calls decrypt
│
├── tests/
└── docs/
```

---

## Run It Yourself

### Option 1: Live demo (no setup)

[huggingface.co/spaces/paulamartya25/SecureLens](https://huggingface.co/spaces/paulamartya25/SecureLens)

The first request may be slow while the server wakes up. See the limitation above about where the key lives in this mode.

### Option 2: Local, two processes

```bash
git clone https://github.com/paulamartya25/SecureLens.git
cd SecureLens

# Terminal 1: FHE server
cd server
pip install -r requirements.txt
python server_fhe.py          # http://localhost:10000

# Terminal 2: Gradio client
cd ..
pip install -r requirements.txt
FHE_SERVER_URL=http://localhost:10000 python app.py   # http://localhost:7860
```

### Option 3: Docker

```bash
docker build -t securelens-server ./server/
docker run -p 10000:10000 securelens-server

docker build -t securelens-client .
docker run -p 7860:7860 -e FHE_SERVER_URL=http://host.docker.internal:10000 securelens-client
```

### Option 4: Native client (secret key stays on your machine)

```bash
python cloud_server/client_pipeline.py cloud_server/models/best_model.pth your_xray.jpg
```

This is the mode that matches the intended threat model.

---

## Technology Stack

| Component | Technology |
| --- | --- |
| Homomorphic encryption | [TenSEAL](https://github.com/OpenMined/TenSEAL) (CKKS) |
| Deep learning | PyTorch 2.0.1, ResNet-18 (ImageNet pretrained) |
| Web interface | Gradio 3.50.2 |
| FHE server | Flask + Gunicorn |
| Hosting | HuggingFace Spaces (client), Render (server) |

---

## Future Work

- Add a polynomial activation (for example, square) between the two encrypted layers so the head is genuinely non-linear, using one extra multiplication level.
- Measure and report agreement between encrypted and plaintext predictions on the full test set.
- Improve specificity, for example with threshold calibration or class-weighted training.
- Report end-to-end latency on a local two-machine setup.
- Explore protecting the model weights as well as the input.

---

## Disclaimer

Research prototype demonstrating homomorphic encryption for medical image classification. **Not intended for clinical use.**

## License

MIT. See [LICENSE](LICENSE).

## Author

Amartya Paul. [GitHub](https://github.com/paulamartya25)
