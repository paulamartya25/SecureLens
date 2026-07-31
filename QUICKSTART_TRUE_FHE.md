# SecureLens True FHE - Quick Start Guide

## What Changed?

SecureLens now implements **TRUE end-to-end Fully Homomorphic Encryption**. The server can NEVER decrypt your medical images or diagnostic results.

## Setup (First Time)

### 1. Install Dependencies

```bash
pip install torch torchvision tenseal numpy pillow flask flask-cors tqdm requests
```

### 2. Retrain Model (FHE-Compatible)

**IMPORTANT:** The new FHE implementation requires a model without ReLU activation.

```bash
cd cloud_server
python train_model_fhe_compatible.py
```

**Expected output:**
```
✅ FHE-compatible model trained successfully.
✅ BatchNorm folded - zero inference accuracy loss.
✅ Architecture matches HE inference exactly.
```

**Time:** ~20-30 minutes (depending on hardware)

### 3. Verify FHE Security

```bash
python test_true_fhe.py
```

**Expected output:**
```
🎉 ALL TESTS PASSED - TRUE FHE IMPLEMENTATION VERIFIED
✅ Server CANNOT decrypt (no secret key)
✅ Server-side code has NO .decrypt() calls
✅ FHE inference matches plaintext
```

If any test fails, **DO NOT DEPLOY** - contact support.

---

## Running the Server

```bash
python app.py
```

**Server will start on:** `http://0.0.0.0:7860`

---

## Using True FHE Mode

### Option A: Command-Line Client (Recommended for Testing)

```bash
python cloud_server/client_pipeline.py \
    cloud_server/models/best_model.pth \
    path/to/chest_xray.jpg
```

**What this does:**
1. ✅ Extracts features on YOUR device (server never sees raw image)
2. ✅ Encrypts features with YOUR secret key (never transmitted)
3. ✅ Sends ONLY ciphertext to server (~326 KB)
4. ✅ Receives encrypted result
5. ✅ Decrypts on YOUR device

**Example output:**
```
[Client] Starting secure diagnosis pipeline...
──────────────────────────────────────────────────
[Client] Step 1: Extracting features locally...
[Client] Features extracted: shape=(512,)
[Client] Step 2: Encrypting with CKKS...
[Client] Encrypted → 326 KB ciphertext
[Client] Step 3: Sending ciphertext to server...
[Client] Server returned 48 KB encrypted result
[Client] Step 4: Decrypting result locally...
[Client] Decrypted: Pneumonia (94.23%)
[Client] Done in 623ms
──────────────────────────────────────────────────

═══════════════════════════════════════════════════════
RESULT
═══════════════════════════════════════════════════════
  Prediction  : Pneumonia
  Confidence  : 94.23%
  Normal      : 0.0577
  Pneumonia   : 0.9423
  Latency     : 623ms
  Server saw  : Ciphertext only — zero plaintext
```

### Option B: Web UI (Demo Mode - NOT True FHE)

**⚠️ WARNING:** Web UI runs in demo mode because browsers cannot run TenSEAL (Python library).

1. Open browser: `http://localhost:7860`
2. Upload X-ray image
3. View results

**Security Note:** In demo mode, the server sees the raw image. This is clearly labeled in all responses. **DO NOT use demo mode for real patient data.**

### Option C: Python API (Production Use)

```python
#!/usr/bin/env python3
"""
Example: True FHE diagnosis from Python
"""
import sys
import requests
import base64
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

# Add project to path
sys.path.insert(0, '/path/to/SecureLens')
from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.train_model import SecureLensNet

# Configuration
SERVER_URL = "http://localhost:7860/api/predict_encrypted"
MODEL_PATH = "cloud_server/models/best_model.pth"
IMAGE_PATH = "path/to/xray.jpg"

# Step 1: Initialize CKKS (client side)
print("[Client] Initializing encryption...")
ckks = CKKSEngine(
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
    global_scale=2**40
)

# Step 2: Load model (client side)
print("[Client] Loading ResNet-18...")
model = SecureLensNet(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# Step 3: Preprocess image (client side)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open(IMAGE_PATH).convert('RGB')
img_t = transform(img).unsqueeze(0)

# Step 4: Extract features (client side)
print("[Client] Extracting features...")
with torch.no_grad():
    features = model.get_backbone_features(img_t)
features_np = features.squeeze().numpy()

# Step 5: Encrypt features (client side)
print("[Client] Encrypting features...")
enc_features = ckks.encrypt_feature_vector(features_np)
ct_bytes = enc_features.serialize()
print(f"[Client] Ciphertext size: {len(ct_bytes)//1024} KB")

# Step 6: Send to server
print("[Client] Sending ciphertext to server...")
response = requests.post(
    SERVER_URL,
    files={'ciphertext': ('features.bin', ct_bytes, 'application/octet-stream')},
    timeout=30
)
response.raise_for_status()
data = response.json()

# Step 7: Decrypt result (client side)
print("[Client] Decrypting result...")
result_b64 = data['encrypted_result_b64']
result_bytes = base64.b64decode(result_b64)
result = ckks.decrypt_prediction_from_bytes(result_bytes)

# Step 8: Display result
print("\n" + "="*60)
print("DIAGNOSIS RESULT")
print("="*60)
print(f"  Prediction  : {result['prediction']}")
print(f"  Confidence  : {result['confidence']:.2%}")
print(f"  Normal      : {result['normal_score']:.4f}")
print(f"  Pneumonia   : {result['pneumonia_score']:.4f}")
print(f"  Server saw  : Ciphertext only (TRUE FHE)")
print("="*60)
```

---

## Verifying True FHE

### Check 1: Server Has No Secret Key

```python
from crypto_layer.ckks_engine import CKKSEngine

ckks = CKKSEngine(8192, [60,40,40,60], 2**40)
features = np.random.randn(512)
enc = ckks.encrypt_feature_vector(features)

# Server gets public context
pub_ctx = ckks.public_context

# Try to decrypt on "server"
enc_server = pub_ctx.ckks_vector_from(enc.serialize())
try:
    enc_server.decrypt()
    print("❌ BROKEN: Server can decrypt!")
except RuntimeError:
    print("✅ SECURE: Server cannot decrypt")
```

### Check 2: No Plaintext in Server Logs

Start server and make a request, then check logs:

```bash
python app.py 2>&1 | grep -i "plaintext\|decrypt"
```

**Expected:** Only logs about "ciphertext", "encrypted", or "FHE"  
**NOT Expected:** Any actual numerical values or "decrypted" messages

### Check 3: Network Traffic Analysis

```bash
# Capture traffic during FHE inference
tcpdump -i lo -X port 7860 > traffic.txt

# Search for plaintext leakage
grep -E "array|tensor|[-+]?[0-9]*\.?[0-9]+" traffic.txt
```

You should only see binary ciphertext, NOT readable numbers.

---

## Performance Expectations

| Operation | Time | Size |
|-----------|------|------|
| Feature extraction (client) | ~50ms | 512 floats (4 KB) |
| Encryption (client) | ~80ms | 326 KB ciphertext |
| Network upload | ~100ms | 326 KB |
| HE inference (server) | ~400ms | Encrypted ops |
| Network download | ~50ms | 48 KB |
| Decryption (client) | ~20ms | 2 logits |
| **Total** | **~700ms** | **374 KB total** |

**Note:** Server inference time increased from ~500ms (broken FHE with decrypt) to ~400ms (true FHE). The previous implementation was slower due to decrypt + re-encrypt operations.

---

## Troubleshooting

### Error: "Model not loaded"

**Cause:** Weights not found

**Fix:**
```bash
python cloud_server/train_model_fhe_compatible.py
```

### Error: "Secret key missing"

**Cause:** Using public context instead of full context on client

**Fix:** Client must create full CKKSEngine:
```python
ckks = CKKSEngine(8192, [60,40,40,60], 2**40)  # Has secret key
```

### Error: "Accuracy dropped after FHE"

**Cause:** Model was trained with ReLU, but inference has no ReLU

**Fix:** Retrain with FHE-compatible script:
```bash
python cloud_server/train_model_fhe_compatible.py
```

### Warning: "Ciphertext too small"

**Cause:** Sending plaintext instead of ciphertext

**Fix:** Encrypt features before sending:
```python
enc = ckks.encrypt_feature_vector(features)
ct_bytes = enc.serialize()  # Send this, not features
```

---

## Security FAQs

### Q: Can the server decrypt my X-ray images?
**A:** No. The server only receives encrypted feature vectors (~326 KB of ciphertext), never the raw image.

### Q: Can the server decrypt the diagnostic result?
**A:** No. The server returns encrypted logits. Only the client (with the secret key) can decrypt.

### Q: What if the server is hacked?
**A:** The attacker gets ciphertext only. Without the secret key (which never leaves the client), they cannot decrypt.

### Q: What if someone intercepts the network traffic?
**A:** They capture ciphertext. CKKS provides 128-bit security - computationally infeasible to break.

### Q: Is the demo mode secure?
**A:** NO. Demo mode (`/api/predict`) sends the raw image to the server. Use only for testing, never for real patient data.

### Q: What about the model weights? Are they secret?
**A:** No. The model weights are NOT sensitive - they're just the trained neural network. FHE protects the DATA (patient images), not the model.

---

## Production Checklist

Before deploying to production:

- [ ] Retrain model with `train_model_fhe_compatible.py`
- [ ] Run `test_true_fhe.py` - all tests must pass
- [ ] Verify server logs show no plaintext
- [ ] Test with real X-ray images
- [ ] Measure accuracy on test set
- [ ] Configure HTTPS for server (encrypt network traffic)
- [ ] Disable demo endpoints (or add authentication)
- [ ] Set up monitoring for ciphertext/plaintext leaks
- [ ] Document key management policy
- [ ] Train staff on FHE vs demo mode
- [ ] Get security audit from third party (recommended)

---

## Support

**Documentation:**
- Full audit: `SECURITY_AUDIT.md`
- Fixes applied: `FHE_FIXES_APPLIED.md`
- Architecture: `CONTRIBUTING.md`

**Testing:**
- Verification suite: `test_true_fhe.py`
- Unit tests: `cloud_server/encrypted_inference/he_inference.py` (run as script)

**Questions:**
If you encounter issues or have questions about the FHE implementation, review the audit documents first. All critical security decisions are explained there.

---

## Next Steps

1. **Run verification tests** to confirm your setup
2. **Try the command-line client** with a sample X-ray
3. **Review the audit report** to understand the security model
4. **Plan your production deployment** using the checklist above

**Remember:** The `/api/predict_encrypted` endpoint is TRUE FHE. The `/api/predict` endpoint is DEMO ONLY.

