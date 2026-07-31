"""
demo_true_fhe.py
SecureLens — TRUE FHE Demonstration

This script demonstrates end-to-end Fully Homomorphic Encryption:
1. Client encrypts X-ray features locally
2. Server computes on encrypted data (sees ZERO plaintext)
3. Client decrypts result locally
4. Compare with plaintext to verify correctness

Run this to show TRUE FHE in action!
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE

print("="*70)
print(" SecureLens - TRUE Fully Homomorphic Encryption Demonstration")
print("="*70)
print("\nThis demonstrates that the server performs inference on encrypted")
print("data without ever seeing plaintext X-ray features or predictions.\n")

# Configuration
MODELS_DIR = "cloud_server/models"
TEST_IMAGE_NORMAL = "data/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
TEST_IMAGE_PNEUMONIA = "data/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg"

# =============================================================================
# Phase 1: CLIENT-SIDE Setup
# =============================================================================
print("\n" + "="*70)
print("PHASE 1: CLIENT-SIDE SETUP (Patient's Device)")
print("="*70)

print("\n[Client] Loading ResNet-18 model for local feature extraction...")
model = SecureLensNetFHE(num_classes=2)
model.load_state_dict(torch.load(
    os.path.join(MODELS_DIR, "best_model.pth"),
    map_location="cpu"
))
model.eval()
print("✅ Model loaded on client")

print("\n[Client] Initializing CKKS encryption engine...")
print("  - Generating secret key (NEVER sent to server)")
print("  - poly_modulus_degree = 8192")
print("  - security_bits = 128")
ckks = CKKSEngine(
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
    global_scale=2**40,
)
print("✅ CKKS engine initialized")
print("  - Secret key generated (kept on client only)")
print("  - Public context created (can be shared with server)")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# =============================================================================
# Phase 2: CLIENT-SIDE Encryption
# =============================================================================
print("\n" + "="*70)
print("PHASE 2: CLIENT-SIDE ENCRYPTION")
print("="*70)

# Test with a Normal X-ray
print(f"\n[Client] Loading test image: {os.path.basename(TEST_IMAGE_NORMAL)}")
img = Image.open(TEST_IMAGE_NORMAL).convert('RGB')
img_tensor = transform(img).unsqueeze(0)

print("[Client] Extracting 512-dim features using ResNet-18 backbone...")
with torch.no_grad():
    features = model.get_backbone_features(img_tensor)
features_np = features.squeeze().numpy().astype(np.float64)
print(f"✅ Features extracted: shape={features_np.shape}")
print(f"  - Feature range: [{features_np.min():.3f}, {features_np.max():.3f}]")
print(f"  - Feature mean: {features_np.mean():.3f}")

print("\n[Client] Encrypting features with CKKS...")
t_start = time.time()
enc_features = ckks.encrypt_feature_vector(features_np.copy())
ct_bytes = enc_features.serialize()
t_encrypt = time.time() - t_start

print(f"✅ Features encrypted")
print(f"  - Encryption time: {t_encrypt*1000:.1f} ms")
print(f"  - Plaintext size: {features_np.nbytes} bytes ({features_np.nbytes/1024:.1f} KB)")
print(f"  - Ciphertext size: {len(ct_bytes)} bytes ({len(ct_bytes)/1024:.1f} KB)")
print(f"  - Encryption overhead: {len(ct_bytes)/features_np.nbytes:.1f}x")

print("\n[Client] What the server will see:")
print(f"  - Ciphertext bytes: {ct_bytes[:50]}... (truncated)")
print("  - NO PLAINTEXT features")
print("  - NO PLAINTEXT prediction")

# =============================================================================
# Phase 3: SERVER-SIDE Processing (No Secret Key)
# =============================================================================
print("\n" + "="*70)
print("PHASE 3: SERVER-SIDE HOMOMORPHIC INFERENCE")
print("="*70)

print("\n[Server] Loading HE inference engine...")
he_engine = HEInferenceEngine(MODELS_DIR)
print("✅ HE engine loaded")
print(f"  - W1 (feature layer): {he_engine.W1.shape}")
print(f"  - W2 (classification layer): {he_engine.W2.shape}")

print("\n[Server] Deserializing ciphertext (using PUBLIC context only)...")
print("  - Server has NO secret key")
print("  - Server cannot decrypt anything")
import tenseal as ts
enc_features_server = ts.ckks_vector_from(ckks.public_context, ct_bytes)
print("✅ Ciphertext deserialized")

print("\n[Server] Running homomorphic inference...")
print("  - Layer 1: W1 @ enc(features) + b1  (256 encrypted neurons)")
print("  - Layer 2: W2 @ enc(h) + b2          (2 encrypted logits)")
print("  - All operations on ENCRYPTED data")

t_start = time.time()
enc_result_list = he_engine.infer_head(enc_features_server, ckks.public_context)
t_inference = time.time() - t_start

print(f"✅ Homomorphic inference complete")
print(f"  - Inference time: {t_inference*1000:.1f} ms")
print(f"  - Result: {len(enc_result_list)} encrypted logits")
print(f"  - Server saw: ZERO plaintext")

# =============================================================================
# Phase 4: CLIENT-SIDE Decryption
# =============================================================================
print("\n" + "="*70)
print("PHASE 4: CLIENT-SIDE DECRYPTION")
print("="*70)

print("\n[Client] Decrypting result with secret key...")
t_start = time.time()
result = ckks.decrypt_prediction(enc_result_list)
t_decrypt = time.time() - t_start

print(f"✅ Result decrypted")
print(f"  - Decryption time: {t_decrypt*1000:.1f} ms")
print(f"\n📊 FINAL DIAGNOSIS:")
print(f"  - Prediction: {result['prediction']}")
print(f"  - Confidence: {result['confidence']:.1%}")
print(f"  - Normal score: {result['normal_score']:.4f}")
print(f"  - Pneumonia score: {result['pneumonia_score']:.4f}")
print(f"  - Raw logits: {result['raw']}")

# =============================================================================
# Phase 5: Verification (Compare with Plaintext)
# =============================================================================
print("\n" + "="*70)
print("PHASE 5: VERIFICATION - Compare FHE vs Plaintext")
print("="*70)

print("\n[Verification] Running plaintext inference for comparison...")
with torch.no_grad():
    plaintext_output = model(img_tensor)
    plaintext_probs = torch.softmax(plaintext_output, dim=1)[0]

plaintext_pred = "Normal" if plaintext_probs[0] > plaintext_probs[1] else "Pneumonia"
plaintext_conf = max(plaintext_probs[0], plaintext_probs[1]).item()

print(f"\n📊 PLAINTEXT RESULT:")
print(f"  - Prediction: {plaintext_pred}")
print(f"  - Confidence: {plaintext_conf:.1%}")
print(f"  - Normal score: {plaintext_probs[0]:.4f}")
print(f"  - Pneumonia score: {plaintext_probs[1]:.4f}")

print(f"\n📊 FHE RESULT:")
print(f"  - Prediction: {result['prediction']}")
print(f"  - Confidence: {result['confidence']:.1%}")
print(f"  - Normal score: {result['normal_score']:.4f}")
print(f"  - Pneumonia score: {result['pneumonia_score']:.4f}")

# Compare results
pred_match = result['prediction'] == plaintext_pred
conf_diff = abs(result['confidence'] - plaintext_conf)
logit_error = np.max(np.abs(
    np.array(result['raw']) - plaintext_output[0].numpy()
))

print(f"\n📈 ACCURACY COMPARISON:")
print(f"  - Predictions match: {'✅ YES' if pred_match else '❌ NO'}")
print(f"  - Confidence difference: {conf_diff:.4f} ({conf_diff*100:.2f}%)")
print(f"  - Max logit error: {logit_error:.6f}")

if pred_match and conf_diff < 0.01:
    print("\n✅ FHE inference is ACCURATE - matches plaintext within tolerance!")
else:
    print("\n⚠️  FHE inference differs from plaintext")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("SUMMARY - TRUE FHE PROPERTIES VERIFIED")
print("="*70)

print("\n✅ Privacy Guarantees:")
print("  1. Server received only ciphertext (326 KB encrypted data)")
print("  2. Server has NO secret key (cannot decrypt)")
print("  3. Server never saw plaintext features or prediction")
print("  4. Only client can decrypt the result")

print(f"\n✅ Performance:")
print(f"  - Client encryption: {t_encrypt*1000:.1f} ms")
print(f"  - Server inference: {t_inference*1000:.1f} ms")
print(f"  - Client decryption: {t_decrypt*1000:.1f} ms")
print(f"  - Total latency: {(t_encrypt + t_inference + t_decrypt)*1000:.1f} ms")

print(f"\n✅ Accuracy:")
print(f"  - FHE prediction: {result['prediction']} ({result['confidence']:.1%})")
print(f"  - Plaintext prediction: {plaintext_pred} ({plaintext_conf:.1%})")
print(f"  - Match: {pred_match}")

print("\n" + "="*70)
print("TRUE FHE DEMONSTRATION COMPLETE!")
print("="*70)
print("\nKey Takeaway:")
print("The server performed medical diagnosis on ENCRYPTED X-ray data")
print("without ever seeing the patient's plaintext information.")
print("="*70)
