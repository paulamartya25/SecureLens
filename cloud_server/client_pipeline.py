"""
client/client_pipeline.py
SecureLens — TRUE Client-Side Pipeline

This is what runs on the PATIENT'S device (laptop, phone, hospital workstation).
It does:
  1. Load and preprocess the X-ray
  2. Extract 512-dim features using ResNet-18 backbone (LOCAL)
  3. Encrypt the features using CKKS (LOCAL)
  4. Send ONLY the ciphertext to the server
  5. Receive encrypted result from server
  6. Decrypt locally using secret key

THE SERVER NEVER SEES PLAINTEXT AT ANY POINT.
"""

import torch
import numpy as np
import tenseal as ts
import requests
import base64
import io
import os
import sys
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from crypto_layer.ckks_engine import CKKSEngine


class SecureLensClient:
    """
    Full client-side pipeline for SecureLens.
    Runs entirely on the patient's device.
    """

    def __init__(
        self,
        model_path: str,
        server_url: str = "http://127.0.0.1:7860",
    ):
        self.server_url = server_url

        # Step 1: Initialize CKKS engine (generates keys on client)
        print("[Client] Initializing CKKS engine...")
        self.ckks = CKKSEngine(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            global_scale=2**40,
        )

        # Step 2: Load ResNet-18 backbone (feature extractor)
        print("[Client] Loading ResNet-18 backbone...")
        self.backbone = self._load_backbone(model_path)

        # Step 3: Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        print("[Client] Ready. Secret key on client only.")

    def _load_backbone(self, model_path: str):
        """Loads ResNet-18 with classification head removed."""
        from cloud_server.train_model import SecureLensNet
        model = SecureLensNet(num_classes=2)
        if os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    # ── Step A: Extract Features (CLIENT SIDE) ─────────────────────────

    def extract_features(self, img_bytes: bytes) -> np.ndarray:
        """
        Runs ResNet-18 backbone on the X-ray IMAGE.
        Produces 512-dim feature vector.

        RUNS ENTIRELY ON CLIENT DEVICE.
        Raw image never leaves this function.

        Args:
            img_bytes: raw image bytes (PNG/JPG)
        Returns:
            numpy array shape (512,) — plaintext features
        """
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_t   = self.transform(img_pil).unsqueeze(0)  # (1,3,224,224)

        with torch.no_grad():
            features = self.backbone.get_backbone_features(img_t)

        feats = features.squeeze().numpy()  # (512,)
        print(f"[Client] Features extracted: shape={feats.shape}")
        return feats

    # ── Step B: Encrypt Features (CLIENT SIDE) ─────────────────────────

    def encrypt_features(self, features: np.ndarray) -> bytes:
        """
        CKKS-encrypts the 512-dim feature vector.

        RUNS ENTIRELY ON CLIENT DEVICE.
        Returns ciphertext bytes ready for server.

        Args:
            features: numpy array (512,)
        Returns:
            bytes: serialized CKKS ciphertext (~326 KB)
        """
        ct_bytes = self.ckks.encrypt_feature_vector_to_bytes(features)

        print(f"[Client] Encrypted → {len(ct_bytes)//1024} KB ciphertext")
        return ct_bytes

    # ── Step C: Send Ciphertext to Server ──────────────────────────────

    def send_to_server(self, ct_bytes: bytes) -> bytes:
        """
        Sends encrypted ciphertext to server.
        Server receives ONLY this — no raw image, no plaintext features.

        Args:
            ct_bytes: serialized CKKS ciphertext
        Returns:
            bytes: server's encrypted result
        """
        files    = {"ciphertext": ("ct.bin", ct_bytes, "application/octet-stream")}
        response = requests.post(
            f"{self.server_url}/api/predict_encrypted",
            files=files,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise ValueError(data["error"])

        # Server returns encrypted result as base64
        result_b64 = data["encrypted_result_b64"]
        result_bytes = base64.b64decode(result_b64)
        print(f"[Client] Server returned {len(result_bytes)//1024} KB encrypted result")
        return result_bytes

    # ── Step D: Decrypt Result (CLIENT SIDE) ───────────────────────────

    def decrypt_result(self, enc_result_bytes: bytes) -> dict:
        """
        Decrypts server's encrypted response.

        RUNS ENTIRELY ON CLIENT DEVICE.
        Uses secret key — which never left the client.

        Args:
            enc_result_bytes: encrypted logits from server
        Returns:
            dict: prediction, confidence, scores
        """
        result = self.ckks.decrypt_prediction_from_bytes(enc_result_bytes)
        print(f"[Client] Decrypted: {result['prediction']} "
              f"({result['confidence']:.2%})")
        return result

    # ── Full Pipeline ───────────────────────────────────────────────────

    def diagnose(self, img_bytes: bytes) -> dict:
        """
        Full end-to-end SECURE pipeline:
          raw image → features (client) → encrypt (client)
          → ciphertext to server → HE inference (server)
          → encrypted result to client → decrypt (client)
          → diagnosis

        SERVER NEVER SEES PLAINTEXT.

        Args:
            img_bytes: raw X-ray image bytes
        Returns:
            dict: prediction, confidence, scores, pipeline_steps
        """
        import time
        t0 = time.time()

        print("\n[Client] Starting secure diagnosis pipeline...")
        print("─" * 50)

        # A: Extract features locally
        print("[Client] Step 1: Extracting features locally...")
        features = self.extract_features(img_bytes)

        # B: Encrypt locally
        print("[Client] Step 2: Encrypting with CKKS...")
        ct_bytes = self.encrypt_features(features.copy())

        # C: Send ciphertext only
        print("[Client] Step 3: Sending ciphertext to server...")
        enc_result_bytes = self.send_to_server(ct_bytes)

        # D: Decrypt locally
        print("[Client] Step 4: Decrypting result locally...")
        result = self.decrypt_result(enc_result_bytes)

        elapsed = (time.time() - t0) * 1000
        print(f"[Client] Done in {elapsed:.0f}ms")
        print("─" * 50)

        result["latency_ms"]     = round(elapsed, 1)
        result["pipeline_steps"] = [
            "X-ray loaded on client device",
            "ResNet-18 extracted 512 features (client)",
            "CKKS encrypted features → 326 KB ciphertext (client)",
            "Ciphertext sent to cloud server",
            "Server ran W1@enc(x)+b1, W2@enc(h)+b2 on ciphertext",
            "Server returned encrypted logits",
            "Client decrypted with secret key",
        ]
        result["server_saw"]     = "Ciphertext only — zero plaintext"
        result["encryption_info"] = self.ckks.get_encryption_info()

        return result


# ── Command-line test ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python client_pipeline.py <model_path> <image_path>")
        print("Example: python client_pipeline.py cloud_server/models/best_model.pth test.jpg")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    print("=" * 55)
    print("SecureLens — Client Pipeline Test")
    print("=" * 55)

    client = SecureLensClient(
        model_path=model_path,
        server_url="http://127.0.0.1:7860",
    )

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    result = client.diagnose(img_bytes)

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print(f"  Normal      : {result['normal_score']:.4f}")
    print(f"  Pneumonia   : {result['pneumonia_score']:.4f}")
    print(f"  Latency     : {result['latency_ms']}ms")
    print(f"  Server saw  : {result['server_saw']}")