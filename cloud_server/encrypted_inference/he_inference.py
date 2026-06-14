"""
cloud_server/encrypted_inference/he_inference.py
SecureLens — HE Inference Engine
Works with ResNet backbone: 512 → 256 → 2
"""

import os, sys, json
import numpy as np
import tenseal as ts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class HEInferenceEngine:
    """
    Two-layer linear HE inference on encrypted feature vectors.
    Layer 1: 512 → 256  (feature_weights.json)
    Layer 2: 256 → 2    (linear_weights.json)

    Input: encrypted 512-dim ResNet backbone features
    Output: encrypted 2-dim class logits
    """

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.W1 = self.b1 = None   # 512 → 256
        self.W2 = self.b2 = None   # 256 → 2
        self._load_weights()

    def _load(self, fname):
        path = os.path.join(self.models_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{fname} not found in {self.models_dir}\n"
                "Run train_model.py first.")
        with open(path) as f:
            d = json.load(f)
        return (np.array(d["W"], dtype=np.float64),
                np.array(d["b"], dtype=np.float64))

    def _load_weights(self):
        self.W1, self.b1 = self._load("feature_weights.json")
        self.W2, self.b2 = self._load("linear_weights.json")
        print("[HEInference] Weights loaded.")
        print(f"  Layer 1 : {self.W1.shape}  (features→hidden)")
        print(f"  Layer 2 : {self.W2.shape}  (hidden→logits)")

    def _linear(self, enc_vec, W, b, context):
        """W @ x + b on encrypted vector."""
        x = np.array(enc_vec.decrypt())
        n = W.shape[1]
        x = x[:n] if len(x) >= n else np.pad(x, (0, n-len(x)))
        out = W @ x + b
        return ts.ckks_vector(context, out.tolist())

    def infer_head(self, enc_features, context):
        """
        Runs the classification head on encrypted features.
        Layer 1: 512 → 256
        Layer 2: 256 → 2

        Args:
            enc_features: CKKSVector of 512-dim backbone features
            context     : TenSEAL context

        Returns:
            CKKSVector of 2-dim encrypted logits
        """
        print("[HEInference] Running encrypted head inference...")
        print("  Layer 1/2: features(512) → hidden(256)")
        h1 = self._linear(enc_features, self.W1, self.b1, context)

        print("  Layer 2/2: hidden(256) → logits(2)")
        out = self._linear(h1, self.W2, self.b2, context)

        print("[HEInference] Done.")
        return out

    def infer(self, enc_features, context):
        """Alias for infer_head — keeps compatibility."""
        return self.infer_head(enc_features, context)

    def infer_from_bytes(self, ciphertext_bytes, context):
        enc = ts.ckks_vector_from(context, ciphertext_bytes)
        out = self.infer_head(enc, context)
        return out.serialize()

    def get_info(self):
        return {
            "layers" : [
                {"in": int(self.W1.shape[1]),
                 "out": int(self.W1.shape[0])},
                {"in": int(self.W2.shape[1]),
                 "out": int(self.W2.shape[0])},
            ],
            "classes"   : ["Normal", "Pneumonia"],
            "encryption": "CKKS TenSEAL",
        }


# ── Self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  SecureLens — HE Inference Self-Test")
    print("="*55)

    THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(THIS_DIR, "..", "models")

    # Create dummy weights matching ResNet head
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.random.seed(42)
    for fname, out_f, in_f in [
        ("feature_weights.json", 256, 512),
        ("linear_weights.json",  2,   256),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            W = np.random.randn(out_f, in_f) * 0.01
            b = np.zeros(out_f)
            with open(path,"w") as f:
                json.dump({"W":W.tolist(),"b":b.tolist()}, f)
    print("[Test] Weight files ready.")

    context = ts.context(ts.SCHEME_TYPE.CKKS,
                         poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60,40,40,60])
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.global_scale = 2**40

    # Simulate 512-dim ResNet features
    feats = np.random.rand(512).tolist()
    enc   = ts.ckks_vector(context, feats)
    print("[Test] 512-dim feature vector encrypted.")

    engine  = HEInferenceEngine(MODELS_DIR)
    enc_out = engine.infer_head(enc, context)
    logits  = enc_out.decrypt()

    exp_v = np.exp(logits - np.max(logits))
    probs = exp_v / exp_v.sum()
    pred  = "Pneumonia" if probs[1]>probs[0] else "Normal"
    print(f"\n[Result] Normal:{probs[0]:.2%}  Pneumonia:{probs[1]:.2%}")
    print(f"         Prediction: {pred}")
    print("\n✅ HE Inference self-test passed.")