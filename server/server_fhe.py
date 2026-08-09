"""
SecureLens — TRUE FHE Inference Server
=======================================

This server ONLY performs homomorphic computation on encrypted data.
It NEVER decrypts anything.
It has NO secret key — mathematically impossible to decrypt.

The client (Gradio Space) sends:
  1. ciphertext bytes  — encrypted 512-dim feature vector
  2. public_context    — CKKS context with NO secret key (safe to share)

This server computes:
  Layer 1: W1 @ enc(features) + b1   (homomorphic)
  Layer 2: W2 @ enc(h)       + b2   (homomorphic)

Returns: encrypted logits (client decrypts these — not us)

TRUE FHE: Server sees ZERO plaintext, has ZERO secret key.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tenseal as ts
import numpy as np
import json
import os
import base64
import struct
import time

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load plaintext model weights at startup ──────────────────────────────
# Weights are NOT sensitive — they are the model, not patient data.
# Only the patient's encrypted features are sensitive.

print("[SecureLens-Server] Loading trained weight matrices...")

def load_weights():
    fw_path = os.path.join(BASE_DIR, "feature_weights.json")
    lw_path = os.path.join(BASE_DIR, "linear_weights.json")

    if not os.path.exists(fw_path):
        raise FileNotFoundError(f"feature_weights.json not found at {fw_path}")
    if not os.path.exists(lw_path):
        raise FileNotFoundError(f"linear_weights.json not found at {lw_path}")

    with open(fw_path) as f:
        fw = json.load(f)
    with open(lw_path) as f:
        lw = json.load(f)

    W1 = np.array(fw["W"], dtype=np.float64)   # (256, 512)
    b1 = np.array(fw["b"], dtype=np.float64)   # (256,)
    W2 = np.array(lw["W"], dtype=np.float64)   # (2, 256)
    b2 = np.array(lw["b"], dtype=np.float64)   # (2,)

    print(f"  W1: {W1.shape}  b1: {b1.shape}")
    print(f"  W2: {W2.shape}  b2: {b2.shape}")
    print("[SecureLens-Server] Weights ready. Server has NO secret key.")
    return W1, b1, W2, b2

W1, b1, W2, b2 = load_weights()


# ── Homomorphic Linear Layer ──────────────────────────────────────────────

def linear_he(enc_vec, W, b, layer_name="layer"):
    """
    Computes W @ enc_vec + b HOMOMORPHICALLY.
    
    - enc_vec is a CKKSVector (ciphertext)
    - W, b are plaintext weights
    - Result is a LIST of CKKSVectors — one per output neuron
    - .decrypt() is NEVER called here
    """
    results = []
    for i in range(W.shape[0]):
        enc_dot    = enc_vec.dot(W[i].tolist())   # homomorphic dot product
        enc_neuron = enc_dot + b[i]               # homomorphic bias add
        results.append(enc_neuron)
    print(f"[Server] {layer_name}: {W.shape[1]}→{W.shape[0]} neurons (ENCRYPTED)")
    return results


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service"         : "SecureLens FHE Inference Server",
        "status"          : "running",
        "has_secret_key"  : False,
        "can_decrypt"     : False,
        "endpoint"        : "/api/predict_encrypted",
        "description"     : "Send encrypted features, receive encrypted logits. True FHE.",
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"         : "ok",
        "weights_loaded" : W1 is not None,
        "W1_shape"       : list(W1.shape),
        "W2_shape"       : list(W2.shape),
        "has_secret_key" : False,
    })


@app.route("/api/predict_encrypted", methods=["POST"])
def predict_encrypted():
    """
    TRUE FHE ENDPOINT — Server sees ZERO plaintext.

    Client sends (multipart/form-data):
      - ciphertext     : binary — encrypted 512-dim feature vector
      - public_context : binary — CKKS public context (no secret key)

    Server does:
      1. Reconstruct public context (cannot decrypt anything)
      2. Deserialize ciphertext using public context
      3. Layer1: W1 @ enc_features + b1  (homomorphic)
      4. Layer2: W2 @ enc_h + b2         (homomorphic)
      5. Serialize encrypted logits
      6. Return as base64 JSON

    Server NEVER calls .decrypt()
    Server NEVER sees plaintext features
    """
    if "ciphertext" not in request.files:
        return jsonify({"error": "Missing 'ciphertext' field. Send encrypted feature vector."}), 400
    if "public_context" not in request.files:
        return jsonify({"error": "Missing 'public_context' field. Send your CKKS public context bytes."}), 400

    try:
        ct_bytes      = request.files["ciphertext"].read()
        pub_ctx_bytes = request.files["public_context"].read()
        ct_size_kb    = len(ct_bytes) / 1024

        if ct_size_kb < 50:
            return jsonify({
                "error": f"Ciphertext too small ({ct_size_kb:.1f} KB). Expected ~326 KB CKKS ciphertext."
            }), 400

        t_start = time.time()
        print(f"\n[Server] Received {ct_size_kb:.1f} KB ciphertext — TRUE FHE inference starting...")
        print("[Server] Server has NO secret key — cannot decrypt.")

        # Step 1: Reconstruct public context from client-sent bytes
        # This context has NO secret key — mathematically cannot decrypt
        context = ts.context_from(pub_ctx_bytes)
        print("[Server] Public context reconstructed (no secret key inside)")

        # Step 2: Deserialize ciphertext using public context
        enc_features = ts.ckks_vector_from(context, ct_bytes)
        print("[Server] Ciphertext deserialized — still encrypted")

        # Step 3: Homomorphic Layer 1 — W1 @ enc_features + b1
        enc_h_list = linear_he(enc_features, W1, b1, "Layer1(512→256)")

        # Step 4: Homomorphic Layer 2 — W2 @ enc_h + b2
        enc_out_list = []
        for i in range(W2.shape[0]):   # 2 output neurons
            enc_sum = enc_h_list[0] * W2[i, 0]
            for j in range(1, len(enc_h_list)):
                enc_sum = enc_sum + (enc_h_list[j] * W2[i, j])
            enc_out_list.append(enc_sum + b2[i])
        print("[Server] Layer2(256→2): 2 encrypted logits computed (ENCRYPTED)")

        # Step 5: Serialize encrypted results
        # Format: [n_vecs(4B)][size1(4B)][vec1][size2(4B)][vec2]
        result_bytes = struct.pack('!I', len(enc_out_list))
        for vec in enc_out_list:
            vec_bytes    = vec.serialize()
            result_bytes += struct.pack('!I', len(vec_bytes))
            result_bytes += vec_bytes

        latency_ms = (time.time() - t_start) * 1000
        print(f"[Server] Done in {latency_ms:.0f}ms. Returning {len(result_bytes)//1024} KB encrypted result.")
        print("[Server] Server decrypted: NOTHING — true FHE ✓")

        return jsonify({
            "success"              : True,
            "mode"                 : "TRUE_FHE",
            "encrypted_result_b64" : base64.b64encode(result_bytes).decode("utf-8"),
            "latency_ms"           : round(latency_ms, 1),
            "server_has_secret_key": False,
            "server_decrypted"     : False,
            "server_saw"           : "Ciphertext only — ZERO plaintext",
            "ciphertext_size_kb"   : round(ct_size_kb, 2),
            "result_size_kb"       : round(len(result_bytes) / 1024, 2),
            "pipeline"             : [
                "Client extracted ResNet-18 features (client device)",
                "Client CKKS-encrypted 512 features → 326 KB ciphertext (client device)",
                f"Server received {ct_size_kb:.0f} KB ciphertext ONLY",
                "Server deserialized with PUBLIC context (no secret key)",
                "Server computed W1 @ enc(x) + b1 homomorphically",
                "Server computed W2 @ enc(h) + b2 homomorphically",
                "Server returned ENCRYPTED logits (cannot decrypt)",
                "Client will decrypt with secret key (client device)",
            ],
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SecureLens — TRUE FHE Inference Server")
    print("  Server has NO secret key")
    print("  Server NEVER decrypts anything")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=7860, debug=False)
