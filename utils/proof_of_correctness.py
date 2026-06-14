"""
utils/proof_of_correctness.py
SecureLens — Proof that FHE produces identical results to plaintext.
Run: python utils/proof_of_correctness.py
"""

import os, sys, json
import numpy as np
import tenseal as ts

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "cloud_server", "models")
DOCS_DIR   = os.path.join(
    os.path.dirname(__file__), "..", "docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def prove_correctness(n_tests=50):
    print("="*55)
    print("  SecureLens — Proof of Correctness")
    print("="*55)

    engine      = CKKSEngine(global_scale=2**40)
    he_engine   = HEInferenceEngine(MODELS_DIR)

    W1, b1 = he_engine.W1, he_engine.b1
    W2, b2 = he_engine.W2, he_engine.b2

    results = []
    np.random.seed(42)

    print(f"\n[Proof] Running {n_tests} test vectors...")
    for i in range(n_tests):
        feat = np.random.randn(512) * 0.5

        # Plaintext computation
        h1_plain  = W1 @ feat + b1
        out_plain = W2 @ h1_plain + b2

        # FHE computation
        enc_feat = ts.ckks_vector(engine.context, feat.tolist())
        enc_h1   = he_engine._linear(
            enc_feat, W1, b1, engine.context)
        enc_out  = he_engine._linear(
            enc_h1, W2, b2, engine.context)
        fhe_out  = np.array(enc_out.decrypt()[:2])

        # Compare
        error     = np.max(np.abs(fhe_plain := out_plain[:2])
                           - np.abs(fhe_out))
        error     = np.max(np.abs(out_plain[:2] - fhe_out))
        pred_plain = "Normal" \
            if out_plain[0] > out_plain[1] else "Pneumonia"
        pred_fhe   = "Normal" \
            if fhe_out[0] > fhe_out[1] else "Pneumonia"

        results.append({
            "test_id"    : i+1,
            "plain_out"  : out_plain[:2].tolist(),
            "fhe_out"    : fhe_out.tolist(),
            "max_error"  : float(error),
            "pred_match" : pred_plain == pred_fhe,
            "pred_plain" : pred_plain,
            "pred_fhe"   : pred_fhe,
        })

    # Summary
    errors      = [r["max_error"] for r in results]
    match_count = sum(1 for r in results if r["pred_match"])

    summary = {
        "total_tests"    : n_tests,
        "prediction_matches": match_count,
        "match_rate"     : f"{match_count/n_tests*100:.1f}%",
        "mean_error"     : f"{np.mean(errors):.2e}",
        "max_error"      : f"{np.max(errors):.2e}",
        "min_error"      : f"{np.min(errors):.2e}",
        "error_threshold": "1e-03",
        "all_pass"       : all(e < 1e-3 for e in errors),
        "ckks_params"    : {
            "poly_modulus_degree": 8192,
            "global_scale"       : "2^40",
            "security_bits"      : 128,
        }
    }

    # Save
    proof = {"summary": summary, "test_results": results[:10]}
    path  = os.path.join(DOCS_DIR, "proof_of_correctness.json")
    with open(path, "w") as f:
        json.dump(proof, f, indent=2)

    print(f"\n  Prediction match rate : {summary['match_rate']}")
    print(f"  Mean CKKS error       : {summary['mean_error']}")
    print(f"  Max  CKKS error       : {summary['max_error']}")
    print(f"  All errors < 1e-3     : {summary['all_pass']}")
    print(f"\n[Saved] → {path}")

    if summary["all_pass"] and match_count == n_tests:
        print("\n✅ PROOF COMPLETE: FHE produces identical results "
              "to plaintext computation.")
        print("   The CKKS approximation error is negligible "
              f"(max {summary['max_error']}) and does not")
        print("   affect diagnostic predictions.")
    else:
        print("\n⚠️  Some tests failed — check proof_of_correctness.json")

    return summary


if __name__ == "__main__":
    prove_correctness(n_tests=50)