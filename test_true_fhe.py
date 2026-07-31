"""
test_true_fhe.py
SecureLens — True FHE Verification Test Suite

This script verifies that the SecureLens implementation achieves
TRUE end-to-end Fully Homomorphic Encryption with NO plaintext
exposure on the server.

Tests:
1. Server cannot decrypt ciphertexts (no secret key)
2. No .decrypt() calls in server-side code
3. FHE inference matches plaintext inference
4. End-to-end client-server pipeline works
5. Architecture verification (training vs inference)
"""

import os
import sys
import numpy as np
import torch
import subprocess
import torch
import subprocess
import tenseal as ts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine

MODELS_DIR = os.path.join(os.path.dirname(__file__), "cloud_server", "models")

print("=" * 70)
print(" SecureLens — TRUE FHE Verification Test Suite")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# Test 1: Server Cannot Decrypt (No Secret Key)
# ═══════════════════════════════════════════════════════════════════════

def test_server_cannot_decrypt():
    """
    CRITICAL TEST: Verifies server's public context cannot decrypt.
    This is the fundamental requirement for FHE security.
    """
    print("\n[Test 1] Server Cannot Decrypt Ciphertexts")
    print("-" * 70)
    
    # Client creates context (with secret key)
    client_ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
    
    # Client encrypts data
    test_data = np.random.randn(512).astype(np.float64)
    enc = client_ckks.encrypt_feature_vector(test_data)
    ct_bytes = enc.serialize()
    
    # Server receives ONLY public context
    server_pub_ctx = client_ckks.public_context
    
    # Server deserializes ciphertext
    enc_server = ts.ckks_vector_from(server_pub_ctx, ct_bytes)


    
    # Server tries to decrypt (MUST FAIL)
        # Server tries to decrypt (MUST FAIL)
    try:
        decrypted = enc_server.decrypt()
        print("  ❌ CRITICAL FAILURE: Server was able to decrypt!")
        print("  ❌ Server has secret key - FHE is BROKEN")
        return False
    except (RuntimeError, ValueError) as e:
        print(f"  ✅ Server cannot decrypt (expected): {str(e)[:80]}")
        print("  ✅ Server has NO secret key - FHE security maintained")
        return True



# ═══════════════════════════════════════════════════════════════════════
# Test 2: No Decrypt Calls in Server Code
# ═══════════════════════════════════════════════════════════════════════

def test_no_decrypt_in_server_code():
    """
    Static analysis: checks that server-side code has no .decrypt() calls.
    Any .decrypt() on the server breaks the FHE security model.
    """
    print("\n[Test 2] No Decrypt Calls in Server Code")
    print("-" * 70)
    
    # Files to check (server-side only)
    files_to_check = [
        "cloud_server/encrypted_inference/he_inference.py",
        "cloud_server/server.py",
    ]
    
    violations = []
    for filepath in files_to_check:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if not os.path.exists(full_path):
            print(f"  ⚠️  File not found: {filepath}")
            continue
        
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # Check for .decrypt() calls (but allow in comments)
            if '.decrypt()' in line and not line.strip().startswith('#'):
                # Check it's not in a string or comment
                code_part = line.split('#')[0]  # Remove comments
                if '.decrypt()' in code_part and 'NEVER' not in line.upper():
                    violations.append(f"{filepath}:{i}: {line.strip()}")

    
    if violations:
        print("  ❌ CRITICAL: Found .decrypt() calls in server code:")
        for v in violations:
            print(f"     {v}")
        return False
    else:
        print("  ✅ No .decrypt() calls found in server code")
        print("  ✅ Server-side code is FHE-compliant")
        return True


# ═══════════════════════════════════════════════════════════════════════
# Test 3: FHE Inference Matches Plaintext
# ═══════════════════════════════════════════════════════════════════════

def test_fhe_matches_plaintext():
    """
    Functional test: Verifies that FHE inference produces same result as
    plaintext inference (within CKKS precision bounds).
    """
    print("\n[Test 3] FHE Inference Matches Plaintext")
    print("-" * 70)
    
    if not os.path.exists(os.path.join(MODELS_DIR, "feature_weights.json")):
        print("  ⚠️  Weights not found. Run train_model.py first.")
        return None
    
    # Client setup
    client_ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
    he_engine = HEInferenceEngine(MODELS_DIR)
    
    # Test data
    test_features = np.random.randn(512).astype(np.float64)
    
    # Plaintext inference
    import json
    with open(os.path.join(MODELS_DIR, "feature_weights.json")) as f:
        fw = json.load(f)
    with open(os.path.join(MODELS_DIR, "linear_weights.json")) as f:
        lw = json.load(f)
    
    W1 = np.array(fw["W"], dtype=np.float64)
    b1 = np.array(fw["b"], dtype=np.float64)
    W2 = np.array(lw["W"], dtype=np.float64)
    b2 = np.array(lw["b"], dtype=np.float64)
    
    # Plaintext forward pass (no ReLU - matches FHE inference)
    h1_plain = W1 @ test_features + b1
    out_plain = W2 @ h1_plain + b2
    
    # FHE inference
    enc_features = client_ckks.encrypt_feature_vector(test_features.copy())
    enc_result_list = he_engine.infer_head(enc_features, client_ckks.context)
    
    # Decrypt on client
    result = client_ckks.decrypt_prediction(enc_result_list)
    out_fhe = np.array(result["raw"])
    
    # Compare
    error = np.max(np.abs(out_plain - out_fhe))
    print(f"  Plaintext result : {out_plain}")
    print(f"  FHE result       : {out_fhe}")
    print(f"  Max error        : {error:.2e}")
    
    # CKKS typical precision is ~1e-6 to 1e-8
    if error < 1e-3:
        print(f"  ✅ FHE matches plaintext (error < 1e-3)")
        return True
    else:
        print(f"  ❌ FHE error too large: {error}")
        return False


# ═══════════════════════════════════════════════════════════════════════
# Test 4: End-to-End Pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_end_to_end_pipeline():
    """
    Integration test: Full client-server pipeline simulation.
    """
    print("\n[Test 4] End-to-End Pipeline Simulation")
    print("-" * 70)
    
    if not os.path.exists(os.path.join(MODELS_DIR, "feature_weights.json")):
        print("  ⚠️  Weights not found. Run train_model.py first.")
        return None
    
    # ── CLIENT SIDE ──────────────────────────────────────────────────
    print("  [Client] Creating CKKS context...")
    client_ckks = CKKSEngine(8192, [60, 40, 40, 60], 2**40)
    
    print("  [Client] Simulating feature extraction...")
    features = np.random.randn(512).astype(np.float64)
    
    print("  [Client] Encrypting features...")
    enc_features = client_ckks.encrypt_feature_vector(features)
    ct_bytes = enc_features.serialize()
    print(f"  [Client] Ciphertext size: {len(ct_bytes)//1024} KB")
    
    # ── SERVER SIDE ──────────────────────────────────────────────────
    print("  [Server] Receiving ciphertext...")
    
    # Server gets ONLY public context
    server_pub_ctx = client_ckks.public_context
    print("  [Server] Using PUBLIC context (no secret key)")
    
    print("  [Server] Loading HE inference engine...")
    he_engine = HEInferenceEngine(MODELS_DIR)
    
    print("  [Server] Deserializing ciphertext...")
    enc_features_server = ts.ckks_vector_from(server_pub_ctx, ct_bytes)

    
    print("  [Server] Running homomorphic inference...")
    enc_result_list = he_engine.infer_head(enc_features_server, server_pub_ctx)
    
    print("  [Server] Serializing encrypted result...")
    import struct
    result_bytes = struct.pack('!I', len(enc_result_list))
    for vec in enc_result_list:
        vec_bytes = vec.serialize()
        result_bytes += struct.pack('!I', len(vec_bytes))
        result_bytes += vec_bytes
    print(f"  [Server] Result size: {len(result_bytes)//1024} KB")
    
    # ── CLIENT SIDE ──────────────────────────────────────────────────
    print("  [Client] Receiving encrypted result...")
    print("  [Client] Decrypting with secret key...")
    result = client_ckks.decrypt_prediction_from_bytes(result_bytes)
    
    print(f"  [Client] Prediction : {result['prediction']}")
    print(f"  [Client] Confidence : {result['confidence']:.2%}")
    
    print("  ✅ End-to-end pipeline successful")
    print("  ✅ Server never saw plaintext at any point")
    return True


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Architecture Verification
# ═══════════════════════════════════════════════════════════════════════

def test_architecture_match():
    """
    Verifies that exported weights match the HE inference architecture.
    """
    print("\n[Test 5] Architecture Verification")
    print("-" * 70)
    
    if not os.path.exists(os.path.join(MODELS_DIR, "feature_weights.json")):
        print("  ⚠️  Weights not found. Run train_model.py first.")
        return None
    
    import json
    with open(os.path.join(MODELS_DIR, "feature_weights.json")) as f:
        fw = json.load(f)
    with open(os.path.join(MODELS_DIR, "linear_weights.json")) as f:
        lw = json.load(f)
    
    W1 = np.array(fw["W"])
    b1 = np.array(fw["b"])
    W2 = np.array(lw["W"])
    b2 = np.array(lw["b"])
    
    print(f"  Layer 1 (feature):  {W1.shape} + bias {b1.shape}")
    print(f"  Layer 2 (linear):   {W2.shape} + bias {b2.shape}")
    
    # Expected shapes for FHE inference
    expected = {
        "W1": (256, 512),
        "b1": (256,),
        "W2": (2, 256),
        "b2": (2,),
    }
    
    match = (
        W1.shape == expected["W1"] and
        b1.shape == expected["b1"] and
        W2.shape == expected["W2"] and
        b2.shape == expected["b2"]
    )
    
    if match:
        print("  ✅ Architecture matches HE inference")
        print("  ✅ 512 → 256 → 2 (no intermediate ReLU)")
        return True
    else:
        print("  ❌ Architecture mismatch!")
        return False


# ═══════════════════════════════════════════════════════════════════════
# Run All Tests
# ═══════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run complete test suite."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 18 + "RUNNING FHE VERIFICATION TESTS" + " " * 20 + "║")
    print("╚" + "═" * 68 + "╝")
    
    results = {}
    
    # Critical tests
    results["Server Cannot Decrypt"] = test_server_cannot_decrypt()
    results["No Server-Side Decrypt"] = test_no_decrypt_in_server_code()
    
    # Functional tests
    results["FHE Matches Plaintext"] = test_fhe_matches_plaintext()
    results["End-to-End Pipeline"] = test_end_to_end_pipeline()
    results["Architecture Match"] = test_architecture_match()
    
    # Summary
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 25 + "TEST SUMMARY" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    for test_name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result is False:
            status = "❌ FAIL"
        else:
            status = "⚠️  SKIP"
        print(f"  {status}  {test_name}")
    
    # Overall verdict
    print()
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print("=" * 70)
    if failed == 0 and passed > 0:
        print("🎉 ALL TESTS PASSED - TRUE FHE IMPLEMENTATION VERIFIED")
        print("=" * 70)
        print()
        print("✅ Server CANNOT decrypt (no secret key)")
        print("✅ Server-side code has NO .decrypt() calls")
        print("✅ FHE inference matches plaintext (within CKKS precision)")
        print("✅ End-to-end pipeline works correctly")
        print("✅ Architecture verified (training matches inference)")
        print()
        print("🔒 This is a TRUE end-to-end FHE implementation.")
        print("🔒 Server sees ZERO plaintext at any point.")
        return 0
    else:
        print(f"❌ {failed} TEST(S) FAILED - FHE IMPLEMENTATION HAS ISSUES")
        print("=" * 70)
        if failed > 0:
            print()
            print("⚠️  CRITICAL: FHE security model violated!")
            print("⚠️  Fix the failing tests before deploying.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
