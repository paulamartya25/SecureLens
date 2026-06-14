"""
utils/benchmark.py
SecureLens — Performance Benchmarking
Measures encryption time, inference time, decryption time.
Run: python utils/benchmark.py
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.encrypted_inference.he_inference import HEInferenceEngine

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "cloud_server", "models")
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "docs")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_RUNS = 10   # number of benchmark runs for averaging


def benchmark_ckks(engine, n_runs=N_RUNS):
    """Benchmarks encrypt, multiply, decrypt operations."""
    print(f"\n[Benchmark] CKKS Operations ({n_runs} runs each)")
    np.random.seed(42)
    vector = np.random.rand(512).tolist()

    # Encryption
    enc_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        enc = engine.encrypt_vector(vector)
        enc_times.append(time.perf_counter() - t)

    # Scalar multiply
    mul_times = []
    enc = engine.encrypt_vector(vector)
    for _ in range(n_runs):
        t = time.perf_counter()
        _ = enc * 0.5
        mul_times.append(time.perf_counter() - t)

    # Decryption
    dec_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        _ = engine.decrypt_vector(enc)
        dec_times.append(time.perf_counter() - t)

    # Serialization
    ser_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        blob = engine.serialize_ciphertext(enc)
        ser_times.append(time.perf_counter() - t)

    blob = engine.serialize_ciphertext(enc)

    results = {
        "encrypt_ms"    : round(np.mean(enc_times)*1000, 3),
        "multiply_ms"   : round(np.mean(mul_times)*1000, 3),
        "decrypt_ms"    : round(np.mean(dec_times)*1000, 3),
        "serialize_ms"  : round(np.mean(ser_times)*1000, 3),
        "ciphertext_kb" : round(len(blob)/1024, 2),
        "plaintext_kb"  : round(len(vector)*8/1024, 4),
        "overhead_ratio": round(len(blob)/(len(vector)*8), 1),
    }

    print(f"  Encryption time    : {results['encrypt_ms']} ms")
    print(f"  Multiply time      : {results['multiply_ms']} ms")
    print(f"  Decryption time    : {results['decrypt_ms']} ms")
    print(f"  Serialization time : {results['serialize_ms']} ms")
    print(f"  Ciphertext size    : {results['ciphertext_kb']} KB")
    print(f"  Plaintext size     : {results['plaintext_kb']} KB")
    print(f"  Overhead ratio     : {results['overhead_ratio']}x")
    return results


def benchmark_inference(engine, ckks_engine, n_runs=N_RUNS):
    """Benchmarks full HE inference pipeline."""
    print(f"\n[Benchmark] HE Inference ({n_runs} runs)")
    np.random.seed(0)
    features = np.random.rand(512).tolist()

    import tenseal as ts
    enc = ts.ckks_vector(ckks_engine.context, features)

    # Layer 1: 512 → 256
    l1_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        h1 = engine._linear(enc, engine.W1, engine.b1,
                             ckks_engine.context)
        l1_times.append(time.perf_counter() - t)

    # Layer 2: 256 → 2
    l2_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        _ = engine._linear(h1, engine.W2, engine.b2,
                            ckks_engine.context)
        l2_times.append(time.perf_counter() - t)

    # Full pipeline
    full_times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        enc_f = ts.ckks_vector(ckks_engine.context, features)
        out   = engine.infer_head(enc_f, ckks_engine.context)
        _     = ckks_engine.decrypt_prediction(out)
        full_times.append(time.perf_counter() - t)

    # Plaintext baseline
    plain_times = []
    W1 = engine.W1
    b1 = engine.b1
    W2 = engine.W2
    b2 = engine.b2
    f  = np.array(features)
    for _ in range(n_runs):
        t  = time.perf_counter()
        h1 = W1 @ f + b1
        h1 = np.maximum(h1, 0)
        _  = W2 @ h1 + b2
        plain_times.append(time.perf_counter() - t)

    results = {
        "layer1_ms"      : round(np.mean(l1_times)*1000, 3),
        "layer2_ms"      : round(np.mean(l2_times)*1000, 3),
        "full_pipeline_ms": round(np.mean(full_times)*1000, 3),
        "plaintext_ms"   : round(np.mean(plain_times)*1000, 4),
        "overhead_factor": round(
            np.mean(full_times)/max(np.mean(plain_times), 1e-9), 1),
    }

    print(f"  Layer 1 (512→256)  : {results['layer1_ms']} ms")
    print(f"  Layer 2 (256→2)    : {results['layer2_ms']} ms")
    print(f"  Full pipeline      : {results['full_pipeline_ms']} ms")
    print(f"  Plaintext baseline : {results['plaintext_ms']} ms")
    print(f"  Overhead factor    : {results['overhead_factor']}x")
    return results


def benchmark_memory():
    """Measures memory usage of key objects."""
    print("\n[Benchmark] Memory Usage")
    import sys as _sys

    np.random.seed(42)
    features   = np.random.rand(512)
    plain_size = features.nbytes

    import tenseal as ts
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60,40,40,60])
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2**40

    enc      = ts.ckks_vector(ctx, features.tolist())
    enc_blob = enc.serialize()
    ctx_blob = ctx.serialize(save_secret_key=True)

    results = {
        "plaintext_bytes"   : int(plain_size),
        "ciphertext_bytes"  : int(len(enc_blob)),
        "context_bytes"     : int(len(ctx_blob)),
        "overhead_ratio"    : round(len(enc_blob)/plain_size, 1),
        "plaintext_kb"      : round(plain_size/1024, 3),
        "ciphertext_kb"     : round(len(enc_blob)/1024, 2),
        "context_kb"        : round(len(ctx_blob)/1024, 2),
    }

    print(f"  Plaintext  : {results['plaintext_kb']} KB")
    print(f"  Ciphertext : {results['ciphertext_kb']} KB "
          f"({results['overhead_ratio']}x overhead)")
    print(f"  Context    : {results['context_kb']} KB")
    return results


def benchmark_accuracy(engine, ckks_engine):
    """Verifies FHE produces same result as plaintext."""
    print("\n[Benchmark] Accuracy / Correctness Verification")
    np.random.seed(123)
    n_tests = 100
    errors  = []
    matches = 0

    for i in range(n_tests):
        feat = np.random.randn(512) * 0.5

        # Plaintext
        h1p    = engine.W1 @ feat + engine.b1
        h1p    = np.maximum(h1p, 0)
        out_p  = engine.W2 @ h1p + engine.b2
        pred_p = "Normal" if out_p[0] > out_p[1] else "Pneumonia"

        # FHE
        import tenseal as ts
        enc     = ts.ckks_vector(ckks_engine.context, feat.tolist())
        enc_out = engine.infer_head(enc, ckks_engine.context)
        result  = ckks_engine.decrypt_prediction(enc_out)
        pred_f  = result["prediction"]

        decrypted = np.array(enc_out.decrypt()[:2])
        error     = np.max(np.abs(decrypted - out_p[:2]))
        errors.append(error)
        if pred_p == pred_f:
            matches += 1

    results = {
        "n_tests"         : n_tests,
        "prediction_match": matches,
        "match_rate_pct"  : round(matches/n_tests*100, 1),
        "mean_error"      : float(f"{np.mean(errors):.2e}"),
        "max_error"       : float(f"{np.max(errors):.2e}"),
        "min_error"       : float(f"{np.min(errors):.2e}"),
    }

    print(f"  Tests run       : {n_tests}")
    print(f"  Prediction match: {matches}/{n_tests} "
          f"({results['match_rate_pct']}%)")
    print(f"  Mean CKKS error : {results['mean_error']}")
    print(f"  Max  CKKS error : {results['max_error']}")
    return results


def main():
    print("="*55)
    print("  SecureLens — Performance Benchmark Suite")
    print("="*55)

    print("\n[Init] Loading CKKS Engine...")
    ckks_engine = CKKSEngine(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60,40,40,60],
        global_scale=2**40)

    print("[Init] Loading HE Inference Engine...")
    he_engine = HEInferenceEngine(MODELS_DIR)

    # Run all benchmarks
    ckks_results     = benchmark_ckks(ckks_engine)
    infer_results    = benchmark_inference(he_engine, ckks_engine)
    memory_results   = benchmark_memory()
    accuracy_results = benchmark_accuracy(he_engine, ckks_engine)

    # Combine all results
    all_results = {
        "model"     : "SecureLensNet (ResNet-18 + Linear HE Head)",
        "dataset"   : "Chest X-Ray (Kaggle) — 5856 images",
        "test_acc"  : "89.42%",
        "ckks_params": {
            "scheme"             : "CKKS",
            "library"            : "TenSEAL 0.3.14",
            "poly_modulus_degree": 8192,
            "coeff_mod_bit_sizes": [60,40,40,60],
            "global_scale"       : "2^40",
            "security_bits"      : 128,
        },
        "ckks_operations" : ckks_results,
        "inference"       : infer_results,
        "memory"          : memory_results,
        "accuracy"        : accuracy_results,
    }

    # Save results
    out_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Results → {out_path}")

    # Print summary
    print("\n" + "="*55)
    print("  BENCHMARK SUMMARY")
    print("="*55)
    print(f"  Encryption time      : {ckks_results['encrypt_ms']} ms")
    print(f"  Inference time (FHE) : "
          f"{infer_results['full_pipeline_ms']} ms")
    print(f"  Decryption time      : {ckks_results['decrypt_ms']} ms")
    print(f"  Total latency        : "
          f"{ckks_results['encrypt_ms'] + infer_results['full_pipeline_ms'] + ckks_results['decrypt_ms']:.1f} ms")
    print(f"  Ciphertext size      : {ckks_results['ciphertext_kb']} KB")
    print(f"  Prediction match rate: "
          f"{accuracy_results['match_rate_pct']}%")
    print(f"  Max CKKS error       : {accuracy_results['max_error']}")
    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()