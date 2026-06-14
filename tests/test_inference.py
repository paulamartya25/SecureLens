"""
tests/test_inference.py
SecureLens — Unit Tests for HE Inference Engine
Run with: pytest tests/test_inference.py -v
"""

import pytest
import numpy as np
import tenseal as ts
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cloud_server.encrypted_inference.he_inference import HEInferenceEngine


# ── Fixtures ──────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "cloud_server", "models")


@pytest.fixture(scope="module")
def context():
    """CKKS context for all tests."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    ctx.global_scale = 2**40
    return ctx


@pytest.fixture(scope="module")
def engine():
    """HE Inference engine using real trained weights."""
    if not os.path.exists(
            os.path.join(MODELS_DIR, "feature_weights.json")):
        pytest.skip("Trained weights not found. Run train_model.py first.")
    return HEInferenceEngine(MODELS_DIR)


@pytest.fixture(scope="module")
def dummy_engine(tmp_path_factory):
    """HE Inference engine with dummy weights for isolated tests."""
    tmp = tmp_path_factory.mktemp("models")
    np.random.seed(42)
    for fname, out_f, in_f in [
        ("feature_weights.json", 256, 512),
        ("linear_weights.json",  2,   256),
    ]:
        W = np.random.randn(out_f, in_f) * 0.01
        b = np.zeros(out_f)
        with open(os.path.join(tmp, fname), "w") as f:
            json.dump({"W": W.tolist(), "b": b.tolist()}, f)
    return HEInferenceEngine(str(tmp))


@pytest.fixture(scope="module")
def enc_features(context):
    """Encrypted 512-dim feature vector."""
    np.random.seed(0)
    feats = np.random.rand(512).tolist()
    return ts.ckks_vector(context, feats)


# ── Test 1: Weight Loading ────────────────────────────────────────────

class TestWeightLoading:

    def test_weights_loaded(self, engine):
        """All weight matrices must be loaded."""
        assert engine.W1 is not None
        assert engine.W2 is not None
        assert engine.b1 is not None
        assert engine.b2 is not None

    def test_w1_shape(self, engine):
        """W1 must be 256x512 (features layer)."""
        assert engine.W1.shape == (256, 512), \
            f"W1 shape {engine.W1.shape} != (256, 512)"

    def test_w2_shape(self, engine):
        """W2 must be 2x256 (classifier layer)."""
        assert engine.W2.shape == (2, 256), \
            f"W2 shape {engine.W2.shape} != (2, 256)"

    def test_bias_shapes(self, engine):
        """Bias vectors must match output dimensions."""
        assert engine.b1.shape == (256,)
        assert engine.b2.shape == (2,)

    def test_missing_weights_raises(self, tmp_path):
        """Missing weight files must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            HEInferenceEngine(str(tmp_path))


# ── Test 2: Encrypted Inference ───────────────────────────────────────

class TestEncryptedInference:

    def test_infer_returns_ciphertext(self, dummy_engine,
                                      enc_features, context):
        """infer_head must return a CKKSVector."""
        result = dummy_engine.infer_head(enc_features, context)
        assert isinstance(result, ts.CKKSVector)

    def test_infer_output_length(self, dummy_engine,
                                 enc_features, context):
        """Output must have exactly 2 values (Normal, Pneumonia)."""
        result  = dummy_engine.infer_head(enc_features, context)
        decrypted = result.decrypt()
        assert len(decrypted) >= 2, \
            f"Expected >= 2 outputs, got {len(decrypted)}"

    def test_infer_alias(self, dummy_engine, enc_features, context):
        """infer() must be an alias for infer_head()."""
        r1 = dummy_engine.infer_head(enc_features, context)
        r2 = dummy_engine.infer(enc_features, context)
        d1 = np.array(r1.decrypt()[:2])
        d2 = np.array(r2.decrypt()[:2])
        np.testing.assert_allclose(d1, d2, atol=1e-6)

    def test_infer_deterministic(self, dummy_engine,
                                  enc_features, context):
        """Same encrypted input must give same output."""
        r1 = dummy_engine.infer_head(enc_features, context)
        r2 = dummy_engine.infer_head(enc_features, context)
        d1 = np.array(r1.decrypt()[:2])
        d2 = np.array(r2.decrypt()[:2])
        np.testing.assert_allclose(d1, d2, atol=1e-6)


# ── Test 3: Bytes Round-Trip ──────────────────────────────────────────

class TestBytesRoundTrip:

    def test_infer_from_bytes_returns_bytes(self, dummy_engine,
                                             enc_features, context):
        """infer_from_bytes must return bytes."""
        blob   = enc_features.serialize()
        result = dummy_engine.infer_from_bytes(blob, context)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_infer_from_bytes_matches_infer(self, dummy_engine,
                                             enc_features, context):
        """infer_from_bytes must give same result as infer_head."""
        direct    = dummy_engine.infer_head(enc_features, context)
        blob      = enc_features.serialize()
        via_bytes = dummy_engine.infer_from_bytes(blob, context)
        recovered = ts.ckks_vector_from(context, via_bytes)
        d1 = np.array(direct.decrypt()[:2])
        d2 = np.array(recovered.decrypt()[:2])
        np.testing.assert_allclose(d1, d2, atol=1e-6)


# ── Test 4: Model Info ────────────────────────────────────────────────

class TestModelInfo:

    def test_get_info_returns_dict(self, dummy_engine):
        """get_info must return a dict."""
        info = dummy_engine.get_info()
        assert isinstance(info, dict)

    def test_get_info_has_layers(self, dummy_engine):
        """get_info must have layers key."""
        info = dummy_engine.get_info()
        assert "layers" in info
        assert len(info["layers"]) == 2

    def test_get_info_has_classes(self, dummy_engine):
        """get_info must list correct classes."""
        info = dummy_engine.get_info()
        assert "classes" in info
        assert "Normal" in info["classes"]
        assert "Pneumonia" in info["classes"]

    def test_get_info_encryption(self, dummy_engine):
        """get_info must mention CKKS."""
        info = dummy_engine.get_info()
        assert "CKKS" in info["encryption"]


# ── Test 5: Real Model Accuracy ───────────────────────────────────────

class TestRealModelAccuracy:

    def test_real_weights_give_valid_output(self, engine,
                                             enc_features, context):
        """Real trained weights must give valid 2-class output."""
        result    = engine.infer_head(enc_features, context)
        decrypted = np.array(result.decrypt()[:2])
        assert len(decrypted) == 2
        assert not np.any(np.isnan(decrypted)), "NaN in output"
        assert not np.any(np.isinf(decrypted)), "Inf in output"

    def test_softmax_sums_to_one(self, engine, enc_features, context):
        """Softmax of output must sum to 1."""
        result    = engine.infer_head(enc_features, context)
        logits    = np.array(result.decrypt()[:2])
        exp_v     = np.exp(logits - np.max(logits))
        probs     = exp_v / exp_v.sum()
        assert abs(probs.sum() - 1.0) < 1e-6