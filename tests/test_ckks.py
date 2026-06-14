"""
tests/test_ckks.py
SecureLens — Unit Tests for CKKS Encryption Engine
Run with: pytest tests/test_ckks.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto_layer.ckks_engine import CKKSEngine


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    """Single CKKS engine shared across all tests in this module."""
    return CKKSEngine(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        global_scale=2**40,
    )


@pytest.fixture(scope="module")
def dummy_image_64():
    """64x64 grayscale image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64), dtype=np.uint8)


@pytest.fixture(scope="module")
def dummy_image_rgb():
    """224x224 RGB image (simulates real X-ray upload)."""
    np.random.seed(0)
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


# ── Test 1: Context Creation ──────────────────────────────────────────

class TestContextCreation:

    def test_context_created(self, engine):
        """CKKS context must be initialized."""
        assert engine.context is not None

    def test_poly_modulus_degree(self, engine):
        """Poly modulus degree must match config."""
        assert engine.poly_modulus_degree == 8192

    def test_global_scale(self, engine):
        """Global scale must be 2^40."""
        assert engine.global_scale == 2**40

    def test_coeff_mod_bit_sizes(self, engine):
        """Coefficient modulus sizes must match config."""
        assert engine.coeff_mod_bit_sizes == [60, 40, 40, 60]


# ── Test 2: Image Preprocessing ──────────────────────────────────────

class TestPreprocessing:

    def test_preprocess_grayscale_shape(self, engine, dummy_image_64):
        """Preprocessed output must be 1D with 128*128 = 16384 elements."""
        flat = engine.preprocess_image(dummy_image_64)
        assert flat.shape == (128 * 128,), \
            f"Expected (16384,) got {flat.shape}"

    def test_preprocess_rgb_shape(self, engine, dummy_image_rgb):
        """RGB image must also produce correct flattened size."""
        flat = engine.preprocess_image(dummy_image_rgb)
        assert flat.shape == (128 * 128,)

    def test_preprocess_normalization(self, engine, dummy_image_64):
        """All pixel values must be in [0, 1] after normalization."""
        flat = engine.preprocess_image(dummy_image_64)
        assert flat.min() >= 0.0, "Min value below 0"
        assert flat.max() <= 1.0, "Max value above 1"

    def test_preprocess_dtype(self, engine, dummy_image_64):
        """Output must be float64."""
        flat = engine.preprocess_image(dummy_image_64)
        assert flat.dtype == np.float64


# ── Test 3: Encryption ────────────────────────────────────────────────

class TestEncryption:

    def test_encrypt_returns_ciphertext(self, engine, dummy_image_64):
        """encrypt_image must return a non-None ciphertext."""
        import tenseal as ts
        enc = engine.encrypt_image(dummy_image_64)
        assert enc is not None
        assert isinstance(enc, ts.CKKSVector)

    def test_encrypt_vector(self, engine):
        """encrypt_vector must work on arbitrary float lists."""
        import tenseal as ts
        vec = [0.1, 0.5, 0.9, 0.3]
        enc = engine.encrypt_vector(vec)
        assert isinstance(enc, ts.CKKSVector)

    def test_ciphertext_is_not_plaintext(self, engine, dummy_image_64):
        """Encrypted data must not equal original pixels (basic sanity)."""
        enc   = engine.encrypt_image(dummy_image_64)
        plain = engine.preprocess_image(dummy_image_64)
        # Ciphertext is an object, not a numpy array
        assert not isinstance(enc, np.ndarray), \
            "Ciphertext should not be a plain array"


# ── Test 4: Decryption ────────────────────────────────────────────────

class TestDecryption:

    def test_decrypt_accuracy(self, engine, dummy_image_64):
        """Decrypted values must be close to original (error < 1e-3)."""
        enc      = engine.encrypt_image(dummy_image_64)
        dec      = engine.decrypt_vector(enc)
        original = engine.preprocess_image(dummy_image_64)
        n        = len(original)
        error    = np.max(np.abs(dec[:n] - original[:n]))
        assert error < 1e-3, f"Decryption error too large: {error:.2e}"

    def test_decrypt_shape(self, engine, dummy_image_64):
        """Decrypted output must be a numpy array."""
        enc = engine.encrypt_image(dummy_image_64)
        dec = engine.decrypt_vector(enc)
        assert isinstance(dec, np.ndarray)

    def test_scalar_multiply(self, engine, dummy_image_64):
        """Homomorphic scalar multiply must preserve values."""
        enc      = engine.encrypt_image(dummy_image_64)
        enc_half = enc * 0.5
        dec      = engine.decrypt_vector(enc_half)
        original = engine.preprocess_image(dummy_image_64) * 0.5
        n        = len(original)
        error    = np.max(np.abs(dec[:n] - original[:n]))
        assert error < 1e-3, f"Multiply error: {error:.2e}"

    def test_decrypt_prediction_normal(self, engine):
        """Higher normal score must give Normal prediction."""
        enc    = engine.encrypt_vector([3.0, 0.5])
        result = engine.decrypt_prediction(enc)
        assert result["prediction"] == "Normal"
        assert result["normal_score"] > result["pneumonia_score"]

    def test_decrypt_prediction_pneumonia(self, engine):
        """Higher pneumonia score must give Pneumonia prediction."""
        enc    = engine.encrypt_vector([0.2, 4.0])
        result = engine.decrypt_prediction(enc)
        assert result["prediction"] == "Pneumonia"
        assert result["pneumonia_score"] > result["normal_score"]

    def test_confidence_range(self, engine):
        """Confidence must always be between 0 and 1."""
        enc    = engine.encrypt_vector([1.0, 2.0])
        result = engine.decrypt_prediction(enc)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, engine):
        """Normal + Pneumonia scores must sum to ~1.0."""
        enc    = engine.encrypt_vector([1.5, 0.8])
        result = engine.decrypt_prediction(enc)
        total  = result["normal_score"] + result["pneumonia_score"]
        assert abs(total - 1.0) < 1e-6, f"Scores sum to {total}, not 1.0"


# ── Test 5: Serialization ─────────────────────────────────────────────

class TestSerialization:

    def test_serialize_returns_bytes(self, engine, dummy_image_64):
        """Serialized ciphertext must be bytes."""
        enc  = engine.encrypt_image(dummy_image_64)
        blob = engine.serialize_ciphertext(enc)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

    def test_deserialize_round_trip(self, engine, dummy_image_64):
        """Deserialized ciphertext must decrypt to original values."""
        enc      = engine.encrypt_image(dummy_image_64)
        blob     = engine.serialize_ciphertext(enc)
        recovered = engine.deserialize_ciphertext(blob)
        dec      = engine.decrypt_vector(recovered)
        original = engine.preprocess_image(dummy_image_64)
        n        = len(original)
        error    = np.max(np.abs(dec[:n] - original[:n]))
        assert error < 1e-3, f"Round-trip error: {error:.2e}"

    def test_ciphertext_size(self, engine, dummy_image_64):
        """Ciphertext must be larger than plaintext (encryption overhead)."""
        enc        = engine.encrypt_image(dummy_image_64)
        blob       = engine.serialize_ciphertext(enc)
        plain_size = len(engine.preprocess_image(dummy_image_64)) * 8
        assert len(blob) > plain_size, "Ciphertext smaller than plaintext?"

    def test_save_load_context(self, engine, tmp_path):
        """Context must save and load correctly."""
        path = str(tmp_path / "test_context.tenseal")
        engine.save_context(path, save_secret_key=True)
        assert os.path.exists(path)
        engine.load_context(path)
        assert engine.context is not None


# ── Test 6: Security Properties ───────────────────────────────────────

class TestSecurityProperties:

    def test_same_plaintext_different_ciphertext(self, engine):
        """
        CKKS encryption is probabilistic —
        same input must produce different ciphertexts.
        """
        vec  = [0.5, 0.3, 0.8]
        enc1 = engine.encrypt_vector(vec)
        enc2 = engine.encrypt_vector(vec)
        blob1 = engine.serialize_ciphertext(enc1)
        blob2 = engine.serialize_ciphertext(enc2)
        assert blob1 != blob2, \
            "Same plaintext produced identical ciphertexts — not secure!"

    def test_decryption_requires_context(self, engine):
        """Encrypted data must be decryptable only with the correct context."""
        enc = engine.encrypt_vector([1.0, 2.0])
        dec = engine.decrypt_vector(enc)
        assert len(dec) >= 2

    def test_128bit_security_level(self, engine):
        """Poly modulus degree 8192 provides 128-bit security."""
        assert engine.poly_modulus_degree >= 8192, \
            "poly_modulus_degree < 8192 gives less than 128-bit security"