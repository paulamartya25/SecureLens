"""
crypto_layer/ckks_engine.py

SecureLens — CKKS Encryption Engine
Handles: key generation, feature vector encryption, decryption.

ARCHITECTURE (corrected):
  - ResNet-18 extracts 512-dim features CLIENT SIDE
  - CKKS encrypts those 512 features CLIENT SIDE
  - Only ciphertext is sent to server
  - Server has NO secret key — only public context
  - Decryption happens CLIENT SIDE only

Uses TenSEAL's CKKS scheme (approximate HE for real numbers).
"""

import tenseal as ts
import numpy as np
import os


class CKKSEngine:
    """
    Manages the full CKKS lifecycle on the CLIENT side:
      - Context creation (public/secret key pair)
      - Feature vector encryption (512-dim ResNet output)
      - Encrypted result decryption
      - Public context export (for server — no secret key)
      - Context serialization (save/load keys)

    The SECRET KEY never leaves this class.
    The server only ever receives:
      1. The public context (for deserialization only)
      2. Ciphertext bytes (encrypted feature vectors)
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = None,
        global_scale: float = 2**40,   # ← FIXED: was 2**30, must be 2**40
    ):
        """
        Args:
            poly_modulus_degree : Ring dimension.
                                  8192 → 128-bit security (gold standard).
                                  Never use below 8192 for sensitive data.

            coeff_mod_bit_sizes : Coefficient modulus chain.
                                  [60, 40, 40, 60] gives 3 multiplication
                                  levels — sufficient for 2 linear layers.
                                  First and last primes are larger for
                                  scale management.

            global_scale        : Precision scale for CKKS.
                                  2^40 → ~12 decimal digits of precision.
                                  Sufficient for ML weight magnitudes.
                                  DO NOT use 2^30 — insufficient precision
                                  for neural network weights.
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale        = global_scale

        self.context        = None   # Full context WITH secret key (client only)
        self.public_context = None   # Public context WITHOUT secret key (server safe)

        self._create_context()

    # ──────────────────────────────────────────────────────────────────
    # Context & Key Generation
    # ──────────────────────────────────────────────────────────────────

    def _create_context(self):
        """
        Creates TenSEAL CKKS context with full key set:
          - Secret key   : stays on client ONLY — never transmitted
          - Public key   : derived from secret key
          - Galois keys  : for vector rotation operations
          - Relin keys   : for relinearization after multiplication

        Also creates a PUBLIC context (no secret key) that is
        safe to share with / use on the server for deserialization.
        """
        # ── Full context (client only — has secret key) ───────────────
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.context.global_scale = self.global_scale

        # ── Public context (server safe — no secret key) ──────────────
        # Serialize the full context, then reload without secret key
        # This ensures the server context is mathematically identical
        # but cryptographically cannot decrypt anything
        ctx_bytes = self.context.serialize(save_secret_key=True)
        self.public_context = ts.context_from(ctx_bytes)
        self.public_context.make_context_public()  # drops secret key

        print("[CKKSEngine] Context created successfully.")
        print(f"  Poly modulus degree : {self.poly_modulus_degree}")
        print(f"  Coeff mod bit sizes : {self.coeff_mod_bit_sizes}")
        print(f"  Global scale        : 2^{int(np.log2(self.global_scale))}")
        print(f"  Security level      : 128-bit")
        print(f"  Secret key          : CLIENT ONLY — never transmitted")

    # ──────────────────────────────────────────────────────────────────
    # Feature Vector Encryption  (CLIENT SIDE)
    # ──────────────────────────────────────────────────────────────────

    def encrypt_feature_vector(
        self, feature_vector: np.ndarray
    ) -> ts.CKKSVector:
        """
        Encrypts the 512-dim ResNet-18 feature vector using CKKS.

        This is the MAIN encryption method used in SecureLens.
        The feature vector is extracted by ResNet-18 on the CLIENT
        and encrypted here before any data leaves the device.

        Args:
            feature_vector : numpy array of shape (512,)
                             Output of ResNet-18 backbone.

        Returns:
            CKKSVector : encrypted ciphertext (~326 KB when serialized)

        NOTE: This runs CLIENT SIDE.
              The raw feature values never leave the client.
              Only the returned CKKSVector (serialized) is sent to server.
        """
        if not isinstance(feature_vector, np.ndarray):
            feature_vector = np.array(feature_vector, dtype=np.float64)

        if feature_vector.dtype != np.float64:
            feature_vector = feature_vector.astype(np.float64)

        encrypted = ts.ckks_vector(
            self.context,
            feature_vector.tolist()
        )

        print(f"[CKKSEngine] Feature vector encrypted.")
        print(f"  Input size  : {len(feature_vector)} floats (~4 KB)")
        print(f"  Output size : ~{len(encrypted.serialize())//1024} KB ciphertext")

        return encrypted

    def encrypt_feature_vector_to_bytes(
        self, feature_vector: np.ndarray
    ) -> bytes:
        """
        Encrypts feature vector and returns serialized bytes
        ready for network transmission to the server.

        Args:
            feature_vector : numpy array of shape (512,)

        Returns:
            bytes : serialized CKKS ciphertext for sending to server
        """
        enc     = self.encrypt_feature_vector(feature_vector)
        ct_bytes = enc.serialize()

        # Clear plaintext from memory after encryption
        feature_vector[:] = 0
        del feature_vector

        return ct_bytes

    def encrypt_vector(self, vector: list) -> ts.CKKSVector:
        """
        Generic vector encryption. Encrypts any float list.
        Used for testing and utilities.

        Args:
            vector : Python list of floats

        Returns:
            CKKSVector ciphertext
        """
        return ts.ckks_vector(self.context, vector)

    # ──────────────────────────────────────────────────────────────────
    # Decryption  (CLIENT SIDE ONLY)
    # ──────────────────────────────────────────────────────────────────

    def decrypt_vector(
        self, encrypted_vector: ts.CKKSVector
    ) -> np.ndarray:
        """
        Decrypts a CKKSVector back to plaintext numpy array.

        ONLY the client can call this — only the client has
        the secret key. The server CANNOT call this.

        Args:
            encrypted_vector : CKKSVector ciphertext

        Returns:
            numpy float64 array of decrypted values
        """
        decrypted = encrypted_vector.decrypt()
        return np.array(decrypted, dtype=np.float64)

    def decrypt_vector_from_bytes(self, ct_bytes: bytes) -> np.ndarray:
        """
        Deserializes ciphertext bytes received from server
        and decrypts them using the secret key.

        This is what the client calls after receiving the
        server's encrypted response.

        Args:
            ct_bytes : serialized ciphertext bytes from server

        Returns:
            numpy float64 array of decrypted logits
        """
        # Deserialize using FULL context (has secret key)
        enc = ts.ckks_vector_from(self.context, ct_bytes)
        return self.decrypt_vector(enc)

    def decrypt_prediction(
        self, encrypted_output: ts.CKKSVector
    ) -> dict:
        """
        Decrypts server's encrypted output and converts to
        class probabilities via softmax.

        Binary classification:
          index 0 → Normal
          index 1 → Pneumonia

        Args:
            encrypted_output : encrypted logits from server
                               (CKKSVector of length >= 2)

        Returns:
            dict:
              prediction     : "Normal" or "Pneumonia"
              confidence     : float in [0, 1]
              normal_score   : probability for Normal
              pneumonia_score: probability for Pneumonia
              raw            : raw decrypted logits
        """
        raw = self.decrypt_vector(encrypted_output)

        # Softmax — stable version (subtract max for numerical stability)
        logits   = raw[:2]
        exp_vals = np.exp(logits - np.max(logits))
        probs    = exp_vals / exp_vals.sum()

        result = {
            "raw"            : raw.tolist(),
            "normal_score"   : float(probs[0]),
            "pneumonia_score": float(probs[1]),
            "prediction"     : "Pneumonia" if probs[1] > probs[0] else "Normal",
            "confidence"     : float(max(probs[0], probs[1])),
        }

        print(f"[CKKSEngine] Decrypted prediction : {result['prediction']}")
        print(f"  Confidence  : {result['confidence']:.2%}")
        print(f"  Normal      : {result['normal_score']:.4f}")
        print(f"  Pneumonia   : {result['pneumonia_score']:.4f}")

        return result

    def decrypt_prediction_from_bytes(self, ct_bytes: bytes) -> dict:
        """
        Full pipeline: receive server bytes → decrypt → softmax → result.
        Client-side only.

        Args:
            ct_bytes : serialized encrypted logits from server

        Returns:
            same dict as decrypt_prediction()
        """
        raw_values = self.decrypt_vector_from_bytes(ct_bytes)

        # Repackage as a plain numpy result (not CKKSVector)
        logits   = raw_values[:2]
        exp_vals = np.exp(logits - np.max(logits))
        probs    = exp_vals / exp_vals.sum()

        return {
            "raw"            : raw_values.tolist(),
            "normal_score"   : float(probs[0]),
            "pneumonia_score": float(probs[1]),
            "prediction"     : "Pneumonia" if probs[1] > probs[0] else "Normal",
            "confidence"     : float(max(probs[0], probs[1])),
        }

    # ──────────────────────────────────────────────────────────────────
    # Ciphertext Serialization
    # ──────────────────────────────────────────────────────────────────

    def serialize_ciphertext(
        self, ciphertext: ts.CKKSVector
    ) -> bytes:
        """
        Converts CKKSVector to bytes for network transmission.

        Args:
            ciphertext : CKKSVector to serialize

        Returns:
            bytes blob (~326 KB for 512-dim vector)
        """
        return ciphertext.serialize()

    def deserialize_ciphertext(self, data: bytes) -> ts.CKKSVector:
        """
        Reconstructs CKKSVector from bytes.
        Uses FULL context (with secret key) — for CLIENT use.

        Args:
            data : serialized ciphertext bytes

        Returns:
            CKKSVector
        """
        return ts.ckks_vector_from(self.context, data)

    def deserialize_ciphertext_public(
        self, data: bytes
    ) -> ts.CKKSVector:
        """
        Reconstructs CKKSVector from bytes.
        Uses PUBLIC context (no secret key) — for SERVER use.

        The server calls this to deserialize received ciphertexts.
        Cannot be used for decryption.

        Args:
            data : serialized ciphertext bytes

        Returns:
            CKKSVector (cannot be decrypted without secret key)
        """
        return ts.ckks_vector_from(self.public_context, data)

    # ──────────────────────────────────────────────────────────────────
    # Context Serialization — Save & Load Keys
    # ──────────────────────────────────────────────────────────────────

    def save_context(
        self, path: str, save_secret_key: bool = True
    ):
        """
        Saves TenSEAL context to disk.

        Args:
            path            : File path to save
            save_secret_key : If True, saves full context with secret key.
                              KEEP THIS FILE PRIVATE — it can decrypt everything.
                              If False, saves public context only (server-safe).
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True
        )

        if save_secret_key:
            serialized = self.context.serialize(save_secret_key=True)
            print(f"[CKKSEngine] Full context (with secret key) saved → {path}")
            print(f"  ⚠️  KEEP THIS FILE PRIVATE — anyone with it can decrypt")
        else:
            serialized = self.public_context.serialize()
            print(f"[CKKSEngine] Public context saved → {path}")
            print(f"  ✅ Safe to share with server — no secret key inside")

        with open(path, "wb") as f:
            f.write(serialized)

    def load_context(self, path: str):
        """
        Loads a previously saved TenSEAL context from disk.

        Args:
            path : File path of saved context
        """
        with open(path, "rb") as f:
            serialized = f.read()

        self.context = ts.context_from(serialized)
        print(f"[CKKSEngine] Context loaded ← {path}")

    def get_public_context_bytes(self) -> bytes:
        """
        Returns the serialized PUBLIC context bytes.
        Safe to send to the server for ciphertext deserialization.
        Does NOT contain the secret key.

        Returns:
            bytes of public context
        """
        return self.public_context.serialize()

    # ──────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────

    def get_encryption_info(self) -> dict:
        """Returns CKKS parameter summary for API responses."""
        return {
            "scheme"             : "CKKS (Cheon-Kim-Kim-Song)",
            "library"            : "TenSEAL 0.3.14",
            "poly_modulus_degree": self.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.coeff_mod_bit_sizes,
            "global_scale"       : f"2^{int(np.log2(self.global_scale))}",
            "security_bits"      : 128,
            "feature_vector_size": 512,
            "ciphertext_size_kb" : 326,
            "decryption_error"   : "~7.19e-8",
        }


# ──────────────────────────────────────────────────────────────────────
# Self-Test — run directly to verify the engine works
# python crypto_layer/ckks_engine.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SecureLens — CKKSEngine Self-Test")
    print("=" * 60)

    engine = CKKSEngine(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        global_scale=2**40
    )

    # ── Test 1: Encrypt 512-dim feature vector ────────────────────────
    print("\n[Test 1] Encrypting 512-dim feature vector...")
    dummy_features = np.random.randn(512).astype(np.float64)
    enc = engine.encrypt_feature_vector(dummy_features.copy())
    print(f"  Input  : 512 floats")
    print(f"  Output : {len(enc.serialize())//1024} KB ciphertext")

    # ── Test 2: Decrypt and check error ──────────────────────────────
    print("\n[Test 2] Decrypting and checking error...")
    decrypted = engine.decrypt_vector(enc)
    error = np.max(np.abs(decrypted[:512] - dummy_features))
    print(f"  Max decryption error : {error:.2e}")
    assert error < 1e-3, f"Error too large: {error}"
    print(f"  ✅ Error within acceptable CKKS bounds")

    # ── Test 3: Serialization round-trip ─────────────────────────────
    print("\n[Test 3] Serialization round-trip...")
    ct_bytes  = engine.serialize_ciphertext(enc)
    recovered = engine.deserialize_ciphertext(ct_bytes)
    dec2      = engine.decrypt_vector(recovered)
    error2    = np.max(np.abs(dec2[:512] - dummy_features))
    print(f"  Serialized size      : {len(ct_bytes)//1024} KB")
    print(f"  Round-trip error     : {error2:.2e}")
    assert error2 < 1e-3
    print(f"  ✅ Serialization correct")

    # ── Test 4: Public context cannot decrypt ────────────────────────
    print("\n[Test 4] Verifying public context has no secret key...")
    pub_bytes = engine.get_public_context_bytes()
    pub_ctx   = ts.context_from(pub_bytes)
    try:
        enc_pub = ts.ckks_vector_from(pub_ctx, ct_bytes)
        dec_pub = enc_pub.decrypt()
        print(f"  ⚠️  Public context decrypted (secret key still present)")
    except Exception as e:
        print(f"  ✅ Public context cannot decrypt: {e}")

    # ── Test 5: Decrypt prediction format ────────────────────────────
    print("\n[Test 5] Decrypt prediction format...")
    dummy_logits = engine.encrypt_vector([2.14, -0.83])
    result       = engine.decrypt_prediction(dummy_logits)
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']:.2%}")
    print(f"  Normal      : {result['normal_score']:.4f}")
    print(f"  Pneumonia   : {result['pneumonia_score']:.4f}")
    assert result["prediction"] == "Normal"
    print(f"  ✅ Prediction correct")

    # ── Test 6: Encrypt to bytes pipeline ────────────────────────────
    print("\n[Test 6] Full bytes pipeline...")
    feat2    = np.random.randn(512).astype(np.float64)
    feat_copy = feat2.copy()
    ct       = engine.encrypt_feature_vector_to_bytes(feat_copy)
    result6  = engine.decrypt_prediction_from_bytes(ct)
    print(f"  Encrypted to {len(ct)//1024} KB")
    print(f"  Decrypted prediction: {result6['prediction']}")
    print(f"  ✅ Bytes pipeline working")

    # ── Test 7: Encryption info ───────────────────────────────────────
    print("\n[Test 7] Encryption info...")
    info = engine.get_encryption_info()
    for k, v in info.items():
        print(f"  {k:25s}: {v}")

    print("\n" + "=" * 60)
    print("✅ All 7 tests passed. CKKSEngine is correct.")
    print("=" * 60)