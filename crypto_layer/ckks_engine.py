"""
crypto_layer/ckks_engine.py

SecureLens — CKKS Encryption Engine
Handles: key generation, image encryption, vector decryption.
Uses TenSEAL's CKKS scheme (approximate HE for real numbers).
"""

import tenseal as ts
import numpy as np
import os
import pickle


class CKKSEngine:
    """
    Manages the full CKKS lifecycle:
      - Context creation (public/secret key pair)
      - Image pixel encryption
      - Encrypted vector decryption
      - Context serialization (save/load keys)
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: list = None,
        global_scale: float = 2**30,
    ):
        """
        Args:
            poly_modulus_degree : Controls security & capacity.
                                  8192 → ~128-bit security.
            coeff_mod_bit_sizes : Coefficient modulus chain.
                                  More levels = more multiplications allowed.
            global_scale        : Precision scale for CKKS (2^30 is a safe default).
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = global_scale

        self.context = None
        self._create_context()

    # ------------------------------------------------------------------
    # Context & Key Generation
    # ------------------------------------------------------------------

    def _create_context(self):
        """
        Creates a TenSEAL CKKS context with full key set:
        public key, secret key, Galois keys (for rotations),
        and relin keys (for relinearization after multiplication).
        """
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        # Generate keys
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()
        self.context.global_scale = self.global_scale

        print("[CKKSEngine] Context created successfully.")
        print(f"  Poly modulus degree : {self.poly_modulus_degree}")
        print(f"  Coeff mod bit sizes : {self.coeff_mod_bit_sizes}")
        print(f"  Global scale        : 2^{int(np.log2(self.global_scale))}")

    # ------------------------------------------------------------------
    # Image Encryption
    # ------------------------------------------------------------------

    def preprocess_image(self, image_array: np.ndarray) -> np.ndarray:
        """
        Prepares an image (H x W or H x W x C) for encryption:
          1. Converts to grayscale if RGB
          2. Resizes to 128x128 (small enough to fit in one CKKS vector)
          3. Normalizes pixel values to [0, 1]
          4. Flattens to 1D

        Args:
            image_array: numpy array of shape (H, W) or (H, W, C)

        Returns:
            flat_pixels: 1D float array of length 128*128 = 16384
        """
        import cv2

        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array.copy()

        # Resize to 128x128
        resized = cv2.resize(gray, (128, 128))

        # Normalize to [0.0, 1.0]
        normalized = resized.astype(np.float64) / 255.0

        # Flatten
        flat = normalized.flatten()
        return flat

    def encrypt_image(self, image_array: np.ndarray) -> ts.CKKSVector:
        """
        Encrypts a preprocessed image into a CKKS ciphertext.

        Args:
            image_array: Raw numpy image (will be preprocessed internally)

        Returns:
            CKKSVector: Encrypted ciphertext containing all pixel values
        """
        flat_pixels = self.preprocess_image(image_array)

        # Encrypt as a CKKS vector
        encrypted = ts.ckks_vector(self.context, flat_pixels.tolist())
        print(f"[CKKSEngine] Image encrypted. Vector size: {len(flat_pixels)}")
        return encrypted

    def encrypt_vector(self, vector: list) -> ts.CKKSVector:
        """
        Encrypts an arbitrary float vector (e.g., model weights row).

        Args:
            vector: Python list of floats

        Returns:
            CKKSVector ciphertext
        """
        return ts.ckks_vector(self.context, vector)

    # ------------------------------------------------------------------
    # Decryption
    # ------------------------------------------------------------------

    def decrypt_vector(self, encrypted_vector: ts.CKKSVector) -> np.ndarray:
        """
        Decrypts a CKKSVector back to a plaintext numpy array.

        Args:
            encrypted_vector: CKKSVector ciphertext

        Returns:
            Decrypted values as numpy float64 array
        """
        decrypted = encrypted_vector.decrypt()
        return np.array(decrypted, dtype=np.float64)

    def decrypt_prediction(self, encrypted_output: ts.CKKSVector) -> dict:
        """
        Decrypts a model output vector and interprets as class probabilities.

        Assumes binary classification:
          index 0 → Normal
          index 1 → Pneumonia

        Args:
            encrypted_output: Encrypted prediction vector (length >= 2)

        Returns:
            dict with keys: 'raw', 'normal_score', 'pneumonia_score', 'prediction'
        """
        raw = self.decrypt_vector(encrypted_output)

        # Softmax to get probabilities
        exp_vals = np.exp(raw[:2] - np.max(raw[:2]))
        probs = exp_vals / exp_vals.sum()

        result = {
            "raw": raw.tolist(),
            "normal_score": float(probs[0]),
            "pneumonia_score": float(probs[1]),
            "prediction": "Pneumonia" if probs[1] > probs[0] else "Normal",
            "confidence": float(max(probs[0], probs[1])),
        }
        print(f"[CKKSEngine] Decrypted prediction: {result['prediction']} "
              f"(confidence: {result['confidence']:.2%})")
        return result

    # ------------------------------------------------------------------
    # Serialization — Save & Load Keys
    # ------------------------------------------------------------------

    def save_context(self, path: str, save_secret_key: bool = True):
        """
        Saves the TenSEAL context (with or without secret key) to disk.

        Args:
            path           : File path to save (e.g., 'keys/context.tenseal')
            save_secret_key: If True, saves secret key (keep this private!)
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        if save_secret_key:
            serialized = self.context.serialize(save_secret_key=True)
        else:
            # Public context only — safe to share with cloud server
            self.context.make_context_public()
            serialized = self.context.serialize()

        with open(path, "wb") as f:
            f.write(serialized)
        print(f"[CKKSEngine] Context saved → {path}")

    def load_context(self, path: str):
        """
        Loads a previously saved TenSEAL context from disk.

        Args:
            path: File path of the saved context
        """
        with open(path, "rb") as f:
            serialized = f.read()
        self.context = ts.context_from(serialized)
        print(f"[CKKSEngine] Context loaded ← {path}")

    def serialize_ciphertext(self, ciphertext: ts.CKKSVector) -> bytes:
        """
        Serializes a ciphertext to bytes for network transmission.

        Args:
            ciphertext: CKKSVector

        Returns:
            bytes blob
        """
        return ciphertext.serialize()

    def deserialize_ciphertext(self, data: bytes) -> ts.CKKSVector:
        """
        Reconstructs a CKKSVector from bytes received over network.

        Args:
            data: serialized ciphertext bytes

        Returns:
            CKKSVector
        """
        return ts.ckks_vector_from(self.context, data)


# ------------------------------------------------------------------
# Quick sanity test — run this file directly to verify setup
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("SecureLens — CKKS Engine Self-Test")
    print("=" * 50)

    # Use higher scale for better precision
    engine = CKKSEngine(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        global_scale=2**40
    )

    # Test with a dummy 128x128 grayscale image
    dummy_image = np.random.randint(0, 256, (128, 128), dtype=np.uint8)

    print("\n[Test 1] Encrypting dummy image...")
    enc = engine.encrypt_image(dummy_image)

    print("\n[Test 2] Simulating encrypted scalar multiply...")
    enc_scaled = enc * 0.5

    print("\n[Test 3] Decrypting result...")
    decrypted = engine.decrypt_vector(enc_scaled)
    original_flat = engine.preprocess_image(dummy_image)
    expected = original_flat * 0.5
    n = len(expected)

    # Debug: print first 5 values side by side
    print("  First 5 decrypted :", decrypted[:5].round(6).tolist())
    print("  First 5 expected  :", expected[:5].round(6).tolist())

    error = np.max(np.abs(decrypted[:n] - expected[:n]))
    print(f"  Max decryption error: {error:.2e}")
    assert error < 1e-3, f"Decryption error too large: {error}"

    print("\n[Test 4] Serialization round-trip...")
    blob = engine.serialize_ciphertext(enc)
    recovered = engine.deserialize_ciphertext(blob)
    dec2 = engine.decrypt_vector(recovered)
    error2 = np.max(np.abs(dec2[:n] - original_flat[:n]))
    print(f"  Serialization error: {error2:.2e}")
    assert error2 < 1e-3

    print("\n[Test 5] Decrypt prediction format...")
    dummy_output = engine.encrypt_vector([2.1, 0.3])
    result = engine.decrypt_prediction(dummy_output)
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.2%}")

    print("\n✅ All tests passed. CKKSEngine is working correctly.")