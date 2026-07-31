# SecureLens FHE Security Audit Report
**Date:** July 31, 2026  
**Auditor:** Senior Cryptography Engineer & ML Researcher  
**Project:** SecureLens - Privacy-Preserving Medical Diagnostics  

---

## Executive Summary

**CRITICAL FINDING: This implementation DOES NOT achieve true end-to-end Fully Homomorphic Encryption (FHE).**

The system contains a **FATAL CRYPTOGRAPHIC VULNERABILITY** in the encrypted inference engine that completely breaks the FHE security model. Plaintext is exposed server-side during inference, violating the fundamental promise of FHE.

**Current Security Rating: 2/10** ⚠️ **BROKEN FHE IMPLEMENTATION**

---

## Detailed Findings

### 🔴 CRITICAL ISSUES (Security Violations)

#### **Issue #1: Plaintext Decryption During Inference**
**Location:** `cloud_server/encrypted_inference/he_inference.py:96-101`  
**Severity:** CRITICAL - Complete FHE Bypass  
**CVSS Score:** 9.8/10

**Code:**
```python
partial_results = np.array(
    [r.decrypt()[0] for r in results], dtype=np.float64
)
enc_output = ts.ckks_vector(context, partial_results.tolist())
```

**Problem:**
The `_linear_he` function **DECRYPTS** intermediate encrypted values on the server during matrix multiplication. This is a **complete violation** of the FHE security model.

**Impact:**
- Server sees plaintext intermediate activations
- Server can reconstruct input features from these activations
- Complete loss of privacy - FHE provides ZERO protection
- The comment "This is the standard approach for multi-output linear layers in TenSEAL's CKKS implementation" is **FALSE**

**Why This Breaks FHE:**
In true FHE, the server must NEVER call `.decrypt()`. The entire computation must remain encrypted. By decrypting intermediate values, the server gains access to sensitive patient data.

**Root Cause:**
Incorrect understanding of TenSEAL's API. The developer tried to concatenate multiple CKKSVector results but used decrypt+re-encrypt instead of proper homomorphic operations.

---

#### **Issue #2: Server Has Secret Key**
**Location:** `cloud_server/server.py:105-106`, `he_inference.py:96`  
**Severity:** CRITICAL - Key Management Failure  
**CVSS Score:** 9.1/10

**Code:**
```python
# Server uses ckks.context which has the secret key
enc_result = he_engine.infer_head(enc_features, ckks.context)
```

**Problem:**
The server creates a FULL CKKS context (with secret key) and passes it to the inference engine. The inference engine then uses this context to decrypt intermediate values.

**Impact:**
- Server CAN decrypt all ciphertexts
- No cryptographic separation between client and server
- If server is compromised, all patient data is exposed
- Defeats the entire purpose of FHE

**Correct Architecture:**
- Client: full context (with secret key) - for encryption/decryption
- Server: public context ONLY (no secret key) - for computation only

---

#### **Issue #3: Server-Side Feature Extraction in Demo Mode**
**Location:** `cloud_server/server.py:353-358`  
**Severity:** CRITICAL - Plaintext Exposure  
**CVSS Score:** 8.5/10

**Code:**
```python
# Server-side feature extraction (DEMO ONLY — not true FHE)
features_512 = _extract_features(img_pil, model, MODELS_DIR)

# Server-side encryption (DEMO ONLY — simulates client action)
enc_features = ts.ckks_vector(ckks.context, features_512.tolist())
```

**Problem:**
The `/api/predict` endpoint (used by web UI) performs feature extraction on the server, exposing raw images and features server-side.

**Impact:**
- Raw medical images transmitted to server
- Server sees all pixel data and extracted features
- If users don't realize this is demo-only, they lose all privacy
- Audit logs may not clearly distinguish demo vs true FHE

**Mitigation:**
- Clearly label demo endpoints
- Add warnings in responses
- Consider disabling demo mode in production
- Add separate audit logs for demo vs FHE

---

### 🟡 MAJOR ISSUES (Design & Implementation)

#### **Issue #4: Incorrect Scale Parameter (Fixed in Comments)**
**Location:** `crypto_layer/ckks_engine.py:44-45`  
**Severity:** MAJOR - Precision Loss  

**Code:**
```python
global_scale: float = 2**40,   # ← FIXED: was 2**30, must be 2**40
```

**Problem:**
Comment indicates the scale was previously `2**30`, which is too small for neural network weights. Current value `2**40` is correct.

**Impact:**
- `2**30` provides only ~9 decimal digits (insufficient for ML)
- `2**40` provides ~12 decimal digits (adequate)
- If reverted to 2**30, would cause accuracy degradation

**Verification Needed:**
Confirm all exported weights and trained models use `2**40`.

---

#### **Issue #5: Missing ReLU in HE Inference**
**Location:** `cloud_server/encrypted_inference/he_inference.py:121-124`  
**Severity:** MAJOR - Architecture Mismatch  

**Problem:**
Training uses ReLU between layers:
```python
# train_model.py:151
nn.ReLU(),  # [2]
```

But HE inference skips ReLU:
```python
# he_inference.py:121-124
# No ReLU between layers (not natively supported in CKKS).
# Linear-only inference — standard in FHE+ML literature.
```

**Impact:**
- **Architecture mismatch**: trained model uses ReLU, inference does not
- Accuracy degradation unknown (not measured)
- May explain any val/test accuracy drop
- NOT standard - CryptoNets and similar papers approximate ReLU

**Solutions:**
1. **Retrain without ReLU** (simplest, acceptable for FHE)
2. **Polynomial approximation** of ReLU (deg-3 polynomial, adds 1-2 mult levels)
3. **Accept accuracy loss** and document it

**Recommendation:**
Retrain the model without ReLU activation to match the HE inference architecture exactly.

---

#### **Issue #6: BatchNorm and Dropout Not Exported**
**Location:** `cloud_server/train_model.py:151-154`  
**Severity:** MAJOR - Architecture Mismatch  

**Code:**
```python
self.head = nn.Sequential(
    nn.Linear(512, 256),      # [0] — exported
    nn.BatchNorm1d(256),      # [1] — NOT exported
    nn.ReLU(),                # [2] — NOT implemented in HE
    nn.Dropout(0.6),          # [3] — NOT exported
    nn.Linear(256, num_classes),  # [4] — exported
)
```

**Problem:**
Training uses BatchNorm and Dropout, but these are not exported for HE inference.

**Impact:**
- BatchNorm: statistics (mean, variance) not applied during HE inference
- Can cause significant accuracy drop if training relied on BN
- In eval mode, BN applies learned statistics - these must be folded into weights

**Solution:**
Fold BatchNorm parameters into the Linear layer weights:
```
W_folded = gamma / sqrt(var + eps) * W
b_folded = gamma / sqrt(var + eps) * (b - mean) + beta
```

---

#### **Issue #7: Context Serialization Has Secret Key**
**Location:** `crypto_layer/ckks_engine.py:78-81`  
**Severity:** MAJOR - Key Leakage Risk  

**Code:**
```python
ctx_bytes = self.context.serialize(save_secret_key=True)
self.public_context = ts.context_from(ctx_bytes)
self.public_context.make_context_public()  # drops secret key
```

**Problem:**
Creating public context by serializing WITH secret key then calling `make_context_public()` is risky. If the intermediate `ctx_bytes` is logged or saved, it contains the secret key.

**Solution:**
Use TenSEAL's direct public context creation:
```python
self.public_context = self.context.copy()
self.public_context.make_context_public()
```

---

### 🟢 MINOR ISSUES (Best Practices)

#### **Issue #8: Audit Logs Don't Distinguish FHE vs Demo**
**Location:** `cloud_server/server.py:246-255`  
**Severity:** MINOR - Observability  

**Problem:**
Audit logs may not clearly separate true FHE requests from demo requests, making compliance verification difficult.

**Solution:**
Add `mode` field to all audit log entries (`"TRUE_FHE"` or `"DEMO"`).

---

#### **Issue #9: Memory Cleanup Incomplete**
**Location:** `crypto_layer/ckks_engine.py:121-123`  
**Severity:** MINOR - Memory Safety  

**Code:**
```python
# Clear plaintext from memory after encryption
feature_vector[:] = 0
del feature_vector
```

**Problem:**
Zero-filling doesn't guarantee secure deletion on modern systems (memory caching, swap, etc.).

**Solution:**
Document that this is best-effort. For true secure deletion, need OS-level APIs (`mlock`, `memset_s`).

---

#### **Issue #10: No Ciphertext Integrity Protection**
**Location:** All network transmission code  
**Severity:** MINOR - Integrity (FHE provides confidentiality only)  

**Problem:**
Ciphertexts transmitted without HMAC or signature. Attacker could corrupt ciphertext in transit.

**Impact:**
- Corrupted ciphertext decrypts to garbage (detected by client)
- No authentication (MITM could replace ciphertexts)

**Solution:**
Add HMAC-SHA256 over ciphertext before transmission.

---

## Architecture Verification

### ❌ Current Flow (BROKEN FHE)
```
Client:
1. Extract features (512-dim) ✅
2. Encrypt with CKKS ✅
3. Send ciphertext to server ✅

Server:
4. Deserialize ciphertext ✅
5. Compute W1 @ enc(x) ⚠️ (decrypts partial results)
6. **DECRYPTS intermediate values** ❌ ❌ ❌
7. Re-encrypts ⚠️
8. Compute W2 @ enc(h) ⚠️ (decrypts again)
9. Return encrypted result ⚠️

Client:
10. Decrypt result ✅
```

**Plaintext Exposed:** Intermediate layer activations (256-dim) - server sees these!

---

### ✅ Required Flow (True FHE)
```
Client:
1. Extract features (512-dim) ✅
2. Encrypt with CKKS ✅
3. Send ciphertext to server ✅

Server (PUBLIC CONTEXT ONLY):
4. Deserialize ciphertext with public context ✅
5. Compute W1 @ enc(x) + b1 → enc(h)  [FULLY HOMOMORPHIC]
6. Compute W2 @ enc(h) + b2 → enc(out)  [FULLY HOMOMORPHIC]
7. Return enc(out) ✅

Client:
8. Decrypt result with secret key ✅
```

**Plaintext Exposed:** NONE - server never sees any plaintext!

---

## CKKS Parameters Review

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Poly modulus degree | 8192 | ✅ Correct (128-bit security) |
| Coeff mod bit sizes | [60,40,40,60] | ✅ Adequate for 2 layers |
| Global scale | 2^40 | ✅ Sufficient precision |
| Security level | 128-bit | ✅ Industry standard |
| Ciphertext size | 326 KB | ✅ Reasonable overhead |

**Assessment:** CKKS parameters are correctly chosen. The issue is not the encryption - it's how it's used.

---

## Key Management Review

### ❌ Current Implementation (INSECURE)
```python
# Server creates context WITH secret key
ckks = CKKSEngine(...)
# Server passes full context to inference
he_engine.infer_head(enc_features, ckks.context)  # has secret key!
# Inference decrypts intermediate values
partial_results = [r.decrypt()[0] for r in results]  # ❌
```

### ✅ Required Implementation (SECURE)
```python
# Client: full context (encrypt + decrypt)
client_ckks = CKKSEngine(...)
enc = client_ckks.encrypt_feature_vector(features)

# Server: public context ONLY (compute only, cannot decrypt)
server_ctx = client_ckks.get_public_context_bytes()
server_pub_ctx = ts.context_from(server_ctx)
# server_pub_ctx.decrypt() → raises error (no secret key)

# Server runs inference with public context
result_enc = he_engine.infer_head(enc, server_pub_ctx)
# Cannot decrypt result - must return to client

# Client decrypts
result = client_ckks.decrypt_prediction(result_enc)
```

---

## Encrypted Inference vs Training Weights

### Weight Export Check
```python
# Training: train_model.py:151-157
nn.Linear(512, 256),      # W1: (256, 512)
nn.BatchNorm1d(256),      # NOT exported ❌
nn.ReLU(),                # NOT in HE ❌
nn.Dropout(0.6),          # NOT exported ❌
nn.Linear(256, 2),        # W2: (2, 256)
```

### HE Inference
```python
# he_inference.py:18-19
W1: (256, 512) ✅ matches
b1: (256,)     ✅ matches
W2: (2, 256)   ✅ matches
b2: (2,)       ✅ matches
```

**Mismatch:** BatchNorm statistics not folded into weights.

**Impact:** Potential accuracy drop if model relied on BN during training.

---

## Threat Model Assessment

| Threat | Mitigated? | Notes |
|--------|------------|-------|
| Server data breach | ❌ NO | Server can decrypt everything |
| Network interception | ⚠️ PARTIAL | Ciphertext confidential, but no integrity |
| Malicious server | ❌ NO | Server sees plaintext activations |
| Compromised client | ⚠️ N/A | Client compromise defeats any crypto |
| Model extraction | ✅ YES | Weights are not sensitive |
| Ciphertext manipulation | ❌ NO | No integrity protection |

---

## Compliance Review

### HIPAA (Health Insurance Portability and Accountability Act)
**Status:** ❌ NON-COMPLIANT

**Issues:**
- PHI (Protected Health Information) exposed server-side
- Encryption claim is FALSE (due to decrypt in inference)
- Audit logs insufficient for true FHE mode

### GDPR Article 25 (Privacy by Design)
**Status:** ❌ NON-COMPLIANT  

**Issues:**
- System does not provide claimed privacy protection
- "FHE" marketing claim is misleading
- Server sees sensitive health data

### India DPDP Act 2023
**Status:** ❌ NON-COMPLIANT  

**Issues:**
- Data minimization violated (server sees more than needed)
- Encryption not end-to-end as claimed

---

## Recommendations

### Immediate Actions (Required for True FHE)

1. **FIX CRITICAL ISSUE #1** ⚠️ TOP PRIORITY
   - Rewrite `_linear_he` to eliminate `.decrypt()` calls
   - Implement proper homomorphic matrix-vector multiplication
   - Use TenSEAL's `CKKSVector.dot()` correctly

2. **FIX CRITICAL ISSUE #2**
   - Server must ONLY use `public_context`
   - Remove all `ckks.context` references from server code
   - Add test to verify server cannot decrypt

3. **FIX MAJOR ISSUE #5**
   - Retrain model WITHOUT ReLU between linear layers
   - OR implement polynomial ReLU approximation
   - Document accuracy impact

4. **FIX MAJOR ISSUE #6**
   - Fold BatchNorm into linear weights before export
   - Add verification test

### Short-term Improvements

5. Add integrity protection (HMAC) to ciphertext transmission
6. Separate demo and FHE endpoints more clearly
7. Enhance audit logging
8. Add end-to-end tests verifying no plaintext exposure

### Long-term Enhancements

9. Consider higher-degree polynomial modulus for ReLU support
10. Implement packed SIMD operations for efficiency
11. Add ciphertext compression
12. Implement circuit privacy (noise flooding)

---

## Verification Tests Required

1. ✅ **Test:** Server cannot decrypt ciphertexts
   ```python
   try:
       server_pub_ctx.decrypt(enc)
       assert False, "Server should not be able to decrypt!"
   except RuntimeError:
       pass  # Expected
   ```

2. ❌ **Test:** No `.decrypt()` calls in server code
   ```bash
   grep -r "\.decrypt()" cloud_server/encrypted_inference/
   # Should return ZERO results
   ```

3. ❌ **Test:** FHE accuracy matches training accuracy
   ```python
   assert abs(fhe_accuracy - model_accuracy) < 0.02
   ```

4. ❌ **Test:** BatchNorm folded correctly
   ```python
   # Verify exported weights include BN statistics
   ```

---

## Overall Score

### Security: 2/10 ⚠️ CRITICAL FAILURES
- Complete FHE bypass via intermediate decryption
- Key management failure
- Plaintext exposure

### Implementation: 4/10
- CKKS parameters correct ✅
- Key generation correct ✅
- Inference logic BROKEN ❌
- Architecture mismatch ❌

### Compliance: 1/10
- Claims are false
- Would fail regulatory audit

### Code Quality: 6/10
- Well-documented
- Clear structure
- But implements wrong cryptographic protocol

---

## Final Assessment

**This implementation would be REJECTED in:**
- ✗ Academic peer review (incorrect FHE usage)
- ✗ Security audit (critical vulnerabilities)
- ✗ Compliance review (false privacy claims)
- ✗ Production deployment (patient data at risk)

**Strengths:**
✅ Good CKKS parameter selection  
✅ Clean code organization  
✅ Comprehensive UI and demo features  
✅ Transfer learning approach is sound  

**Fatal Flaws:**
❌ Server decrypts intermediate values (breaks FHE)  
❌ Server has secret key (breaks key management)  
❌ Architecture mismatch (training ≠ inference)  

---

## Conclusion

**The current implementation is NOT true FHE.** It is "encryption theater" - it looks like FHE, uses FHE libraries, but violates the fundamental FHE security model by decrypting on the server.

**To achieve true FHE, you MUST:**
1. Remove ALL `.decrypt()` calls from server code
2. Use only public context on server
3. Implement proper homomorphic matrix-vector operations
4. Retrain without ReLU or implement polynomial approximation
5. Fold BatchNorm into weights

**After fixes, expected rating: 8.5/10** (production-ready FHE system)

---

**Auditor Notes:**
The developer clearly understands FHE concepts and has good intentions. The implementation is 90% correct. However, the 10% that's wrong is in the most critical part - the actual homomorphic computation. This is a common mistake when learning FHE: the temptation to "peek" at intermediate values for debugging leads to decrypt calls that destroy the security model.

**The fix is straightforward** - just remove the decrypt and use proper homomorphic ops. This is a FIXABLE issue, not a fundamental design flaw.

