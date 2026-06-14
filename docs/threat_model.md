# SecureLens — Threat Model Documentation

## System Overview
SecureLens is a privacy-preserving medical image diagnostic system.
A client uploads a chest X-ray, features are encrypted using CKKS
Fully Homomorphic Encryption, and the cloud server runs inference
on ciphertexts without accessing plaintext data.

---

## Assets Being Protected
| Asset | Sensitivity | Protection Method |
|-------|-------------|-------------------|
| Raw X-ray pixels | Critical | Never transmitted |
| 512-dim feature vector | High | CKKS encrypted |
| Diagnostic result | Medium | Decrypted client-side |
| CKKS secret key | Critical | Client-side only |
| Model weights | Medium | Server-side plaintext |

---

## Threat Actors
| Actor | Capability | Goal |
|-------|------------|------|
| Malicious cloud provider | Full server access | Read patient data |
| Network attacker (MITM) | Intercept traffic | Steal X-ray images |
| Server-side attacker | Read server memory | Access patient records |
| Insider threat | Physical server access | View diagnostic data |

---

## Attack Vectors and Mitigations

### 1. Server Data Breach
- **Attack**: Attacker gains full access to cloud server
- **Without FHE**: All patient X-rays exposed
- **With FHE**: Only CKKS ciphertexts stored — mathematically useless without secret key
- **Status**: MITIGATED

### 2. Network Interception (MITM)
- **Attack**: Attacker intercepts HTTP traffic
- **Without FHE**: Raw feature vectors readable
- **With FHE**: 326KB ciphertext — indistinguishable from random noise
- **Status**: MITIGATED

### 3. Image Tampering Attack
- **Attack**: Attacker modifies image during transmission
- **Without FHE**: Corrupted image reaches server → wrong diagnosis
- **With FHE**: Image encrypted before transmission → tampering affects only ciphertext noise
- **Status**: MITIGATED (demonstrated in attack demo)

### 4. Inference Attack on Model Output
- **Attack**: Attacker learns patient features from repeated queries
- **Mitigation**: CKKS noise + rate limiting on API
- **Status**: PARTIALLY MITIGATED

### 5. Timing Side-Channel
- **Attack**: Measure inference time to infer input characteristics
- **Mitigation**: Consistent O(n) linear inference time
- **Status**: LOW RISK

---

## What FHE Does NOT Protect Against
1. Compromised client device (secret key stolen)
2. Malicious model weights (server-side poisoning)
3. Volume analysis (attacker sees number of queries)
4. Social engineering attacks

---

## CKKS Security Parameters
| Parameter | Value | Security Guarantee |
|-----------|-------|-------------------|
| Scheme | CKKS | Approximate HE for real numbers |
| Poly modulus degree | 8192 | 128-bit security level |
| Coefficient modulus | [60,40,40,60] | 3 multiplication levels |
| Global scale | 2^40 | ~10 decimal digits precision |
| Security level | 128-bit | Equivalent to AES-128 |
| Decryption error | ~7×10⁻⁸ | Negligible for classification |

---

## Compliance
- HIPAA: Compliant (no PHI stored on server)
- India DPDP Act 2023: Compliant (sensitive health data encrypted)
- GDPR Article 25: Compliant (privacy by design)