"""
Bug Condition Exploration Test — Normal X-rays Misclassified as Pneumonia

**Validates: Requirements 2.1, 2.2**

Property 1: Bug Condition - Normal X-rays Classified Correctly

This test encodes the EXPECTED BEHAVIOR: Normal X-ray images should be classified
as "Normal" with normal_score > pneumonia_score.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

EXPECTED OUTCOME: Test FAILS - all or most Normal images are predicted as Pneumonia.

GOAL: Surface counterexamples that demonstrate Normal images are misclassified as Pneumonia.
Document raw logits, probabilities, and patterns to guide root cause diagnosis.
"""

import os
import sys
import random
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import io

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_layer.ckks_engine import CKKSEngine
from cloud_server.train_model_fhe_compatible import SecureLensNetFHE


class TestNormalXrayMisclassification:
    """
    Bug Condition Exploration: Test that Normal X-rays are correctly classified.
    
    This is a property-based test using real Normal X-ray images from the test dataset.
    """
    
    @classmethod
    def setup_class(cls):
        """Setup the model and CKKS engine for testing."""
        # Try FHE-compatible model first
        model_path = project_root / "cloud_server" / "models" / "securelens_fhe.pth"
        if not model_path.exists():
            model_path = project_root / "cloud_server" / "models" / "best_model_fhe.pth"
        if not model_path.exists():
            pytest.skip(f"FHE model not found")
        
        print("\n[Test Setup] Loading FHE-compatible model and CKKS engine...")
        
        # Load the SecureLens FHE model
        cls.model = SecureLensNetFHE(num_classes=2)
        cls.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        cls.model.eval()
        print(f"[Test Setup] Model loaded from {model_path.name}")
        
        # Initialize CKKS engine
        cls.ckks = CKKSEngine(
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            global_scale=2**40
        )
        print("[Test Setup] CKKS engine initialized")
        
        # Image preprocessing
        cls.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Get list of Normal test images
        normal_dir = project_root / "data" / "chest_xray" / "test" / "NORMAL"
        if not normal_dir.exists():
            pytest.skip(f"Test data not found at {normal_dir}")
        
        cls.normal_images = [
            normal_dir / f for f in os.listdir(normal_dir) 
            if f.endswith('.jpeg')
        ]
        
        if len(cls.normal_images) == 0:
            pytest.skip("No Normal test images found")
        
        print(f"[Test Setup] Found {len(cls.normal_images)} Normal X-ray images")
        
        # Sample subset for testing (10-20 images as per spec)
        cls.sample_size = min(20, len(cls.normal_images))
        random.seed(42)  # For reproducibility
        cls.sample_images = random.sample(cls.normal_images, cls.sample_size)
        print(f"[Test Setup] Sampled {cls.sample_size} images for testing")
        
        # Storage for counterexamples and diagnostics
        cls.counterexamples = []
        cls.diagnostic_data = []
    
    @classmethod
    def extract_features_and_predict(cls, img_path):
        """
        Extract features from image and run inference to get prediction.
        Returns dict with prediction, scores, and raw logits.
        """
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        img_tensor = cls.transform(img).unsqueeze(0)  # Add batch dimension
        
        # Extract features using ResNet-18 backbone
        with torch.no_grad():
            # Get raw logits from model
            logits = cls.model(img_tensor).squeeze().numpy()
        
        # Apply softmax to get probabilities (matching ckks_engine.py logic)
        exp_vals = np.exp(logits - np.max(logits))
        probs = exp_vals / exp_vals.sum()
        
        # Create result dict matching the expected format
        result = {
            'raw': logits.tolist(),
            'normal_score': float(probs[0]),
            'pneumonia_score': float(probs[1]),
            'prediction': "Pneumonia" if probs[1] > probs[0] else "Normal",
            'confidence': float(max(probs[0], probs[1]))
        }
        
        return result
    
    def test_normal_images_classified_correctly(self):
        """
        Property-based test: All Normal X-ray images should be classified as "Normal"
        
        For each Normal image:
        - Load and preprocess image
        - Extract features and run inference
        - Assert result['prediction'] == 'Normal'
        - Assert result['normal_score'] > result['pneumonia_score']
        - Assert result['confidence'] > 0.5
        
        CRITICAL: This test is EXPECTED TO FAIL on unfixed code.
        Failure confirms the bug exists.
        """
        print("\n" + "="*70)
        print("BUG CONDITION EXPLORATION TEST - Normal X-ray Classification")
        print("="*70)
        print("Testing that Normal X-rays are classified as 'Normal'")
        print(f"Sample size: {self.sample_size} images")
        print("="*70)
        
        failures = []
        
        for idx, img_path in enumerate(self.sample_images):
            print(f"\n[Test {idx+1}/{self.sample_size}] Processing: {img_path.name}")
            
            # Run inference
            try:
                result = self.extract_features_and_predict(img_path)
            except Exception as e:
                print(f"  ❌ ERROR during inference: {e}")
                import traceback
                traceback.print_exc()
                failures.append({
                    'image': img_path.name,
                    'error': str(e)
                })
                continue
            
            # === DIAGNOSTIC LOGGING ===
            print(f"\n  [DIAGNOSTIC] Raw Logits: {result.get('raw', 'N/A')}")
            print(f"  [DIAGNOSTIC] Probabilities:")
            print(f"    - Normal:    {result['normal_score']:.6f}")
            print(f"    - Pneumonia: {result['pneumonia_score']:.6f}")
            print(f"  [DIAGNOSTIC] Argmax Result: {result['prediction']}")
            print(f"  [DIAGNOSTIC] Confidence: {result['confidence']:.4f}")
            
            # Store diagnostic data
            diagnostic_entry = {
                'image': img_path.name,
                'raw_logits': result.get('raw', None),
                'normal_score': result['normal_score'],
                'pneumonia_score': result['pneumonia_score'],
                'prediction': result['prediction'],
                'confidence': result['confidence']
            }
            self.diagnostic_data.append(diagnostic_entry)
            
            # === ASSERTIONS (Expected to FAIL on unfixed code) ===
            
            # Check 1: Prediction should be "Normal"
            if result['prediction'] != 'Normal':
                print(f"  ❌ FAIL: Predicted '{result['prediction']}' instead of 'Normal'")
                failures.append({
                    'image': img_path.name,
                    'expected': 'Normal',
                    'actual': result['prediction'],
                    'normal_score': result['normal_score'],
                    'pneumonia_score': result['pneumonia_score'],
                    'raw_logits': result.get('raw', None)
                })
            else:
                print(f"  ✓ PASS: Correctly predicted 'Normal'")
            
            # Check 2: Normal score should be higher than Pneumonia score
            if result['normal_score'] <= result['pneumonia_score']:
                print(f"  ❌ FAIL: Normal score ({result['normal_score']:.4f}) "
                      f"<= Pneumonia score ({result['pneumonia_score']:.4f})")
                if not any(f['image'] == img_path.name for f in failures):
                    failures.append({
                        'image': img_path.name,
                        'issue': 'normal_score <= pneumonia_score',
                        'normal_score': result['normal_score'],
                        'pneumonia_score': result['pneumonia_score'],
                        'raw_logits': result.get('raw', None)
                    })
            else:
                print(f"  ✓ PASS: Normal score > Pneumonia score")
            
            # Check 3: Confidence should be > 0.5
            if result['confidence'] <= 0.5:
                print(f"  ❌ FAIL: Low confidence ({result['confidence']:.4f})")
                if not any(f['image'] == img_path.name for f in failures):
                    failures.append({
                        'image': img_path.name,
                        'issue': 'low_confidence',
                        'confidence': result['confidence'],
                        'raw_logits': result.get('raw', None)
                    })
            else:
                print(f"  ✓ PASS: Confidence > 0.5")
        
        # === ANALYSIS OF COUNTEREXAMPLES ===
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        print(f"Total images tested: {self.sample_size}")
        print(f"Failures detected: {len(failures)}")
        print(f"Success rate: {((self.sample_size - len(failures)) / self.sample_size * 100):.1f}%")
        
        if failures:
            print("\n" + "="*70)
            print("COUNTEREXAMPLES FOUND")
            print("="*70)
            print("The following Normal images were misclassified:")
            
            for i, failure in enumerate(failures[:5]):  # Show first 5
                print(f"\n{i+1}. Image: {failure['image']}")
                if 'expected' in failure:
                    print(f"   Expected: {failure['expected']}")
                    print(f"   Actual: {failure['actual']}")
                if 'normal_score' in failure:
                    print(f"   Normal Score: {failure['normal_score']:.6f}")
                    print(f"   Pneumonia Score: {failure['pneumonia_score']:.6f}")
                if 'raw_logits' in failure and failure['raw_logits']:
                    print(f"   Raw Logits: {failure['raw_logits']}")
                if 'issue' in failure:
                    print(f"   Issue: {failure['issue']}")
            
            if len(failures) > 5:
                print(f"\n... and {len(failures) - 5} more failures")
            
            # === PATTERN ANALYSIS ===
            print("\n" + "="*70)
            print("PATTERN ANALYSIS")
            print("="*70)
            
            # Analyze logits
            logits_data = [f for f in failures if f.get('raw_logits')]
            if logits_data:
                print("\nLogits Analysis:")
                logit_0_values = [f['raw_logits'][0] for f in logits_data if len(f['raw_logits']) >= 2]
                logit_1_values = [f['raw_logits'][1] for f in logits_data if len(f['raw_logits']) >= 2]
                
                if logit_0_values and logit_1_values:
                    print(f"  Logit[0] (Normal):    mean={np.mean(logit_0_values):.4f}, "
                          f"std={np.std(logit_0_values):.4f}")
                    print(f"  Logit[1] (Pneumonia): mean={np.mean(logit_1_values):.4f}, "
                          f"std={np.std(logit_1_values):.4f}")
                    
                    # Check if logits are consistently inverted
                    inverted_count = sum(1 for i in range(len(logit_0_values)) 
                                       if logit_0_values[i] < logit_1_values[i])
                    print(f"  Inverted logits: {inverted_count}/{len(logit_0_values)} "
                          f"({inverted_count/len(logit_0_values)*100:.1f}%)")
            
            # Analyze score patterns
            normal_scores = [f['normal_score'] for f in failures if 'normal_score' in f]
            pneumonia_scores = [f['pneumonia_score'] for f in failures if 'pneumonia_score' in f]
            
            if normal_scores and pneumonia_scores:
                print("\nScore Analysis:")
                print(f"  Normal scores:    mean={np.mean(normal_scores):.4f}, "
                      f"std={np.std(normal_scores):.4f}")
                print(f"  Pneumonia scores: mean={np.mean(pneumonia_scores):.4f}, "
                      f"std={np.std(pneumonia_scores):.4f}")
            
            # Root cause hypotheses
            print("\n" + "="*70)
            print("ROOT CAUSE HYPOTHESES")
            print("="*70)
            
            if logits_data and logit_0_values and logit_1_values:
                avg_logit_0 = np.mean(logit_0_values)
                avg_logit_1 = np.mean(logit_1_values)
                
                print("\nBased on diagnostic data:")
                if avg_logit_1 > avg_logit_0:
                    print("  ⚠️  Hypothesis 1: Logits are inverted")
                    print("      - Logit[1] (Pneumonia) is consistently higher than Logit[0] (Normal)")
                    print("      - This suggests the model outputs or weight loading is reversed")
                    print("      - Check: Weight loading order in he_inference.py")
                    print("      - Check: Training class mapping vs inference interpretation")
                
                if inverted_count > len(logit_0_values) * 0.8:
                    print("\n  ⚠️  Hypothesis 2: Label mapping is inverted")
                    print("      - Model produces correct logits but interpretation is wrong")
                    print("      - Check: ckks_engine.py decrypt_prediction() label assignment")
                    print("      - Check: Training CLASSES = {'NORMAL': 0, 'PNEUMONIA': 1}")
                
                score_diff = np.mean(pneumonia_scores) - np.mean(normal_scores)
                if score_diff > 0.3:
                    print("\n  ⚠️  Hypothesis 3: Systematic bias in model weights")
                    print("      - Probabilities show strong bias toward Pneumonia class")
                    print("      - Check: Class imbalance in training data")
                    print("      - Check: Weight statistics in linear_weights.json")
            
            print("\n" + "="*70)
            
            # Store counterexamples for reference
            self.counterexamples = failures
        
        # === ASSERTION (Expected to FAIL on unfixed code) ===
        if failures:
            failure_rate = len(failures) / self.sample_size * 100
            pytest.fail(
                f"\n\nBUG CONFIRMED: {len(failures)}/{self.sample_size} Normal images "
                f"misclassified ({failure_rate:.1f}% failure rate)\n"
                f"This is EXPECTED on unfixed code - the test confirms the bug exists.\n"
                f"See diagnostic output above for root cause analysis."
            )
        else:
            print("\n✅ All Normal images correctly classified!")
            print("NOTE: If this passes on supposedly unfixed code, the bug may already be fixed,")
            print("or the root cause analysis may need revision.")


if __name__ == "__main__":
    # Run the test directly
    pytest.main([__file__, "-v", "-s"])
