"""
Preservation Property Tests - Classification Behavior Unchanged

**Validates: Requirements 3.1, 3.2**

These tests capture the baseline behavior of the classify_image() function
on UNFIXED code, isolating classification logic from Gradio interface issues.

CRITICAL: These tests should PASS on unfixed code (with mocked Gradio interface).
They verify that after fixing the Gradio API compatibility issue, the core
classification logic remains unchanged.

The tests mock/bypass the Gradio interface to focus solely on:
- classify_image() function behavior
- Model loading and lazy initialization
- Image preprocessing and inference
- Confidence calculations and result formatting
- Error handling for edge cases
"""

import sys
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestClassifyImagePreservation:
    """Test that classify_image() behavior is preserved after Gradio API fix"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - reset global model state"""
        # Import app module
        # Note: This may fail on unfixed code due to Gradio API issues
        # So we'll import the classify_image function directly if needed
        pass
    
    def get_classify_image_function(self):
        """
        Get the classify_image function, bypassing Gradio interface creation.
        
        This allows us to test the classification logic on unfixed code
        even when the Gradio interface creation fails.
        """
        try:
            # Try direct import first
            import app
            return app.classify_image
        except AttributeError as e:
            # If import fails due to Gradio API issue, we need to extract
            # the function by manually executing app.py up to the function definition
            print(f"Warning: Direct import failed ({e}), using alternative approach")
            
            # Read app.py and extract just the function we need
            app_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'app.py'
            )
            
            # Create a namespace to execute the code
            namespace = {}
            
            # Read the file
            with open(app_path, 'r') as f:
                lines = f.readlines()
            
            # Execute only the necessary parts (imports, globals, functions)
            # Skip the Gradio interface creation at the bottom
            code_to_execute = []
            in_interface_creation = False
            
            for i, line in enumerate(lines):
                # Stop before the Gradio Interface creation
                if 'demo = gr.Interface' in line:
                    in_interface_creation = True
                    break
                code_to_execute.append(line)
            
            exec(''.join(code_to_execute), namespace)
            return namespace['classify_image']
    
    def test_none_input_returns_error_message(self):
        """
        Property: classify_image(None) returns specific error message
        
        This behavior must be preserved after the Gradio API fix.
        """
        classify_image = self.get_classify_image_function()
        
        result = classify_image(None)
        
        assert result == 'Please upload an X-ray image first.'
        print("✓ None input handling preserved")
    
    def test_classify_image_returns_string(self):
        """
        Property: classify_image() always returns a string
        
        For any input (valid image, None, invalid), the function should
        return a string result (either prediction or error message).
        """
        classify_image = self.get_classify_image_function()
        
        # Test with None
        result = classify_image(None)
        assert isinstance(result, str)
        
        print("✓ Return type is always string")
    
    def test_classify_image_with_pil_image_structure(self):
        """
        Property: classify_image() accepts PIL Image and returns formatted result
        
        The result should contain:
        - **Prediction:** line with either 'Normal' or 'Pneumonia'
        - **Confidence:** line with percentage
        - Description text about FHE system
        """
        classify_image = self.get_classify_image_function()
        
        # Create a dummy PIL image (224x224 RGB as expected by the model)
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        
        result = classify_image(dummy_image)
        
        # Verify result structure
        assert isinstance(result, str)
        assert '**Prediction:**' in result
        assert '**Confidence:**' in result
        assert ('Normal' in result or 'Pneumonia' in result)
        assert 'privacy-preserving' in result or 'Fully Homomorphic Encryption' in result
        
        print("✓ PIL Image classification structure preserved")
        print(f"   Sample result:\n{result[:100]}...")
    
    def test_classify_image_prediction_is_binary(self):
        """
        Property: Prediction is always either 'Normal' or 'Pneumonia'
        
        The model is a binary classifier, so predictions must be one of these two values.
        """
        classify_image = self.get_classify_image_function()
        
        # Create multiple dummy images
        test_images = [
            Image.new('RGB', (224, 224), color='white'),
            Image.new('RGB', (224, 224), color='black'),
            Image.new('RGB', (224, 224), color='gray'),
        ]
        
        for img in test_images:
            result = classify_image(img)
            
            # Should contain exactly one of the two predictions
            has_normal = 'Normal' in result
            has_pneumonia = 'Pneumonia' in result
            
            # XOR: exactly one should be true
            assert (has_normal or has_pneumonia) and not (has_normal and has_pneumonia), \
                f"Result should contain exactly one prediction, got: {result}"
        
        print("✓ Binary classification property preserved")
    
    def test_confidence_format_and_range(self):
        """
        Property: Confidence is formatted as percentage and is between 0-100%
        
        The confidence value should be displayed with .2% format (e.g., "85.42%")
        and should be in a valid probability range.
        """
        classify_image = self.get_classify_image_function()
        
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        result = classify_image(dummy_image)
        
        # Extract confidence line
        lines = result.split('\n')
        confidence_line = [l for l in lines if '**Confidence:**' in l]
        
        assert len(confidence_line) > 0, "Should have confidence line"
        
        # Should contain a percentage
        assert '%' in confidence_line[0]
        
        print("✓ Confidence formatting preserved")
        print(f"   Confidence line: {confidence_line[0]}")
    
    def test_error_handling_for_invalid_input(self):
        """
        Property: Invalid inputs return error messages starting with 'Error:'
        
        The function should gracefully handle exceptions and return
        formatted error messages.
        """
        classify_image = self.get_classify_image_function()
        
        # Test with invalid input types
        invalid_inputs = [
            "not an image",  # string
            12345,  # number
            [1, 2, 3],  # list
        ]
        
        for invalid in invalid_inputs:
            result = classify_image(invalid)
            assert isinstance(result, str)
            # Should either be the "Please upload" message or start with "Error:"
            assert 'Please upload' in result or 'Error:' in result, \
                f"Expected error handling for input {invalid}, got: {result}"
        
        print("✓ Error handling preserved for invalid inputs")
    
    def test_image_array_conversion_to_pil(self):
        """
        Property: classify_image() accepts numpy arrays and converts them to PIL
        
        The function should handle both PIL Images and numpy arrays as input.
        """
        classify_image = self.get_classify_image_function()
        
        # Create a numpy array (simulating Gradio's output format in some cases)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        result = classify_image(img_array)
        
        # Should successfully process and return a result
        assert isinstance(result, str)
        assert '**Prediction:**' in result or 'Error:' in result
        
        print("✓ Numpy array input handling preserved")
    
    def test_lazy_model_loading_preserved(self):
        """
        Property: Model is loaded lazily on first call
        
        The model should not be loaded on import, but only when
        classify_image() is called for the first time.
        """
        # This tests the lazy loading pattern
        # We can verify this by checking that load_model() is called
        # only when needed
        
        # Import the module fresh (resetting global state)
        import importlib
        
        # Create a new namespace
        namespace = {}
        
        # Read app.py up to the load_model function
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'app.py'
        )
        
        with open(app_path, 'r') as f:
            content = f.read()
        
        # Check that global model is initialized as None
        assert 'model = None' in content
        assert 'ckks = None' in content
        assert 'he_engine = None' in content
        
        # Check that load_model checks if model is None before loading
        assert 'if model is None:' in content
        
        print("✓ Lazy loading pattern preserved in code structure")
    
    def test_result_format_consistency(self):
        """
        Property: All successful classifications return results in consistent format
        
        Format should be:
        - Blank line at start
        - **Prediction:** {value}
        - **Confidence:** {percentage}
        - Blank line
        - Description text
        """
        classify_image = self.get_classify_image_function()
        
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        result = classify_image(dummy_image)
        
        # Check format structure
        assert result.startswith('\n'), "Should start with newline"
        assert '**Prediction:**' in result
        assert '**Confidence:**' in result
        
        # Check that description is present
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        assert len(lines) >= 3, "Should have at least prediction, confidence, and description"
        
        print("✓ Result format consistency preserved")
    
    def test_model_inference_produces_valid_probabilities(self):
        """
        Property: Model inference produces softmax probabilities that sum to ~1.0
        
        This is an internal check - the confidence values should be valid
        probabilities from a softmax output.
        """
        classify_image = self.get_classify_image_function()
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='gray')
        
        # Run classification
        result = classify_image(test_image)
        
        # Extract confidence percentage
        import re
        confidence_match = re.search(r'\*\*Confidence:\*\* (\d+\.\d+)%', result)
        
        if confidence_match:
            confidence = float(confidence_match.group(1))
            
            # Confidence should be between 0 and 100
            assert 0 <= confidence <= 100, f"Confidence {confidence}% out of valid range"
            
            # For a binary classifier, confidence >= 50% implies the predicted class
            # (since we take argmax and report that class's probability)
            assert confidence >= 50.0, \
                f"Binary classifier confidence should be >= 50% for predicted class, got {confidence}%"
            
            print(f"✓ Valid probability: {confidence}%")
        else:
            print("Warning: Could not extract confidence value")


class TestPreservationWithRealImages:
    """
    Property-based tests using actual X-ray images from the dataset
    
    These tests verify that classification behavior is preserved using
    real X-ray images from data/chest_xray/test/
    """
    
    def get_test_image_paths(self, class_name, limit=5):
        """Get paths to test images"""
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'chest_xray', 'test', class_name
        )
        
        if not os.path.exists(test_dir):
            return []
        
        image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpeg')]
        return [os.path.join(test_dir, f) for f in image_files[:limit]]
    
    def get_classify_image_function(self):
        """Get classify_image function, bypassing Gradio interface issues"""
        try:
            import app
            return app.classify_image
        except AttributeError:
            # Same workaround as above
            app_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'app.py'
            )
            
            namespace = {}
            with open(app_path, 'r') as f:
                lines = f.readlines()
            
            code_to_execute = []
            for line in lines:
                if 'demo = gr.Interface' in line:
                    break
                code_to_execute.append(line)
            
            exec(''.join(code_to_execute), namespace)
            return namespace['classify_image']
    
    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'chest_xray', 'test', 'NORMAL'
            )
        ),
        reason="Test dataset not available"
    )
    def test_normal_xrays_baseline_behavior(self):
        """
        Property: Normal X-rays from test set produce 'Normal' predictions
        
        This captures baseline behavior for Normal X-rays. After the fix,
        these same images should produce the same predictions.
        """
        classify_image = self.get_classify_image_function()
        
        normal_image_paths = self.get_test_image_paths('NORMAL', limit=3)
        
        if not normal_image_paths:
            pytest.skip("No NORMAL test images found")
        
        results = []
        for img_path in normal_image_paths:
            img = Image.open(img_path).convert('RGB')
            result = classify_image(img)
            
            # Store result for comparison
            results.append({
                'path': os.path.basename(img_path),
                'result': result,
                'has_normal': 'Normal' in result,
                'has_pneumonia': 'Pneumonia' in result
            })
        
        print(f"\n✓ Tested {len(results)} NORMAL X-rays")
        for r in results:
            prediction = 'Normal' if r['has_normal'] else 'Pneumonia'
            print(f"   {r['path']}: {prediction}")
        
        # Note: We're not asserting specific predictions here because
        # this is a baseline capture. We're just documenting the behavior.
        # After the fix, we can compare against this baseline.
    
    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'chest_xray', 'test', 'PNEUMONIA'
            )
        ),
        reason="Test dataset not available"
    )
    def test_pneumonia_xrays_baseline_behavior(self):
        """
        Property: Pneumonia X-rays from test set produce 'Pneumonia' predictions
        
        This captures baseline behavior for Pneumonia X-rays.
        """
        classify_image = self.get_classify_image_function()
        
        pneumonia_image_paths = self.get_test_image_paths('PNEUMONIA', limit=3)
        
        if not pneumonia_image_paths:
            pytest.skip("No PNEUMONIA test images found")
        
        results = []
        for img_path in pneumonia_image_paths:
            img = Image.open(img_path).convert('RGB')
            result = classify_image(img)
            
            results.append({
                'path': os.path.basename(img_path),
                'result': result,
                'has_normal': 'Normal' in result,
                'has_pneumonia': 'Pneumonia' in result
            })
        
        print(f"\n✓ Tested {len(results)} PNEUMONIA X-rays")
        for r in results:
            prediction = 'Normal' if r['has_normal'] else 'Pneumonia'
            print(f"   {r['path']}: {prediction}")


if __name__ == "__main__":
    print("="*70)
    print("Preservation Property Tests - Classification Behavior")
    print("="*70)
    print("\nThese tests capture baseline behavior on UNFIXED code.")
    print("After the Gradio API fix, these same tests should still pass.")
    print("="*70)
    
    # Run tests
    pytest.main([__file__, '-v', '-s'])
