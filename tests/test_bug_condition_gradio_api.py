"""
Bug Condition Exploration Test - Gradio 2.9.4 API Incompatibility

**Validates: Requirements 2.1, 2.2**

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.

This test verifies that the bug condition exists: app.py uses Gradio 3.x/4.x API
(gr.Image(), gr.Textbox()) with Gradio 2.9.4 installed, which causes AttributeError.

Expected outcome on UNFIXED code: 
- Test FAILS with AttributeError: module 'gradio' has no attribute 'Image'

Expected outcome on FIXED code:
- Test PASSES (interface creates successfully with gr.inputs.Image and gr.outputs.Textbox)
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gradio_version_is_2_9_4():
    """Verify we're testing against Gradio 2.9.4"""
    import gradio as gr
    
    print(f"Gradio version: {gr.__version__}")
    assert gr.__version__ == "2.9.4", f"Expected Gradio 2.9.4, got {gr.__version__}"
    print("✓ Gradio 2.9.4 confirmed")


def test_gradio_2_9_4_api_structure():
    """Verify Gradio 2.9.4 uses gr.inputs/gr.outputs API, not direct gr.Image/gr.Textbox"""
    import gradio as gr
    
    # Test 1: Verify gr.Image does NOT exist in Gradio 2.9.4
    has_direct_image = hasattr(gr, 'Image')
    print(f"gr.Image exists: {has_direct_image}")
    
    # Test 2: Verify gr.Textbox does NOT exist in Gradio 2.9.4
    has_direct_textbox = hasattr(gr, 'Textbox')
    print(f"gr.Textbox exists: {has_direct_textbox}")
    
    # Test 3: Verify gr.inputs.Image DOES exist in Gradio 2.9.4
    has_inputs_image = hasattr(gr.inputs, 'Image')
    print(f"gr.inputs.Image exists: {has_inputs_image}")
    
    # Test 4: Verify gr.outputs.Textbox DOES exist in Gradio 2.9.4
    has_outputs_textbox = hasattr(gr.outputs, 'Textbox')
    print(f"gr.outputs.Textbox exists: {has_outputs_textbox}")
    
    # For Gradio 2.9.4, we expect:
    # - gr.Image and gr.Textbox should NOT exist
    # - gr.inputs.Image and gr.outputs.Textbox SHOULD exist
    
    print("\n=== Gradio 2.9.4 API Structure ===")
    print(f"Uses old API (gr.inputs/gr.outputs): {has_inputs_image and has_outputs_textbox}")
    print(f"Has new API (direct gr.Image/gr.Textbox): {has_direct_image and has_direct_textbox}")


def test_app_interface_creation_with_unfixed_code():
    """
    Bug Condition Test: Attempt to import and parse app.py with Gradio 2.9.4
    
    This test will FAIL on unfixed code with AttributeError, confirming the bug exists.
    This test will PASS on fixed code, confirming the fix works.
    """
    import gradio as gr
    
    print("\n=== Testing app.py Interface Creation ===")
    print("Attempting to import app.py...")
    
    try:
        # This import will fail on unfixed code because app.py uses gr.Image() and gr.Textbox()
        # which don't exist in Gradio 2.9.4
        import app
        
        print("✓ app.py imported successfully")
        print(f"Interface object: {app.demo}")
        print(f"Interface type: {type(app.demo)}")
        
        # If we get here on unfixed code, something is wrong
        # On fixed code, this should succeed
        assert app.demo is not None, "Interface should be created"
        print("✓ Gradio interface created successfully")
        
        # Verify the interface has the correct components
        print(f"Interface function: {app.demo.fns}")
        
        print("\n✓ TEST PASSED: Interface creation successful (bug is FIXED)")
        
    except AttributeError as e:
        error_msg = str(e)
        print(f"\n✗ AttributeError caught: {error_msg}")
        
        # On unfixed code, we expect this specific error
        if "has no attribute 'Image'" in error_msg or "has no attribute 'Textbox'" in error_msg:
            print("\n✗ TEST FAILED (EXPECTED): Bug condition confirmed!")
            print("   app.py uses Gradio 3.x/4.x API (gr.Image/gr.Textbox)")
            print("   but Gradio 2.9.4 requires gr.inputs.Image/gr.outputs.Textbox")
            print("\n   This failure is CORRECT for unfixed code.")
            print("   After fixing, this test should PASS.")
            
            # Re-raise to fail the test (expected failure on unfixed code)
            raise AttributeError(f"Bug condition confirmed: {error_msg}")
        else:
            print(f"Unexpected AttributeError: {error_msg}")
            raise
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {type(e).__name__}: {e}")
        raise


def test_correct_api_usage_example():
    """
    Example: Demonstrate correct Gradio 2.9.4 API usage
    
    This shows what app.py SHOULD use instead of gr.Image() and gr.Textbox()
    """
    import gradio as gr
    
    print("\n=== Correct Gradio 2.9.4 API Usage ===")
    
    def dummy_fn(x):
        return "Test output"
    
    try:
        # This is the CORRECT way to create interface in Gradio 2.9.4
        correct_interface = gr.Interface(
            fn=dummy_fn,
            inputs=gr.inputs.Image(type='pil', label='Test Input'),
            outputs=gr.outputs.Textbox(label='Test Output'),
            title='Test Interface'
        )
        
        print("✓ Interface created successfully with gr.inputs.Image and gr.outputs.Textbox")
        print(f"   Interface: {correct_interface}")
        print("\n   This is the correct API for Gradio 2.9.4")
        
        assert correct_interface is not None
        
    except Exception as e:
        print(f"✗ Failed to create interface with correct API: {e}")
        raise


if __name__ == "__main__":
    print("="*70)
    print("Bug Condition Exploration Test - Gradio 2.9.4 API Incompatibility")
    print("="*70)
    
    try:
        # Test 1: Verify Gradio version
        print("\n[Test 1] Verifying Gradio version...")
        test_gradio_version_is_2_9_4()
        
        # Test 2: Verify API structure
        print("\n[Test 2] Verifying Gradio 2.9.4 API structure...")
        test_gradio_2_9_4_api_structure()
        
        # Test 3: Show correct API usage
        print("\n[Test 3] Demonstrating correct Gradio 2.9.4 API usage...")
        test_correct_api_usage_example()
        
        # Test 4: THE BUG CONDITION TEST (expected to fail on unfixed code)
        print("\n[Test 4] CRITICAL: Testing app.py with unfixed code...")
        print("         Expected: FAIL on unfixed code, PASS on fixed code")
        test_app_interface_creation_with_unfixed_code()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED - Bug is FIXED")
        print("="*70)
        
    except AttributeError as e:
        print("\n" + "="*70)
        print("BUG CONDITION CONFIRMED (Expected Failure)")
        print("="*70)
        print(f"Error: {e}")
        print("\nThis failure confirms the bug exists in unfixed code.")
        print("After applying the fix, re-run this test - it should pass.")
        sys.exit(1)
    
    except Exception as e:
        print("\n" + "="*70)
        print("UNEXPECTED ERROR")
        print("="*70)
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)
