"""
Bug Condition Exploration Test - Static Analysis of Gradio API Usage

**Validates: Requirements 2.1, 2.2**

CRITICAL: This test analyzes the source code to confirm the bug exists.

This test verifies that the bug condition exists by analyzing app.py source code:
- app.py uses gr.Image() and gr.Textbox() (Gradio 3.x/4.x API)
- requirements.txt specifies Gradio 2.9.4
- Gradio 2.9.4 requires gr.inputs.Image() and gr.outputs.Textbox()

Expected outcome on UNFIXED code:
- Test FAILS, documenting the incompatible API usage

Expected outcome on FIXED code:
- Test PASSES (app.py uses correct gr.inputs.* and gr.outputs.* API)
"""

import os
import re


def test_requirements_has_gradio_2_9_4():
    """Verify requirements.txt specifies Gradio 2.9.4"""
    print("\n[Test 1] Checking requirements.txt for Gradio version...")
    
    req_path = "requirements.txt"
    assert os.path.exists(req_path), f"requirements.txt not found"
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    gradio_line = [line for line in content.split('\n') if 'gradio' in line.lower()]
    
    print(f"   Found: {gradio_line}")
    
    assert any('2.9.4' in line for line in gradio_line), \
        "requirements.txt should specify Gradio 2.9.4"
    
    print("   ✓ Gradio 2.9.4 confirmed in requirements.txt")
    return True


def test_app_py_uses_incompatible_api():
    """
    Bug Condition Test: Verify app.py uses Gradio 3.x/4.x API
    
    This test analyzes app.py source code to find incompatible API usage.
    On unfixed code: FAILS (finds gr.Image and gr.Textbox)
    On fixed code: PASSES (finds gr.inputs.Image and gr.outputs.Textbox)
    """
    print("\n[Test 2] Analyzing app.py for Gradio API usage...")
    
    app_path = "app.py"
    assert os.path.exists(app_path), "app.py not found"
    
    with open(app_path, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    # Search for Gradio component usage
    issues = []
    
    # Pattern 1: gr.Image( - Gradio 3.x/4.x API (WRONG for 2.9.4)
    image_pattern = r'gr\.Image\s*\('
    image_matches = [(i+1, line.strip()) for i, line in enumerate(lines) 
                     if re.search(image_pattern, line)]
    
    # Pattern 2: gr.Textbox( - Gradio 3.x/4.x API (WRONG for 2.9.4)
    textbox_pattern = r'gr\.Textbox\s*\('
    textbox_matches = [(i+1, line.strip()) for i, line in enumerate(lines) 
                       if re.search(textbox_pattern, line)]
    
    # Pattern 3: gr.inputs.Image( - Gradio 2.9.4 API (CORRECT)
    inputs_image_pattern = r'gr\.inputs\.Image\s*\('
    inputs_image_matches = [(i+1, line.strip()) for i, line in enumerate(lines) 
                            if re.search(inputs_image_pattern, line)]
    
    # Pattern 4: gr.outputs.Textbox( - Gradio 2.9.4 API (CORRECT)
    outputs_textbox_pattern = r'gr\.outputs\.Textbox\s*\('
    outputs_textbox_matches = [(i+1, line.strip()) for i, line in enumerate(lines) 
                               if re.search(outputs_textbox_pattern, line)]
    
    print("\n   Analysis Results:")
    print(f"   - gr.Image() usage (wrong for 2.9.4): {len(image_matches)} occurrence(s)")
    if image_matches:
        for line_num, line_content in image_matches:
            print(f"     Line {line_num}: {line_content}")
            issues.append(f"Line {line_num} uses gr.Image() which doesn't exist in Gradio 2.9.4")
    
    print(f"   - gr.Textbox() usage (wrong for 2.9.4): {len(textbox_matches)} occurrence(s)")
    if textbox_matches:
        for line_num, line_content in textbox_matches:
            print(f"     Line {line_num}: {line_content}")
            issues.append(f"Line {line_num} uses gr.Textbox() which doesn't exist in Gradio 2.9.4")
    
    print(f"   - gr.inputs.Image() usage (correct for 2.9.4): {len(inputs_image_matches)} occurrence(s)")
    if inputs_image_matches:
        for line_num, line_content in inputs_image_matches:
            print(f"     Line {line_num}: {line_content}")
    
    print(f"   - gr.outputs.Textbox() usage (correct for 2.9.4): {len(outputs_textbox_matches)} occurrence(s)")
    if outputs_textbox_matches:
        for line_num, line_content in outputs_textbox_matches:
            print(f"     Line {line_num}: {line_content}")
    
    # Determine if code is fixed or unfixed
    has_wrong_api = len(image_matches) > 0 or len(textbox_matches) > 0
    has_correct_api = len(inputs_image_matches) > 0 or len(outputs_textbox_matches) > 0
    
    if has_wrong_api and not has_correct_api:
        print("\n   ✗ BUG CONFIRMED: app.py uses Gradio 3.x/4.x API")
        print("     Expected AttributeError when running with Gradio 2.9.4")
        print("\n   Bug Details:")
        for issue in issues:
            print(f"     - {issue}")
        print("\n   Fix Required:")
        print("     - Replace gr.Image() with gr.inputs.Image()")
        print("     - Replace gr.Textbox() with gr.outputs.Textbox()")
        return False
    
    elif has_correct_api and not has_wrong_api:
        print("\n   ✓ FIXED: app.py uses correct Gradio 2.9.4 API")
        print("     Interface should launch successfully")
        return True
    
    elif has_wrong_api and has_correct_api:
        print("\n   ⚠ MIXED: app.py has both old and new API usage")
        print("     This might indicate partial fix or transitional state")
        return False
    
    else:
        print("\n   ⚠ UNCLEAR: No Gradio interface components found")
        print("     app.py might not use Gradio interface at all")
        return False


def test_demo_launch_configuration():
    """Verify demo.launch() has HF Spaces configuration"""
    print("\n[Test 3] Checking demo.launch() configuration...")
    
    app_path = "app.py"
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Search for demo.launch() calls
    launch_pattern = r'demo\.launch\((.*?)\)'
    matches = re.findall(launch_pattern, content, re.DOTALL)
    
    print(f"   Found {len(matches)} demo.launch() call(s)")
    
    if matches:
        for i, match in enumerate(matches):
            print(f"\n   Call {i+1}: demo.launch({match})")
            
            has_server_name = 'server_name' in match
            has_server_port = 'server_port' in match
            
            print(f"     - Has server_name: {has_server_name}")
            print(f"     - Has server_port: {has_server_port}")
            
            if not has_server_name or not has_server_port:
                print(f"     ✗ Missing HF Spaces configuration")
                print(f"       Should have: server_name='0.0.0.0', server_port=7860")
                return False
            else:
                print(f"     ✓ Has HF Spaces configuration")
    else:
        print("   ⚠ No demo.launch() calls found")
        return False
    
    return True


def main():
    print("="*70)
    print("Bug Condition Exploration Test - Static Code Analysis")
    print("="*70)
    
    results = {}
    
    try:
        # Test 1: Verify Gradio version in requirements
        results['gradio_version'] = test_requirements_has_gradio_2_9_4()
    except AssertionError as e:
        print(f"   ✗ FAILED: {e}")
        results['gradio_version'] = False
    
    try:
        # Test 2: THE BUG CONDITION TEST (should fail on unfixed code)
        results['api_compatibility'] = test_app_py_uses_incompatible_api()
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        results['api_compatibility'] = False
    
    try:
        # Test 3: Check launch configuration
        results['launch_config'] = test_demo_launch_configuration()
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        results['launch_config'] = False
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n✓ Gradio 2.9.4 in requirements.txt: {results['gradio_version']}")
    print(f"{'✗' if not results['api_compatibility'] else '✓'} API Compatibility: {results['api_compatibility']}")
    print(f"{'✗' if not results['launch_config'] else '✓'} Launch Configuration: {results['launch_config']}")
    
    if not results['api_compatibility']:
        print("\n" + "="*70)
        print("BUG CONDITION CONFIRMED")
        print("="*70)
        print("\nThe bug exists: app.py uses Gradio 3.x/4.x API with Gradio 2.9.4")
        print("When executed, this will cause: AttributeError: module 'gradio' has no attribute 'Image'")
        print("\nFix needed:")
        print("  1. Replace gr.Image() with gr.inputs.Image()")
        print("  2. Replace gr.Textbox() with gr.outputs.Textbox()")
        print("  3. Add server_name and server_port to demo.launch()")
        return 1
    
    elif all(results.values()):
        print("\n" + "="*70)
        print("ALL TESTS PASSED - Bug is FIXED")
        print("="*70)
        return 0
    
    else:
        print("\n" + "="*70)
        print("PARTIAL FIX DETECTED")
        print("="*70)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
