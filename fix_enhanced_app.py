#!/usr/bin/env python3
"""
Script to remove orphaned broken code from app_gradio_enhanced.py

Problem: Lines 236-284 contain orphaned code (no function definition, just floating code).
This code is duplicated - the REAL run_comparison function starts at line 286.

Solution: Keep lines 1-235, skip lines 236-284, keep lines 285-end.
"""

def fix_enhanced_app():
    input_file = 'app_gradio_enhanced.py'
    
    print(f"[FIX] Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"[FIX] Total lines in file: {total_lines}")
    
    # Keep lines 1-235 (indices 0-234)
    # Skip lines 236-284 (indices 235-283)
    # Keep lines 285-end (indices 284+)
    
    cleaned_lines = lines[:235] + lines[284:]
    
    print(f"[FIX] Original lines: {total_lines}")
    print(f"[FIX] Removed lines: {284 - 235} (lines 236-284)")
    print(f"[FIX] New total lines: {len(cleaned_lines)}")
    
    print(f"[FIX] Writing cleaned content back to {input_file}...")
    
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
    
    print(f"[FIX] ✅ SUCCESS! {input_file} has been cleaned.")
    print(f"[FIX] Removed orphaned code block (lines 236-284)")
    print(f"[FIX] File now has proper structure:")
    print(f"[FIX]   - Lines 1-235: Complete run_attack function")
    print(f"[FIX]   - Lines 236+: Proper run_comparison, generate_gradcam, and Gradio interface")

if __name__ == "__main__":
    fix_enhanced_app()
