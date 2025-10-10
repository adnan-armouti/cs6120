#!/usr/bin/env python3
"""
Verification script to check that original and roundtrip outputs match when both exist.
Checks all test subdirectories: core, float, long, mem, mixed.
"""

import os
import glob
from pathlib import Path

def verify_directory(directory_name):
    """Check outputs in a specific directory."""
    base_dir = Path(directory_name)
    
    if not base_dir.exists():
        print(f"Directory {directory_name} does not exist")
        return 0, 0, 0
    
    # Find all .bril files
    bril_files = list(base_dir.glob("*.bril"))
    print(f"\n{directory_name.upper()} - Total .bril files: {len(bril_files)}")
    
    # Count output files
    original_files = list(base_dir.glob("*.original.out"))
    roundtrip_files = list(base_dir.glob("*.roundtrip.out"))
    
    print(f"Total .original.out files: {len(original_files)}")
    print(f"Total .roundtrip.out files: {len(roundtrip_files)}")
    
    # Check matches
    matches = 0
    mismatches = 0
    missing_roundtrip = 0
    
    print(f"\nVerifying {directory_name} outputs...")
    print("=" * 60)
    
    for bril_file in bril_files:
        base_name = bril_file.stem
        original_file = base_dir / f"{base_name}.original.out"
        roundtrip_file = base_dir / f"{base_name}.roundtrip.out"
        
        if not original_file.exists():
            print(f"FAIL {base_name}: Missing original output")
            continue
            
        if not roundtrip_file.exists():
            print(f"WARN {base_name}: Missing roundtrip output")
            missing_roundtrip += 1
            continue
        
        # Compare files
        try:
            with open(original_file, 'r') as f:
                original_content = f.read().strip()
            
            with open(roundtrip_file, 'r') as f:
                roundtrip_content = f.read().strip()
            
            if original_content == roundtrip_content:
                print(f"PASS {base_name}: Match")
                matches += 1
            else:
                print(f"FAIL {base_name}: Mismatch")
                print(f"   Original: {repr(original_content[:100])}{'...' if len(original_content) > 100 else ''}")
                print(f"   Roundtrip: {repr(roundtrip_content[:100])}{'...' if len(roundtrip_content) > 100 else ''}")
                mismatches += 1
                
        except Exception as e:
            print(f"FAIL {base_name}: Error reading files - {e}")
            mismatches += 1
    
    print(f"\n{directory_name.upper()} Summary:")
    print(f"  PASS Matches: {matches}")
    print(f"  FAIL Mismatches: {mismatches}")
    print(f"  WARN Missing roundtrip: {missing_roundtrip}")
    if matches + mismatches > 0:
        print(f"  Success rate: {matches}/{matches + mismatches} ({100 * matches / (matches + mismatches):.1f}%)")
    
    return matches, mismatches, missing_roundtrip

def verify_all_outputs():
    """Check outputs in all test subdirectories."""
    subdirectories = ["core", "float", "long", "mem", "mixed"]
    
    total_matches = 0
    total_mismatches = 0
    total_missing = 0
    
    print("SSA Round-trip Verification")
    print("=" * 80)
    
    for subdir in subdirectories:
        matches, mismatches, missing = verify_directory(subdir)
        total_matches += matches
        total_mismatches += mismatches
        total_missing += missing
    
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total PASS Matches: {total_matches}")
    print(f"Total FAIL Mismatches: {total_mismatches}")
    print(f"Total WARN Missing roundtrip: {total_missing}")
    
    total_tests = total_matches + total_mismatches
    if total_tests > 0:
        success_rate = 100 * total_matches / total_tests
        print(f"Overall Success Rate: {total_matches}/{total_tests} ({success_rate:.1f}%)")
    
    if total_mismatches == 0 and total_missing == 0:
        print("ALL TESTS PASSED!")
    elif total_mismatches == 0:
        print("All existing tests passed, but some roundtrip outputs are missing.")
    else:
        print(f"Found {total_mismatches} mismatches that need attention.")
    
    return total_matches, total_mismatches, total_missing

if __name__ == "__main__":
    verify_all_outputs()
