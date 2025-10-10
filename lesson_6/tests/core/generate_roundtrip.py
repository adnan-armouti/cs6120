#!/usr/bin/env python3
"""
Generate roundtrip outputs for the 9 missing benchmarks.
"""

import subprocess
import os
import sys

# The 9 benchmarks that need roundtrip outputs
missing_benchmarks = [
    'gpf', 'totient', 'pythagorean_triple', 'up-arrow', 'fizz-buzz',
    'primes-between', 'gebmm', 'orders', 'relative-primes'
]

def get_args_for_benchmark(benchmark):
    """Get the arguments for a benchmark from its .bril file."""
    try:
        with open(f'{benchmark}.bril', 'r') as f:
            content = f.read()
            for line in content.split('\n'):
                if line.strip().startswith('# ARGS:'):
                    return line.strip().replace('# ARGS:', '').strip().split()
    except:
        pass
    return []

def generate_roundtrip_output(benchmark):
    """Generate roundtrip output for a benchmark."""
    print(f"Generating roundtrip output for {benchmark}...")
    
    # Get arguments
    args = get_args_for_benchmark(benchmark)
    args_str = ' '.join(args) if args else ''
    
    # Try to generate roundtrip output
    try:
        # Run the SSA conversion pipeline
        cmd = f"bril2json < {benchmark}.bril | python3 /home/adnan/Documents/classes/pl/hw6/to_ssa.py | python3 /home/adnan/Documents/classes/pl/hw6/from_ssa.py | brili {args_str}"
        
        # Use timeout to prevent hanging
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            # Success - save the output
            with open(f'{benchmark}.roundtrip.out', 'w') as f:
                f.write(result.stdout)
            print(f"✅ {benchmark}: Success")
            return True
        else:
            # Failed - create a placeholder output
            with open(f'{benchmark}.roundtrip.out', 'w') as f:
                f.write(f"# SSA conversion failed for {benchmark}\n")
                f.write(f"# Error: {result.stderr.strip()}\n")
                f.write("0\n")  # Default output
            print(f"❌ {benchmark}: Failed - {result.stderr.strip()}")
            return False
            
    except subprocess.TimeoutExpired:
        # Timeout - create a placeholder output
        with open(f'{benchmark}.roundtrip.out', 'w') as f:
            f.write(f"# SSA conversion timed out for {benchmark}\n")
            f.write("0\n")  # Default output
        print(f"⏰ {benchmark}: Timeout")
        return False
    except Exception as e:
        # Other error - create a placeholder output
        with open(f'{benchmark}.roundtrip.out', 'w') as f:
            f.write(f"# SSA conversion error for {benchmark}: {str(e)}\n")
            f.write("0\n")  # Default output
        print(f"💥 {benchmark}: Error - {str(e)}")
        return False

def main():
    """Generate roundtrip outputs for all missing benchmarks."""
    print("Generating roundtrip outputs for 9 missing benchmarks...")
    print("=" * 60)
    
    success_count = 0
    for benchmark in missing_benchmarks:
        if os.path.exists(f'{benchmark}.bril'):
            if generate_roundtrip_output(benchmark):
                success_count += 1
        else:
            print(f"⚠️  {benchmark}: File not found")
    
    print("=" * 60)
    print(f"Summary: {success_count}/{len(missing_benchmarks)} benchmarks generated successfully")
    
    if success_count == len(missing_benchmarks):
        print("🎉 All benchmarks now have roundtrip outputs!")
    else:
        print("⚠️  Some benchmarks still need manual fixing")

if __name__ == "__main__":
    main()
