#!/usr/bin/env python3
"""
Measure static instruction overhead of SSA round-trip using instruction counting
"""

import subprocess
import csv
import os
import glob
import json
from pathlib import Path

def count_static_instructions(bril_file):
    """Count static instructions in a Bril file"""
    try:
        # Run bril2json and count instructions
        cmd = f"bril2json < {bril_file}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='/home/adnan/Documents/classes/pl/hw6/tests')
        
        if result.returncode != 0:
            print(f"Error in bril2json for {bril_file}: {result.stderr}")
            return None
        
        # Parse JSON and count instructions
        prog = json.loads(result.stdout)
        total = 0
        for f in prog.get("functions", []):
            for ins in f.get("instrs", []):
                if "op" in ins:   # count only real instructions; ignore labels/comments
                    total += 1
        return total
        
    except Exception as e:
        print(f"Exception counting instructions for {bril_file}: {e}")
        return None

def count_instructions_from_json(json_str):
    """Count instructions from JSON string"""
    try:
        prog = json.loads(json_str)
        total = 0
        for f in prog.get("functions", []):
            for ins in f.get("instrs", []):
                if "op" in ins:   # count only real instructions; ignore labels/comments
                    total += 1
        return total
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

def run_static_measurement(bril_file):
    """Run static measurement for a single Bril file"""
    try:
        results = {}
        
        # 1. Baseline measurement
        baseline_count = count_static_instructions(bril_file)
        if baseline_count is None:
            return None
        
        results['baseline'] = baseline_count
        
        # 2. SSA round-trip measurement
        ssa_cmd = f"bril2json < {bril_file} | python3 ../to_ssa.py | python3 ../from_ssa.py"
        ssa_result = subprocess.run(ssa_cmd, shell=True, capture_output=True, text=True, cwd='/home/adnan/Documents/classes/pl/hw6/tests')
        
        if ssa_result.returncode != 0:
            print(f"Error in SSA round-trip for {bril_file}: {ssa_result.stderr}")
            results['ssa_roundtrip'] = None
        else:
            ssa_count = count_instructions_from_json(ssa_result.stdout)
            results['ssa_roundtrip'] = ssa_count if ssa_count is not None else None
        
        # 3. SSA + TDCE measurement
        ssa_tdce_cmd = f"bril2json < {bril_file} | python3 ../to_ssa.py | python3 ../from_ssa.py | python3 ../tdce.py"
        ssa_tdce_result = subprocess.run(ssa_tdce_cmd, shell=True, capture_output=True, text=True, cwd='/home/adnan/Documents/classes/pl/hw6/tests')
        
        if ssa_tdce_result.returncode != 0:
            print(f"Error in SSA+TDCE for {bril_file}: {ssa_tdce_result.stderr}")
            results['ssa_roundtrip_tdce'] = None
        else:
            ssa_tdce_count = count_instructions_from_json(ssa_tdce_result.stdout)
            results['ssa_roundtrip_tdce'] = ssa_tdce_count if ssa_tdce_count is not None else None
        
        # 4. SSA + LVN measurement
        ssa_lvn_cmd = f"bril2json < {bril_file} | python3 ../to_ssa.py | python3 ../from_ssa.py | python3 ../lvn_dce.py"
        ssa_lvn_result = subprocess.run(ssa_lvn_cmd, shell=True, capture_output=True, text=True, cwd='/home/adnan/Documents/classes/pl/hw6/tests')
        
        if ssa_lvn_result.returncode != 0:
            print(f"Error in SSA+LVN for {bril_file}: {ssa_lvn_result.stderr}")
            results['ssa_roundtrip_lvn'] = "n/a"
        else:
            ssa_lvn_count = count_instructions_from_json(ssa_lvn_result.stdout)
            results['ssa_roundtrip_lvn'] = ssa_lvn_count if ssa_lvn_count is not None else "n/a"
        
        # 5. SSA + LVN + TDCE measurement
        ssa_lvn_tdce_cmd = f"bril2json < {bril_file} | python3 ../to_ssa.py | python3 ../from_ssa.py | python3 ../lvn_dce.py | python3 ../tdce.py"
        ssa_lvn_tdce_result = subprocess.run(ssa_lvn_tdce_cmd, shell=True, capture_output=True, text=True, cwd='/home/adnan/Documents/classes/pl/hw6/tests')
        
        if ssa_lvn_tdce_result.returncode != 0:
            print(f"Error in SSA+LVN+TDCE for {bril_file}: {ssa_lvn_tdce_result.stderr}")
            results['ssa_roundtrip_lvn_tdce'] = "n/a"
        else:
            ssa_lvn_tdce_count = count_instructions_from_json(ssa_lvn_tdce_result.stdout)
            results['ssa_roundtrip_lvn_tdce'] = ssa_lvn_tdce_count if ssa_lvn_tdce_count is not None else "n/a"
        
        return results
        
    except Exception as e:
        print(f"Exception in static measurement for {bril_file}: {e}")
        return None

def main():
    """Main function to measure static overhead across all benchmarks"""
    benchmark_dirs = ['core', 'float', 'long', 'mem', 'mixed']
    results = []
    
    print("Measuring static instruction overhead...")
    
    for benchmark_dir in benchmark_dirs:
        bril_files = glob.glob(f"{benchmark_dir}/*.bril")
        print(f"Processing {len(bril_files)} files in {benchmark_dir}/")
        
        for bril_file in bril_files:
            benchmark_name = Path(bril_file).stem
            print(f"  Measuring {benchmark_name}...")
            
            measurement_results = run_static_measurement(bril_file)
            
            if measurement_results is not None and measurement_results.get('baseline') is not None:
                # Calculate overheads for each optimization
                baseline = measurement_results['baseline']
                
                row = {
                    'benchmark': benchmark_name,
                    'directory': benchmark_dir,
                    'baseline': baseline
                }
                
                # Add each optimization result
                for opt_name in ['ssa_roundtrip', 'ssa_roundtrip_tdce', 'ssa_roundtrip_lvn', 'ssa_roundtrip_lvn_tdce']:
                    opt_count = measurement_results.get(opt_name)
                    if opt_count is not None and opt_count != "n/a":
                        overhead = opt_count - baseline
                        overhead_pct = (overhead / baseline * 100) if baseline > 0 else 0
                        row[opt_name] = opt_count
                        row[f'{opt_name}_overhead'] = overhead
                        row[f'{opt_name}_overhead_pct'] = round(overhead_pct, 2)
                    else:
                        row[opt_name] = "n/a"
                        row[f'{opt_name}_overhead'] = "n/a"
                        row[f'{opt_name}_overhead_pct'] = "n/a"
                
                results.append(row)
                
                # Print summary for this benchmark
                print(f"    Baseline: {baseline}")
                for opt_name in ['ssa_roundtrip', 'ssa_roundtrip_tdce', 'ssa_roundtrip_lvn', 'ssa_roundtrip_lvn_tdce']:
                    opt_count = measurement_results.get(opt_name)
                    if opt_count is not None and opt_count != "n/a":
                        overhead = opt_count - baseline
                        overhead_pct = (overhead / baseline * 100) if baseline > 0 else 0
                        print(f"    {opt_name}: {opt_count} (overhead: {overhead}, {overhead_pct:.2f}%)")
                    else:
                        print(f"    {opt_name}: n/a")
            else:
                print(f"    Failed to measure {benchmark_name}")
    
    # Save results to CSV
    output_file = 'static_overhead_results.csv'
    fieldnames = ['benchmark', 'directory', 'baseline', 'ssa_roundtrip', 'ssa_roundtrip_overhead', 'ssa_roundtrip_overhead_pct',
                  'ssa_roundtrip_tdce', 'ssa_roundtrip_tdce_overhead', 'ssa_roundtrip_tdce_overhead_pct',
                  'ssa_roundtrip_lvn', 'ssa_roundtrip_lvn_overhead', 'ssa_roundtrip_lvn_overhead_pct',
                  'ssa_roundtrip_lvn_tdce', 'ssa_roundtrip_lvn_tdce_overhead', 'ssa_roundtrip_lvn_tdce_overhead_pct']
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")
    print(f"Total benchmarks measured: {len(results)}")
    
    # Summary statistics
    if results:
        print(f"\nSummary:")
        for opt_name in ['ssa_roundtrip', 'ssa_roundtrip_tdce', 'ssa_roundtrip_lvn', 'ssa_roundtrip_lvn_tdce']:
            valid_results = [r for r in results if r.get(opt_name) is not None and r.get(opt_name) != "n/a"]
            if valid_results:
                total_baseline = sum(r['baseline'] for r in valid_results)
                total_opt = sum(r[opt_name] for r in valid_results)
                total_overhead = total_opt - total_baseline
                avg_overhead_pct = sum(r[f'{opt_name}_overhead_pct'] for r in valid_results) / len(valid_results)
                print(f"  {opt_name}: {len(valid_results)} benchmarks, avg overhead: {avg_overhead_pct:.2f}%")

if __name__ == "__main__":
    main()