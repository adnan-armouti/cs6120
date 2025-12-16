# Global Value Numbering (GVN) for Bril

A compiler optimization pass implementing Global Value Numbering for the Bril intermediate representation.

## Project Overview

This project implements GVN, an optimization technique that eliminates redundant computations by identifying expressions that compute the same value across basic block boundaries using SSA form.

### Key Results

- **123 benchmarks**: 100% correctness (all tests pass)
- **Static instructions**: 15.9% reduction (geometric mean)
- **Dynamic instructions**: 18.3% reduction (geometric mean)

## Repository Structure

```
hw_final/
|-- gvn.py                  # Main GVN implementation
|-- to_ssa.py               # SSA conversion (pruned SSA with liveness)
|-- from_ssa.py             # SSA lowering (phi node elimination)
|-- tdce.py                 # Trivial dead code elimination
|-- cfg.py                  # Control flow graph construction
|-- df.py                   # Dominance frontier computation
|-- is_ssa.py               # SSA form verification
|-- lvn_dce.py              # Local value numbering with DCE
|-- test_gvn.py             # Test harness for running benchmarks
|-- generate_figures.py     # Figure generation for evaluation
|-- REPORT.md               # Detailed implementation report
|-- README.md               # This file
|-- tests/                  # Test benchmark directories
|   |-- core/               # Core benchmarks (68 tests)
|   |-- float/              # Floating-point benchmarks (18 tests)
|   |-- mem/                # Memory operation benchmarks (31 tests)
|   |-- mixed/              # Mixed operation benchmarks (4 tests)
|   |-- long/               # Long-running benchmarks (2 tests)
|-- figures/                # Generated visualization figures
|-- tmp/                    # Temporary files from test runs
|-- results_all.json        # Benchmark results
|-- ssa_overhead_results.json # SSA overhead analysis results
```

## File Descriptions

| File | Description |
|------|-------------|
| `gvn.py` | Core GVN algorithm with dominator tree walk, expression hashing, copy coalescing, and parameter renaming |
| `to_ssa.py` | Converts Bril to SSA form using pruned SSA (liveness-aware phi placement) |
| `from_ssa.py` | Lowers SSA form back to regular Bril by converting phi nodes to copies |
| `tdce.py` | Trivial dead code elimination pass |
| `cfg.py` | Builds control flow graphs from Bril instructions |
| `df.py` | Computes dominators, dominator trees, and dominance frontiers |
| `is_ssa.py` | Verifies that a program is in valid SSA form |
| `lvn_dce.py` | Local value numbering with integrated dead code elimination |
| `test_gvn.py` | Test harness that runs GVN on benchmarks and collects metrics |
| `generate_figures.py` | Generates visualization figures for the evaluation |

## Requirements

- Python 3.8+
- Bril tools (`bril2json`, `bril2txt`, `brili`)
- matplotlib (for figure generation)
- numpy (for figure generation)
- tqdm (optional, for progress bars)

## Usage

### Running GVN on a single file

```bash
cat program.bril | bril2json | python3 gvn.py | bril2txt
```

### Running GVN and executing the result

```bash
cat program.bril | bril2json | python3 gvn.py | brili
```

### Running the test suite

Run on all benchmarks:
```bash
python3 test_gvn.py tests/core tests/float tests/mem tests/mixed tests/long --out results_all.json
```

Run on specific benchmark directory:
```bash
python3 test_gvn.py tests/core --out results_core.json
```

Run on a single file:
```bash
python3 test_gvn.py tests/core/euclid.bril --out results_single.json
```

### Generating figures

```bash
python3 generate_figures.py
```

This generates figures in the `figures/` directory and outputs summary statistics.

### Individual pipeline stages

Convert to SSA:
```bash
cat program.bril | bril2json | python3 to_ssa.py | bril2txt
```

Lower from SSA:
```bash
cat program.bril | bril2json | python3 from_ssa.py | bril2txt
```

Run dead code elimination:
```bash
cat program.bril | bril2json | python3 tdce.py | bril2txt
```

Check if program is in SSA form:
```bash
cat program.bril | bril2json | python3 is_ssa.py
```

View CFG structure:
```bash
cat program.bril | bril2json | python3 cfg.py
```

View dominance information:
```bash
cat program.bril | bril2json | python3 df.py
```

## Pipeline Overview

```
Input Program (Bril)
SSA Conversion (to_ssa.py)
GVN Pass (gvn.py)
SSA Copy Propagation (in gvn.py)
Copy Coalescing (in gvn.py)
SSA Lowering (from_ssa.py)
Dead Code Elimination (tdce.py)
Optimized Program (Bril)
```

## Key Optimizations

1. **Pruned SSA**: Only places phi nodes where variables are live, reducing overhead
2. **GVN with dominator tree walk**: Finds redundant expressions across basic blocks
3. **Copy coalescing**: Eliminates redundant copies from phi lowering
4. **Parameter renaming**: Reduces copies for function parameters
5. **Constant copy fusion**: Merges constants with their single-use copies

## Output Files

After running benchmarks:

- `results_all.json`: Per-benchmark results with static/dynamic instruction counts
- `ssa_overhead_results.json`: SSA overhead analysis data

After running figure generation:

- `figures/static_improvement_bar.png`: Top benchmarks by static reduction
- `figures/static_ratio_histogram.png`: Distribution of static ratios
- `figures/dynamic_improvement_bar.png`: Top benchmarks by dynamic reduction
- `figures/category_comparison.png`: Results by benchmark category
- `figures/static_vs_dynamic_scatter.png`: Static vs dynamic correlation
- `figures/summary_table.png`: Summary statistics
- `figures/waterfall_savings.png`: Cumulative savings
- `figures/ssa_overhead_bar.png`: SSA overhead vs GVN savings
- `figures/ssa_overhead_histogram.png`: SSA overhead distribution
- `figures/ssa_vs_gvn_scatter.png`: SSA overhead vs final result
- `figures/ssa_pipeline_comparison.png`: Instruction counts by stage
- `figures/ssa_summary_table.png`: SSA overhead summary

## References

- Briggs, P., Cooper, K. D., & Simpson, L. T. (1997). Value Numbering. Softw: Pract. Exper.
- Cooper, K. D., & Torczon, L. (2011). Engineering a Compiler. Morgan Kaufmann.
- Cytron, R., et al. (1991). Efficiently Computing Static Single Assignment Form and the Control Dependence Graph. ACM TOPLAS.