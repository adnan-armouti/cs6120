# Comparison with Cooper/Briggs Paper

Based on my analysis of the new implementation and the [Cooper/Briggs 1997 paper](https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/value-numbering.pdf), here's a detailed comparison:

## What the New Implementation Addresses

### Fully Fixed Issues

| Issue from Feedback | Status | Evidence |
|---------------------|--------|----------|
| **get/set phi encoding** | **FIXED** | Uses real `"op": "phi"` instructions (line 280, `process_phi` at line 146). `to_ssa.py` creates real phis, `from_ssa.py` lowers them. |
| **No DCE after GVN** | **FIXED** | Line 886: `tdce.trivial_dce_func(func)` runs after GVN |
| **Dominator tree traversal** | **MATCHES PAPER** | `gvn_walk` does recursive dom-tree walk (lines 263-323) |
| **Scoped expression tables** | **MATCHES PAPER** | `expr2vn_stack` with `push_scope`/`pop_scope` (lines 100-116) |
| **Expression canonicalization** | **MATCHES PAPER** | Commutative sorting (line 82-84), relational flipping (lines 76-78) |
| **Meaningless phi detection** | **MATCHES PAPER** | Line 161: `if arg_vns and all(v == arg_vns[0] for v in arg_vns)` |
| **Redundant phi detection** | **MATCHES PAPER** | Lines 166-168: checks if phi pattern already in expression table |

### Acceptable Differences (Documented Conservative Choices)

| Aspect | Paper | Your Implementation | Impact |
|--------|-------|---------------------|--------|
| **Memory operations** | Suggests memory SSA or alias analysis | Conservative: clears expression table on effectful ops (lines 301-308) | Safe but limits optimization on `mem/` benchmarks |
| **Loop headers** | Can do iterative refinement | Conservative: fresh VN for backedge phis (lines 152-155) | Safe but may miss some loop-carried redundancies |

---

## What's Still Different from the Paper

### 1. No SCC-Based Value Numbering (SCC-VN)

The paper describes **two** algorithms:
- **DVNT** (Dominator-based Value Numbering with Tables) - hash-based, single pass
- **SCC-VN** (Strongly Connected Component Value Numbering) - partition-based, iterative

Your implementation is **DVNT only**. The paper shows SCC-VN can find more equivalences, especially in loops with cyclic dependencies:

```
# Example where SCC-VN is better:
x.1 = phi(0, x.2)
y.1 = phi(0, y.2)
x.2 = y.1 + 1
y.2 = x.1 + 1
# SCC-VN can prove x.1 == y.1, DVNT cannot
```

**Impact**: Minor - DVNT is the more commonly implemented approach and handles most cases.

### 2. No Algebraic Simplifications

The paper mentions that DVNT can be extended with algebraic identities:
- `x + 0 -> x`
- `x * 1 -> x`
- `x - x -> 0`
- `x AND true -> x`

Your implementation doesn't do these (only expression hashing).

**Impact**: Minor - these are typically done in a separate simplification pass.

### 3. No Global Code Motion

The paper's "complete" algorithm includes **hoisting**: moving computations to dominators when beneficial. Your implementation only does **replacement** (substituting redundant computations with copies).

**Impact**: Minor for correctness, moderate for optimization power.

### 4. No Memory Versioning/SSA

For loads/stores, the paper suggests treating memory as having "versions" so that:
```
store p, v1
x = load p      # x gets value v1
store p, v2
y = load p      # y gets value v2, not v1
```

Your implementation conservatively clears all expressions on any effectful operation.

**Impact**: Limits optimization on memory-heavy benchmarks.

---

## Summary Comparison

| Feature | Paper (DVNT) | Your Implementation |
|---------|--------------|---------------------|
| Real SSA phi nodes | Required | Yes |
| Dom-tree traversal | Required | Yes |
| Scoped hash tables | Required | Yes |
| Meaningless phi removal | Yes | Yes |
| Redundant phi removal | Yes | Yes |
| Expression canonicalization | Yes | Yes |
| DCE post-pass | Recommended | Yes |
| SCC-VN (partition-based) | Optional | No |
| Algebraic simplifications | Optional | No |
| Global code motion | Optional | No |
| Memory SSA | Optional | No (conservative) |

---

## Verdict

**The new implementation is now faithful to the paper's DVNT algorithm.** The main structural issue (get/set phi encoding) is completely fixed. You're now operating on real SSA phi nodes exactly as the paper describes.

The remaining differences are:
1. **Optional extensions** (SCC-VN, algebraic simplifications, code motion)
2. **Conservative choices** for memory (acceptable and documented)

Your implementation matches the "hash-based DVNT" approach from the paper. The only way to be "more faithful" would be to also implement SCC-VN (the partition-based approach), but that's a separate algorithm, not a requirement for DVNT correctness.