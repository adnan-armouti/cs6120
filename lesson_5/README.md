GAI Tools Disclaimer:  

**We used OpenAI's ChatGPT for the following:**
1. In the df.py file, it helped implement changes to my original version to help pass some stubborn benchmarks. 
2. It proposed adding the Cooper–Harvey–Kennedy (CHK) dominance method as an additional check for testing, and provided the entirety of the chk.py file.
3. Implemented check_methods.py to parse .out files and provide concise pass/fail methods during debugging.
4. Please note that we also used Github Copilot when writing code throughout this project.

**Concrete example of generated code or Q&A output:**  
chk.py file:
```
# ---------- utilities (next/prev, RPO) ----------

def invert_next_to_prev(blocks, next_map):
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for prev_block, next_blocks in next_map.items():
        for next_block in next_blocks:
            prev_map[next_block].append(prev_block)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx[n])
    return prev_map

def rpo_order(blocks, next_map):
    """Reverse postorder numbering over the CFG."""
    names = [b["name"] for b in blocks]
    if not names: return {}, []
    entry = names[0]
    seen, post = set(), []
    def dfs(u):
        if u in seen: return
        seen.add(u)
        for v in next_map.get(u, []):
            dfs(v)
        post.append(u)
    dfs(entry)
    order = list(reversed(post))
    num = {n: i for i, n in enumerate(order)}
    return num, order

# ---------- CHK: idom via reverse postorder + intersect ----------

def intersect(u, v, idom, rpo_num):
    while u != v:
        while rpo_num[u] > rpo_num[v]:
            u = idom[u]
        while rpo_num[v] > rpo_num[u]:
            v = idom[v]
    return u

def idom_chk(blocks, next_map):
    names = [b["name"] for b in blocks]
    if not names: return {}
    entry = names[0]
    prv = invert_next_to_prev(blocks, next_map)
    rpo_num, _ = rpo_order(blocks, next_map)

    # seed
    idom = {n: None for n in names}
    idom[entry] = entry  # CHK uses self-idom during iteration

    changed = True
    while changed:
        changed = False
        # process in RPO, skip entry
        for n in sorted(names, key=lambda x: rpo_num.get(x, 10**9)):
            if n == entry: continue
            # pick a processed predecessor to start
            preds = [p for p in prv.get(n, []) if idom[p] is not None]
            if not preds:
                continue  # unreachable
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(new_idom, p, idom, rpo_num)
            if idom[n] != new_idom:
                idom[n] = new_idom
                changed = True

    # normalize: entry has no idom (None)
    idom[entry] = None
    return idom

# ---------- dom tree + dominance frontier ----------

def build_tree(idom):
    tree = {n: [] for n in idom}
    for b, d in idom.items():
        if d is not None:
            tree[d].append(b)
    for k in tree: tree[k].sort()
    return tree

def dom_frontier(blocks, next_map, idom, tree):
    names = [b["name"] for b in blocks]
    DF = {n: set() for n in names}

    # local
    for b in names:
        for s in next_map.get(b, []):
            if idom.get(s) != b:
                DF[b].add(s)

    # postorder over dom tree
    post = []
    def walk(u):
        for c in tree.get(u, []): walk(c)
        post.append(u)
    for r, d in idom.items():
        if d is None:
            walk(r)

    # upward
    for b in post:
        for c in tree.get(b, []):
            for n in DF[c]:
                if idom.get(n) != b:
                    DF[b].add(n)

    return DF
```

minimal diff to our original df.py to get the three .bril benchmark test cases "mandelbrot", "gol" and "montecarlo" to pass:
```diff
@@ -25,6 +25,7 @@ def append_dom_tree_postorder(dom_tree, node, out_list):
     out_list.append(node)
 
 
+def is_reachable(block_name, entry, next_map):
+    """Check if a block is reachable from the entry using DFS."""
+    if block_name == entry:
+        return True
+    visited = set()
+    stack = [entry]
+    while stack:
+        curr = stack.pop()
+        if curr == block_name:
+            return True
+        if curr in visited:
+            continue
+        visited.add(curr)
+        for next_block in next_map.get(curr, []):
+            if next_block not in visited:
+                stack.append(next_block)
+    return False
+
 def get_dom_sets(blocks, next_map):
     block_names_ordered = [b["name"] for b in blocks]
     if not block_names_ordered: return {}
@@ -35,6 +56,15 @@ def get_dom_sets(blocks, next_map):
     # init dom sets
     doms_map = {entry: {entry}}
     for name in block_names_ordered[1:]:
         doms_map[name] = set(all_block_names)
+    # iter until convergence
+    is_changed = True
+    while is_changed:
+        is_changed = False
+        for block_name in block_names_ordered[1:]:
+            prev = prev_map[block_name]
+            if prev:
+                # Only consider predecessors that are reachable
+                reachable_preds = [p for p in prev if is_reachable(p, entry, next_map)]
+                if reachable_preds:
+                    predoms_intersection = set.intersection(*(doms_map[p] for p in reachable_preds))
+                else:
+                    # No reachable predecessors, block is unreachable
+                    predoms_intersection = set()
+            else:
+                # No predecessors, block is unreachable
+                predoms_intersection = set()
+            updated_doms = {block_name} | predoms_intersection
+            if updated_doms != doms_map[block_name]:
+                doms_map[block_name] = updated_doms
+                is_changed = True
     return doms_map
 
-def get_immediate_doms_and_tree(doms_map):
+def get_immediate_doms_and_tree(doms_map, next_map):
     block_names_ordered = sorted(doms_map.keys())
     entry = next(n for n, ds in doms_map.items() if ds == {n})
     immediate_dom = {entry: None}
     for block_name in block_names_ordered:
         if block_name == entry: continue
+        
+        # A block is unreachable if it's not reachable from the entry
+        if not is_reachable(block_name, entry, next_map):
+            immediate_dom[block_name] = None  # Unreachable
+            continue
+            
         candidate_doms = doms_map[block_name] - {block_name}
         parent = None
         for d in sorted(candidate_doms):
-            if not any(d in doms_map[e] and d != e for e in candidate_doms):
+            # Check if d is dominated by any other candidate dominator
+            # A dominator d is immediate if no other dominator in the candidate set dominates d
+            is_dominated = any(d in doms_map[e] and d != e for e in candidate_doms if e != d)
+            if not is_dominated:
                 parent = d
                 break
         immediate_dom[block_name] = parent
@@ -104,7 +134,7 @@ def run(func, check_slow=False, check_fast=False):
     entry = names[0]
 
     doms_map = get_dom_sets(blocks, next_map)
-    immediate_dom, dom_tree = get_immediate_doms_and_tree(doms_map)
+    immediate_dom, dom_tree = get_immediate_doms_and_tree(doms_map, next_map)
     dominance_frontier = get_dominance_frontier(blocks, next_map, immediate_dom, dom_tree)
 
     print(f"Function {func['name']}:")
@@ -124,9 +154,15 @@ def run(func, check_slow=False, check_fast=False):
     if check_slow:
         is_match = True
         for candidate_a in names:
             for candidate_b in names:
-                if (candidate_a in doms_map[candidate_b]) != dominates_slow(entry, candidate_a, candidate_b, next_map):
+                # Special case: entry dominates all blocks, even unreachable ones
+                if candidate_a == entry:
+                    fast_result = True
+                else:
+                    fast_result = candidate_a in doms_map[candidate_b]
+                if fast_result != dominates_slow(entry, candidate_a, candidate_b, next_map):
                     is_match = False
                     print(f"  [mismatch] slow-check dom {candidate_a} vs {candidate_b}")
         if is_match:
             print("  [check-slow] dom sets agree with slow test.")
```

minimal diff to our original naive.py to get our .bril benchmark test cases "mandelbrot", "gol" and "montecarlo" to pass:
```diff
@@ -1,4 +1,5 @@
 def dominates_slow(entry, dom_block, target_block, next_map):
     if dom_block == entry: return True
+    if dom_block == target_block: return True  # Self-dominance
+    # First check if target is reachable
+    target_reachable = False
     seen, stack = set(), [entry]
     while stack:
         curr_block = stack.pop()
-        if curr_block == dom_block: continue
-        if curr_block == target_block: return False
+        if curr_block == target_block: 
+            target_reachable = True
+            break
+        if curr_block in seen: continue
+        seen.add(curr_block)
         for next_block in next_map.get(curr_block, []):
             if next_block not in seen:
                 seen.add(next_block)
-    return True
+    # If target is unreachable, dominance is undefined - return False
+    if not target_reachable:
+        return False
+    # Now check if dom_block dominates target_block
+    seen, stack = set(), [entry]
+    while stack:
+        curr_block = stack.pop()
+        if curr_block == dom_block: continue
+        if curr_block == target_block: return False
+        if curr_block in seen: continue
+        seen.add(curr_block)
+        for next_block in next_map.get(curr_block, []):
+            if next_block not in seen:
+                stack.append(next_block)
+    return True
```

**Times when the tool was unhelpful:**  
Here are some issues we encountered during debugging:
1. **Overly complex reachability solutions**: Initially suggested complex multi-pass algorithms for identifying unreachable blocks when a simple DFS was sufficient
2. **Incorrect dominance semantics**: Proposed dominance rules that didn't match standard definitions, particularly around entry block dominance over unreachable blocks
3. **Premature optimization**: Suggested performance improvements (like caching reachability) before we had a working basic implementation

**Conclusion:**  
- **Strengths**: ChatGPT was invaluable for implementing the CHK algorithm correctly, providing the complete `chk.py` implementation with proper reverse postorder traversal and intersection logic. During debugging, once we identified the issues with our existing implementation, it was able to eventually help us with fixes (but it took a lot of negotiating and prompt engineering to make that happen - see "Weaknesses" below). It also helped speed up debugging by implementing check_methods.py correctly immediately (no back-and-forth required).
- **Weaknesses**: Completely struggled with dominance semantics, particularly around unreachable block handling and entry block dominance rules. It often suggested overly complex solutions when simpler approaches (like basic DFS reachability) were sufficient. And it frequently provided incorrect `turnt.toml` syntax and struggled with the specific output file naming conventions we needed for our three verification methods (`df_slow.out`, `df_fast.out`, `df_both.out`), which we had to do ourselves (minor nitpick, but quite annoying regardless).