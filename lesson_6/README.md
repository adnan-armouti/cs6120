How to run files:  

For correctness:  
cd /tests/ && python verify_outputs.py  

For dynamic and static overhead: 
initially we wanted to use brench and .toml for this, but we kept running into errors. We decided on alternative approach for dynamic and static instruction count:
python3 measure_dynamic_overhead.py
python3 measure_static_overhead.py


GAI Tools Disclaimer:  

**We used OpenAI's ChatGPT for the following:**
1. In the to_ssa.py file, it helped fix the phi placement algorithm by removing the overly restrictive has_def_in_pred filter that was preventing necessary phi nodes from being placed.
2. It proposed the correct approach for handling undefined values along paths by creating real `undef` instructions in predecessor blocks instead of using string placeholders.
3. It helped implement the make_undef_in_pred() function and modify the patch_phi_inputs() logic to properly wire phi node arguments.
4. It assisted with removing the special casing in from_ssa.py that was incorrectly skipping copies from undef values.
5. It also provided quick scaffolding for verifiy_outputs.py so we could get a very quick detailed overview on the correctness of our SSA round trip.
6. Please note that we also used Github Copilot when writing code throughout this project.

**Concrete example of generated code or Q&A output:**  
ChatGPT helped formalize injecting typed undef instructions in predecessors and indexing phi node arguments by label position. Here's the diff for to_ssa.py:
```diff
@@
 def place_phis(blocks, next_map, df_map, defs, vtype):
-    prev_map = invert_next_to_prev(blocks, next_map)
+    prev_map = invert_next_to_prev(blocks, next_map)
     have_phi = set()  # (block_name, var)
     name_to_block = {b["name"]: b for b in blocks}
@@
-                if key not in have_phi:
-                    # create phi at start of block y
-                    b = name_to_block[y]
-                    preds = prev_map.get(y, [])
-                    # build placeholder phi: fill labels now; args later in rename/patch
-                    phi = {
-                        "op": "phi",
-                        "dest": v,
-                        "type": vtype.get(v, "int"),  # default int if unknown
-                        "args": [v]*len(preds),       # placeholder
-                        "labels": preds[:]            # predecessor block names
-                    }
-                    b["instrs"].insert(0, phi)
+                if key not in have_phi:
+                    # create phi at start of block y; fill labels now, args later
+                    preds = prev_map.get(y, [])
+                    b = name_to_block[y]
+                    phi = {
+                        "op": "phi",
+                        "dest": v,
+                        "type": vtype.get(v, "int"),
+                        "args": [v]*len(preds),
+                        "labels": preds[:]
+                    }
+                    b["instrs"].insert(0, phi)
                     have_phi.add(key)
@@
-def rename(func, blocks, next_map, dom_tree):
+def rename(func, blocks, next_map, dom_tree):
     stack = {}
     fresh = {}
@@
-    def patch_phi_inputs(curr_block_name):
-        # for each successor s, set its phi args at position of curr_block_name
-        for s in next_map.get(curr_block_name, []):
-            sb = name_to_block[s]
-            for ins in sb["instrs"]:
-                if ins.get("op") != "phi":
-                    break
-                dest_now = ins["dest"]
-                base = _base(dest_now)
-                labels = ins.get("labels", [])
-                args = ins.get("args", [])
-                try:
-                    k = labels.index(curr_block_name)
-                except ValueError:
-                    continue
-                # supply current version for 'base' (if any), else leave placeholder
-                if base in stack and stack[base]:
-                    args[k] = stack[base][-1]
-                ins["args"] = args
+    def make_undef_in_pred(pred_block_name, ty):
+        blist = name_to_block[pred_block_name]["instrs"]
+        cut = len(blist) - 1 if (blist and blist[-1].get("op") in ('jmp','br','ret')) else len(blist)
+        tmp = f"__undef.{ty}.{len(blist)}"
+        blist[cut:cut] = [{"op":"undef","dest":tmp,"type":ty}]
+        return tmp
+
+    def patch_phi_inputs(curr_block_name):
+        # set each successor's phi arg at the index for curr_block_name
+        for s in next_map.get(curr_block_name, []):
+            sb = name_to_block[s]
+            for ins in sb["instrs"]:
+                if ins.get("op") != "phi":
+                    break
+                base = _base(ins["dest"])
+                labels, args = ins.get("labels", []), ins.get("args", [])
+                try:
+                    k = labels.index(curr_block_name)
+                except ValueError:
+                    continue
+                ty = ins.get("type", "int")
+                args[k] = stack[base][-1] if (base in stack and stack[base]) else make_undef_in_pred(curr_block_name, ty)
+                ins["args"] = args
@@
-    idom, dom_tree = df.get_immediate_doms_and_tree(doms_map)
+    idom, dom_tree = df.get_immediate_doms_and_tree(doms_map, next_map)
     df_map = df.get_dominance_frontier(blocks, next_map, idom, dom_tree)
```

For from_ssa.py: ChatGPT helped formalize keeping undef sources when lowering phi nodes:
```diff
@@
 def lower_phis(func):
@@
-            for k, pred in enumerate(labels):
-                arg = args[k]
-                copies_for_pred[pred].append((dest, arg, ty))
+            for k, pred in enumerate(labels):
+                arg = args[k]
+                # carry all sources, including explicit 'undef' values
+                copies_for_pred[pred].append((dest, arg, ty))
             i += 1
```

The main issue was that our original implementation was skipping copies from undef values, which meant that some phi node destinations weren't receiving the correct incoming values, breaking the value chain and causing the gpf.bril test to fail (output changed from 29 to 1). This was really the first file that we decided to tackle, illustrated the problems with our initial approach, and we found that by fixing the issue with this benchmark, we were then able to pass most of the others.

**Times when the tool was unhelpful:**  
Here are some issues we encountered during debugging (which mostly had to do with handling the undef instruction in Bril):
1. **Incorrect phi node placement semantics**: Initially suggested complex heuristics for phi node placement when the standard dominance frontier algorithm was sufficient
2. **String placeholder approach**: Proposed using `__undef_` string placeholders instead of real Bril `undef` instructions, which the interpreter doesn't understand
3. **Overly complex undef handling**: Suggested complex multi-pass algorithms for handling undefined values when inserting `undef` instructions in predecessors was the correct approach

**Conclusion:**  
- **Strengths**: ChatGPT was pretty helpful for identifying the root cause of our SSA round-trip failures. It correctly diagnosed that the `has_def_in_pred` filter was too restrictive and that we needed to use real `undef` instructions instead of string placeholders. It also helped implement the `make_undef_in_pred()` function correctly and provided the proper logic for wiring phi node arguments.
- **Weaknesses**: Initially struggled with understanding the standard SSA construction algorithm, particularly around phi node placement via dominance frontiers. It often suggested overly complex solutions when the textbook approach was sufficient, and it also initially suggested using string placeholders for undefined values, which doesn't work with the Bril interpreter. It took several iterations to get the correct understanding of how to handle "undefined along a path" scenarios in SSA form.