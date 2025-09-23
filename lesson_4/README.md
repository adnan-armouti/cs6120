GAI Tools Disclaimer:  

**We used OpenAI's ChatGPT for the following:**
1. In the dfa.py file, implemented the while loops in the solver algorithm provided in class 
2. Optimize the prepare() function in each data flow analysis method, and double check all functions for syntax and logic errors.
3. Brainstorm a set of simple but useful benchmarks to ensure that the basic ideas of each data flow analysis method are implemented correctly.  

**Concrete example of generated code or Q&A output:**  
The two while loops within the solve() function in the dfa.py file:
```
    def solve(self, func: dict, blocks: List[dict], next_map: Dict[str, List[str]]):
        if not blocks:
            return {}, {}
        self.prepare(func, blocks)
        names = [b["name"] for b in blocks]
        entry = names[0]
        prev = self._prev_map(blocks, next_map)

        in_dict  = {n: set(self.initial_in(n))  for n in names}
        out_dict = {n: set(self.initial_out(n)) for n in names}
        order = {n: i for i, n in enumerate(names)}
        work = set(names)

        if self.direction == "forward":
            while work:
                # pick earliest by program order (no nested helper)
                b = min(work, key=lambda k: order[k])
                work.remove(b)
                if b != entry:
                    in_dict[b] = self.merge([out_dict[p] for p in prev[b]])
                new_out = self.transfer(b, in_dict[b])
                if new_out != out_dict[b]:
                    out_dict[b] = new_out
                    for n in next_map.get(b, []):
                        work.add(n)
        else:
            while work:
                b = min(work, key=lambda k: order[k])
                work.remove(b)
                next = next_map.get(b, [])
                if next:
                    out_dict[b] = self.merge([in_dict[s] for s in next])
                new_in = self.transfer(b, out_dict[b])
                if new_in != in_dict[b]:
                    in_dict[b] = new_in
                    for p in prev[b]:
                        work.add(p)

        return in_dict, out_dict
```

**Times when the tool was unhelpful:**  
Here are some weird, minor inefficiencies it kept trying to force on us:
1. including this import line 
```
from typing import Dict, List, Set
```
which was completely unnecessary for implementing this task. 
2. related to above, when double checking the merge() function for initialized_variables.py, it kept recommending:
```
    def merge(self, values):
        acc = set(self._all_vars)
        for s in values: acc &= s
        return acc
```
instead of the faster, more readable:
```
    def merge(self, values):
        return set.intersection(*values) if values else set(self._all_vars)
```
3. adding multiple unnecessary helper functions when implementing the prepare() function -> to the point where tracing simple calls became very convoluted and messy.

**Conclusion:**  
- Strengths: it really helped implement the solver loop quite quickly, which we verified against the pseudocode provided in class, and helped with refactoring our initial prepare() functions for readability and efficiency. We did also ask it to provide additional benchmarks to test against, and it did a good job of providing edge case to test against but only when explicitly told to do so. Please note that in this repo, you will only find the bril benchmarks provided in the "bril/examples/test/df" repo directory to serve as examples for how we tested these files (and especially since we can compare directly against the provided .out files in the case of the live variables method).
- Weaknesses: In addition to everythin already listed above, it almost always hallucinated Turnt details (e.g., non-existent placeholders/keys) and TOML semantics. Since we had three data flow anaylsis implementations to test, and to stay consistent with the naming convention for .out files in "bril/examples/test/df", we wanted a unique {filename}.{data_flow_analysis_method}.out file for each benchmark (and to therefore avoid overwriting just a single .out file for each benchmark). We eventually enabled this oursevles after playing around with turnt (e.g., per-env output.<ext> = "-", and quoting dotted extensions like "rd.out"), but it would have been nice if it was able to do this for us.

**Additional Acknowledgements:**
We would like to thank and credit our classmate Jake for the discussion around our approach to this assignment - he was the one who suggested the modular parent/child class implementation you see in this repo to speed up debugging.