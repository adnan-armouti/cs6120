GAI Tools Disclaimer:  

**We used OpenAI's ChatGPT for the following:**
1. answer questions on Bril's syntax, which was helpful when coming up with the op_ecounter.py file  
2. building on this file, it helped with the yield_blocks() helper function implementation in cfg.py file by suggesting the state dict variable for readability  
3. brainstorm a simple but useful benchmark -> idea behind the closed form equation for the arithmetic series formula.
We also used the Github CoPilot Feature to create a quick scaffold for each .py file ebfore iteratively refining and testing.  

**Concrete example of generated code or Q&A output:**  
Any use of the state dictionary in the yield_block function in the cfg.py file, copied below for ease of reference:
```
def yield_block(state, blocks):
    if not state['instrs'] and state['label'] is None:
        return
    name = state['label'] if state['label'] is not None else f"B{state['next_id']}"
    if state['label'] is None:
        state['next_id'] += 1
    blocks.append({"name": name, "label": state['label'], "instrs": state['instrs']})
    state['label'] = None
    state['instrs'] = []
```

**Times when the tool was unhelpful:**  
The Turnt guidance it provided was not reliable, and I ultimately ended up ignoring it and we followed Adnan's September 9 class notes during the programming section to understand hwo to use Turnt and how it interfaces with Bril, especially when attempting to generate the .out and .prof files.

**Conclusion:**  
- Strengths: coming up with a quick and simple benchmark -> we really liked the arithmetic series suggestion because of how familiar and easy to implement it was. In addition, it was helpful in pointing out ways to improve the readability of our code through the state dictionary when implementing the blocks and control flow graphy python file.  
- Weaknesses: Completely hallucinated information on Turnt, and it really failed to grasp the way we are supposed to use it for testing. Instead of wasting time trying to coach it through the process, we will definitely just refer to our class notes to guide us through any testing surrounding compiler optimization/implementation.