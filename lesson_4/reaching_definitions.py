import sys, json
import cfg
from dfa import DataFlowAnalysis

class ReachingDefinitions(DataFlowAnalysis):
    def __init__(self):
        super().__init__()
        self.direction = "forward"
        self.defs_dict = {}   # block -> set[("def", block, idx)]
        self.kill_dict = {}   # block -> set[DefId]
        self._args = []
        self._entry = ""

    def prepare(self, func, blocks):
        # Seed
        self._args = [a["name"] for a in func.get("args", [])]
        if blocks: self._entry = blocks[0]["name"]
        # First collect all defs by variable across the function for kill_dict
        all_defs_by_var = {}
        for b in blocks:
            bn = b["name"]
            for i, ins in enumerate(b["instrs"]):
                d = ins.get("dest")
                if d:
                    all_defs_by_var.setdefault(d, set()).add(("def", bn, i))
        for a in self._args:
            all_defs_by_var.setdefault(a, set()).add(("arg", a))   # arg “defs” at entry
        # Then collect per-block defs_dict and kill_dict
        for b in blocks:
            bn = b["name"]
            last = {}
            vars_in_b = set()
            for i, ins in enumerate(b["instrs"]):
                d = ins.get("dest")
                if d:
                    last[d] = ("def", bn, i)
                    vars_in_b.add(d)
            defs = set(last.values())
            kill = set()
            for v in vars_in_b:
                keep = {last[v]} if v in last else set()
                kill |= (all_defs_by_var.get(v, set()) - keep)
            self.defs_dict[bn] = defs
            self.kill_dict[bn] = kill
    
    def merge(self, values):
        return set().union(*values) if values else set()
    
    def transfer(self, block_name, in_set):
        return self.defs_dict[block_name] | (in_set - self.kill_dict[block_name])
    
    def initial_in(self, block_name):
        return {("arg", a) for a in self._args} if block_name == self._entry else set()
    
    def initial_out(self, block_name):
        return set()

def _fmt(s):
    return "{" + ", ".join(sorted(map(str, s))) + "}" if s else "{}"

def analyze_function(func: dict):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    rd = ReachingDefinitions()
    in_list, out_list = rd.solve(func, blocks, next_map)

    print(f"{func['name']}:")
    for b in blocks:
        n = b["name"]
        print(f"{n}:")
        print(f"  in:  {_fmt(in_list[n])}")
        print(f"  out: {_fmt(out_list[n])}")
        # print(f"  defs: {_fmt(rd.defs_dict[n])}")
        # print(f"  kill: {_fmt(rd.kill_dict[n])}")

def main():
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        analyze_function(f)

if __name__ == "__main__":
    main()
