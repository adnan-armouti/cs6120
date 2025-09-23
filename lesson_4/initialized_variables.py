from __future__ import annotations
from typing import Dict, List, Set
import sys, json
import cfg
from dfa import DataFlowAnalysis

class InitializedVariables(DataFlowAnalysis):
    def __init__(self):
        super().__init__()
        self.direction = "forward"
        self.defs_dict: Dict[str, Set[str]] = {}
        self._all_vars: Set[str] = set()
        self._args: Set[str] = set()
        self._entry: str = ""
    
    def prepare(self, func: dict, blocks: List[dict]) -> None:
        if blocks:
            self._entry = blocks[0]["name"]
        # TOP = all variables that appear anywhere (args, dests, used)
        all_vars: Set[str] = set(a["name"] for a in func.get("args", []))
        for b in blocks:
            for ins in b["instrs"]:
                for x in ins.get("args", []): all_vars.add(x)
                d = ins.get("dest")
                if d: all_vars.add(d)
        self._all_vars = all_vars
        self._args = set(a["name"] for a in func.get("args", []))
        # defs per block (definitely assigned variables)
        for b in blocks:
            defs = set()
            for ins in b["instrs"]:
                d = ins.get("dest")
                if d: defs.add(d)
            self.defs_dict[b["name"]] = defs
    
    def merge(self, values):
        return set.intersection(*values) if values else set(self._all_vars)

    def transfer(self, block_name, in_set):
        return in_set | self.defs_dict[block_name]

    def initial_in(self, block_name):
        return set(self._args) if block_name == self._entry else set(self._all_vars)

    def initial_out(self, block_name):
        return set(self._all_vars)

def _fmt(s):
    return "{" + ", ".join(sorted(map(str, s))) + "}" if s else "{}"

def analyze_function(func: dict):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    iv = InitializedVariables()
    in_list, out_list = iv.solve(func, blocks, next_map)
    
    print(f"{func['name']}:")
    for b in blocks:
        n = b["name"]
        print(f"{n}:")
        print(f"  in:  {_fmt(in_list[n])}")
        print(f"  out: {_fmt(out_list[n])}")
        # print(f"  defs: {_fmt(iv.defs_dict[n])}")

def main():
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        analyze_function(f)

if __name__ == "__main__":
    main()
