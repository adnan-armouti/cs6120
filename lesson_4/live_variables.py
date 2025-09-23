import sys, json
import cfg
from dfa import DataFlowAnalysis

class LiveVariables(DataFlowAnalysis):
    def __init__(self):
        super().__init__()
        self.direction = "backward"
        self.use_dict = {}
        self.kill_dict = {}
    
    def prepare(self, func, blocks):
        for b in blocks:
            use, kill = set(), set()
            for ins in b["instrs"]:
                for a in ins.get("args", []):
                    if a not in kill:
                        use.add(a)
                d = ins.get("dest")
                if d:
                    kill.add(d)
            self.use_dict[b["name"]]  = use
            self.kill_dict[b["name"]] = kill

    def merge(self, values):
        return set().union(*values) if values else set()

    def transfer(self, block_name, out_set):
        return self.use_dict[block_name] | (out_set - self.kill_dict[block_name])
    
    def initial_in(self, block_name):
        return set()

    def initial_out(self, block_name):
        return set()

def _fmt(s):
    return "{" + ", ".join(sorted(map(str, s))) + "}" if s else "{}"

def analyze_function(func: dict):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    lv = LiveVariables()
    in_list, out_list = lv.solve(func, blocks, next_map)

    print(f"{func['name']}:")
    for b in blocks:
        n = b["name"]
        print(f"{n}:")
        print(f"  in:  {_fmt(in_list[n])}")
        print(f"  out: {_fmt(out_list[n])}")
        # print(f"  use: {_fmt(lv.use_dict[n])}")
        # print(f"  kill: {_fmt(lv.kill_dict[n])}")

def main():
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        analyze_function(f)

if __name__ == "__main__":
    main()
