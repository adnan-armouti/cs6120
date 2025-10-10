import sys, json
import cfg

def invert_next_to_prev(blocks, next_map):
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for u, outs in next_map.items():
        for v in outs:
            prev_map[v].append(u)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx[n])
    return prev_map

def check(func):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    prev_map = invert_next_to_prev(blocks, next_map)
    name_to_block = {b["name"]: b for b in blocks}

    seen = set()
    for ins in func.get("instrs", []):
        if "op" in ins and "dest" in ins:
            d = ins["dest"]
            if d in seen:
                return False
            seen.add(d)

    args_ok = set(a["name"] for a in func.get("args", []))
    all_defs = set(args_ok) | seen
    for ins in func.get("instrs", []):
        if "args" in ins:
            for a in ins["args"]:
                if isinstance(a, str) and a not in all_defs:
                    return False

    for b in blocks:
        preds = prev_map.get(b["name"], [])
        pred_set = set(preds)
        i = 0
        while i < len(b["instrs"]) and b["instrs"][i].get("op") == "phi":
            phi = b["instrs"][i]
            labels = phi.get("labels", [])
            args   = phi.get("args", [])
            if len(labels) != len(args): return False
            if len(labels) != len(preds): return False
            if set(labels) != pred_set:   return False
            i += 1

    return True

def main():
    prog = json.load(sys.stdin)
    ok = True
    for f in prog.get("functions", []):
        if not check(f):
            ok = False
            break
    print("yes" if ok else "no")

if __name__ == "__main__":
    main()
