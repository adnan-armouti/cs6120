import sys, json
import cfg

TERMINATORS = ('jmp', 'br', 'ret')

def invert_next_to_prev(blocks, next_map):
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for u, outs in next_map.items():
        for v in outs:
            prev_map[v].append(u)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx[n])
    return prev_map

def flatten_blocks(blocks):
    instrs = []
    for b in blocks:
        if b["label"] is not None:
            instrs.append({"label": b["label"]})
        instrs.extend(b["instrs"])
    return instrs

def _insert_before_terminator(blist, new_instrs):
    if blist and blist[-1].get("op") in TERMINATORS:
        cut = len(blist) - 1
        blist[cut:cut] = new_instrs
    else:
        blist.extend(new_instrs)

def _parallel_copies_to_seq(copies):
    seq = []
    temp_i = 0

    while copies:
        dests = {d for (d, s, t) in copies}
        sources = {s for (d, s, t) in copies}

        picked = None
        for i, (d, s, t) in enumerate(copies):
            source_safe = (s not in dests) or (s == d)
            dest_safe = (d not in sources) or (d == s)

            if source_safe and dest_safe:
                picked = i
                break

        if picked is not None:
            d, s, t = copies.pop(picked)
            if d != s:
                seq.append({"op": "id", "dest": d, "type": t, "args": [s]})
            continue

        cycle_var = None
        cycle_type = None
        for (d, s, t) in copies:
            if d in sources and d in dests:
                cycle_var = d
                cycle_type = t
                break

        if cycle_var is None:
            for (d, s, t) in copies:
                if d in sources:
                    cycle_var = d
                    cycle_type = t
                    break

        if cycle_var is None:
            for (d, s, t) in copies:
                if s in dests:
                    cycle_var = s
                    cycle_type = t
                    break

        if cycle_var is None:
            d, s, t = copies.pop(0)
            if d != s:
                seq.append({"op": "id", "dest": d, "type": t, "args": [s]})
            continue

        tmp = f"__phi_tmp{temp_i}"
        temp_i += 1
        seq.append({"op": "id", "dest": tmp, "type": cycle_type, "args": [cycle_var]})

        new_copies = []
        for (dd, ss, tt) in copies:
            new_copies.append((dd, tmp if ss == cycle_var else ss, tt))
        copies = new_copies

    return seq

def lower_phis(func):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    prev_map = invert_next_to_prev(blocks, next_map)
    name_to_block = {b["name"]: b for b in blocks}
    copies_for_pred = {b["name"]: [] for b in blocks}

    for b in blocks:
        i = 0
        while i < len(b["instrs"]) and b["instrs"][i].get("op") == "phi":
            phi = b["instrs"][i]
            dest = phi["dest"]
            ty   = phi.get("type", "int")
            labels = phi.get("labels", [])
            args = phi.get("args", [])
            for k, pred in enumerate(labels):
                arg = args[k]
                if arg:
                    copies_for_pred[pred].append((dest, arg, ty))
            i += 1

    for pname, moves in copies_for_pred.items():
        if not moves: continue
        seq = _parallel_copies_to_seq(moves[:])
        blist = name_to_block[pname]["instrs"]
        _insert_before_terminator(blist, seq)

    for b in blocks:
        new_instrs = []
        for ins in b["instrs"]:
            if ins.get("op") == "phi":
                continue
            new_instrs.append(ins)
        b["instrs"] = new_instrs

    func["instrs"] = flatten_blocks(blocks)

if __name__ == "__main__":
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        lower_phis(f)
    print(json.dumps(prog, indent=None, separators=(",",":")))
