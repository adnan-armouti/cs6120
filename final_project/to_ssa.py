import sys, json
import cfg
import df

def invert_next_to_prev(blocks, next_map):
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for u, outs in next_map.items():
        for v in outs:
            prev_map[v].append(u)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx[n])
    return prev_map


def compute_liveness(blocks, next_map):
    prev_map = invert_next_to_prev(blocks, next_map)

    use_sets = {}
    def_sets = {}
    for b in blocks:
        uses, defs = set(), set()
        for ins in b["instrs"]:
            for arg in ins.get("args", []):
                if isinstance(arg, str) and arg not in defs:
                    uses.add(arg)
            if "dest" in ins:
                defs.add(ins["dest"])
        use_sets[b["name"]] = uses
        def_sets[b["name"]] = defs

    live_in = {b["name"]: set() for b in blocks}
    live_out = {b["name"]: set() for b in blocks}

    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            name = b["name"]
            new_out = set()
            for succ in next_map.get(name, []):
                new_out |= live_in[succ]

            new_in = use_sets[name] | (new_out - def_sets[name])

            if new_in != live_in[name] or new_out != live_out[name]:
                changed = True
                live_in[name] = new_in
                live_out[name] = new_out

    return live_in, live_out

def flatten_blocks(blocks):
    instrs = []
    for b in blocks:
        if b["label"] is not None:
            instrs.append({"label": b["label"]})
        instrs.extend(b["instrs"])
    return instrs

def _base(name):
    if isinstance(name, str) and "." in name:
        return name.rsplit(".", 1)[0]
    return name

def collect_defs_and_types(func):
    defs = {}
    vtype = {}
    for a in func.get("args", []):
        vtype[a["name"]] = a["type"]

    for ins in func.get("instrs", []):
        if "op" in ins and "dest" in ins:
            v = ins["dest"]
            t = ins.get("type")
            if t is not None and v not in vtype:
                vtype[v] = t
    return defs, vtype

def seed_block_defs(blocks, defs):
    for b in blocks:
        for ins in b["instrs"]:
            if "op" in ins and "dest" in ins:
                v = ins["dest"]
                defs.setdefault(v, set()).add(b["name"])

def place_phis(blocks, next_map, df_map, defs, vtype, live_in=None):
    prev_map = invert_next_to_prev(blocks, next_map)
    have_phi = set()
    name_to_block = {b["name"]: b for b in blocks}

    for v, def_blocks in list(defs.items()):
        W = list(def_blocks)
        seen = set(W)
        while W:
            n = W.pop()
            for y in df_map.get(n, set()):
                key = (y, v)
                if key not in have_phi:
                    if live_in is not None and v not in live_in.get(y, set()):
                        continue

                    preds = prev_map.get(y, [])
                    b = name_to_block[y]
                    phi = {
                        "op": "phi",
                        "dest": v,
                        "type": vtype.get(v, "int"),
                        "args": [v]*len(preds),
                        "labels": preds[:]
                    }
                    b["instrs"].insert(0, phi)
                    have_phi.add(key)
                    if y not in defs.get(v, set()):
                        defs.setdefault(v, set()).add(y)
                        if y not in seen:
                            W.append(y)
                            seen.add(y)

def fresh_name(fresh, x):
    k = fresh.get(x, 0) + 1
    fresh[x] = k
    return f"{x}.{k}"

def rewrite_args(ins, stack):
    if "args" in ins:
        new = []
        for a in ins["args"]:
            if isinstance(a, str) and a in stack and stack[a]:
                new.append(stack[a][-1])
            else:
                new.append(a)
        ins["args"] = new

def make_undef_in_pred(name_to_block, pred_block_name, ty):
    blist = name_to_block[pred_block_name]["instrs"]
    cut = len(blist) - 1 if (blist and blist[-1].get("op") in ('jmp','br','ret')) else len(blist)
    tmp = f"__undef.{ty}.{len(blist)}"
    u = {"op":"undef", "dest": tmp, "type": ty}
    blist[cut:cut] = [u]
    return tmp

def patch_phi_inputs(curr_block_name, next_map, name_to_block, stack):
    for s in next_map.get(curr_block_name, []):
        sb = name_to_block[s]
        for ins in sb["instrs"]:
            if ins.get("op") != "phi":
                break
            dest_now = ins["dest"]
            base = _base(dest_now)
            labels = ins.get("labels", [])
            args = ins.get("args", [])
            try:
                k = labels.index(curr_block_name)
            except ValueError:
                continue
            ty = ins.get("type", "int")
            if base in stack and stack[base]:
                args[k] = stack[base][-1]
            else:
                args[k] = make_undef_in_pred(name_to_block, curr_block_name, ty)
            ins["args"] = args

def dfs_rename(bname, name_to_block, next_map, dom_tree, stack, fresh):
    b = name_to_block[bname]
    pushed = []
    i = 0
    while i < len(b["instrs"]) and b["instrs"][i].get("op") == "phi":
        ins = b["instrs"][i]
        v = _base(ins["dest"])
        new = fresh_name(fresh, v)
        ins["dest"] = new
        stack.setdefault(v, []).append(new)
        pushed.append(v)
        i += 1
    for ins in b["instrs"][i:]:
        rewrite_args(ins, stack)
        if "op" in ins and "dest" in ins:
            v = _base(ins["dest"])
            new = fresh_name(fresh, v)
            ins["dest"] = new
            stack.setdefault(v, []).append(new)
            pushed.append(v)
    patch_phi_inputs(bname, next_map, name_to_block, stack)
    for child in dom_tree.get(bname, []):
        dfs_rename(child, name_to_block, next_map, dom_tree, stack, fresh)
    for v in reversed(pushed):
        stack[v].pop()

def rename(func, blocks, next_map, dom_tree):
    stack = {}
    fresh = {}
    for a in func.get("args", []):
        nm = a["name"]
        stack.setdefault(nm, []).append(nm)
        fresh.setdefault(nm, 0)
    name_to_block = {b["name"]: b for b in blocks}
    order = [b["name"] for b in blocks]
    entry = order[0]
    dfs_rename(entry, name_to_block, next_map, dom_tree, stack, fresh)

def run(func, use_pruned_ssa=True):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)

    doms_map = df.get_dom_sets(blocks, next_map)
    idom, dom_tree = df.get_immediate_doms_and_tree(doms_map, next_map)
    df_map = df.get_dominance_frontier(blocks, next_map, idom, dom_tree)

    defs, vtype = collect_defs_and_types(func)
    seed_block_defs(blocks, defs)

    live_in = None
    if use_pruned_ssa:
        live_in, _ = compute_liveness(blocks, next_map)

    place_phis(blocks, next_map, df_map, defs, vtype, live_in)
    rename(func, blocks, next_map, dom_tree)

    func["instrs"] = flatten_blocks(blocks)

if __name__ == "__main__":
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        run(f)
    print(json.dumps(prog, indent=None, separators=(",",":")))
