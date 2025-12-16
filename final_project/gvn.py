from __future__ import annotations

import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any

import cfg
import df
import to_ssa
import from_ssa
import tdce

TERMINATORS = {"br", "jmp", "ret"}

@dataclass(frozen=True)
class VN:
    n: int

PURE_BIN: Set[str] = {
    "add", "sub", "mul",
    "and", "or", "xor",
    "eq", "lt", "le", "gt", "ge",
    "fadd", "fsub", "fmul", "fdiv",
    "feq", "flt", "fle", "fgt", "fge",
}
PURE_UN: Set[str] = {"not", "id"}
PURE_MISC: Set[str] = {"const"}
PURE_OPS: Set[str] = PURE_BIN | PURE_UN | PURE_MISC

COMMUTATIVE: Set[str] = {
    "add", "mul", "and", "or", "xor", "eq",
    "fadd", "fmul", "feq",
}

REL_FLIPS = {
    "gt": "lt",
    "ge": "le",
    "fgt": "flt",
    "fge": "fle",
}


def type_key(ty: Any) -> Optional[str]:
    if ty is None:
        return None
    if isinstance(ty, (dict, list)):
        try:
            return json.dumps(ty, sort_keys=True)
        except (TypeError, ValueError):
            return str(ty)
    return str(ty)


def rep_key(vn: VN, ty: Any) -> Tuple[int, Optional[str]]:
    return (vn.n, type_key(ty))


def is_pure(instr: Dict[str, Any]) -> bool:
    op = instr.get("op")
    if op is None:
        return False
    return op in PURE_OPS


def expr_key(instr: Dict[str, Any], arg_vns: List[int]) -> Optional[Tuple]:
    op = instr.get("op")
    ty = type_key(instr.get("type"))

    if op == "const":
        return ("const", ty, instr.get("value"))

    if op == "id":
        return None

    if op in REL_FLIPS:
        op = REL_FLIPS[op]
        arg_vns = list(reversed(arg_vns))

    if op in PURE_BIN or op in PURE_UN:
        key_args = arg_vns[:]
        if op in COMMUTATIVE and len(key_args) == 2:
            if key_args[0] > key_args[1]:
                key_args = [key_args[1], key_args[0]]
        return (op, ty, *key_args)

    return None


class GVNContext:
    def __init__(self):
        self.vn_counter = 0
        self.expr2vn_stack: List[Dict[Tuple, VN]] = []
        self.vn_of: Dict[str, VN] = {}

    def new_vn(self) -> VN:
        self.vn_counter += 1
        return VN(self.vn_counter)

    def push_scope(self):
        self.expr2vn_stack.append({})

    def pop_scope(self):
        if self.expr2vn_stack:
            self.expr2vn_stack.pop()

    def lookup_expr(self, key: Tuple) -> Optional[VN]:
        for scope in reversed(self.expr2vn_stack):
            if key in scope:
                return scope[key]
        return None

    def insert_expr(self, key: Tuple, vn: VN):
        if not self.expr2vn_stack:
            self.expr2vn_stack.append({})
        self.expr2vn_stack[-1][key] = vn

    def clear_all_scopes(self):
        for scope in self.expr2vn_stack:
            scope.clear()

    def get_vn(self, name: str) -> VN:
        if name not in self.vn_of:
            self.vn_of[name] = self.new_vn()
        return self.vn_of[name]


def flatten_blocks(blocks: List[dict]) -> List[dict]:
    instrs = []
    for b in blocks:
        if b.get("label") is not None:
            instrs.append({"label": b["label"]})
        instrs.extend(b.get("instrs", []))
    return instrs


def bind(dest: str, vn: VN, ty: Any, ctx: GVNContext,
         env: Dict[str, VN], val2rep: Dict[Tuple, str]):
    ctx.vn_of[dest] = vn
    env[dest] = vn
    rk = rep_key(vn, ty)
    if rk not in val2rep:
        val2rep[rk] = dest


def process_phi(phi: dict, ctx: GVNContext, env: Dict[str, VN],
                val2rep: Dict[Tuple, str], has_backedge: bool) -> VN:
    dest = phi["dest"]
    args = phi.get("args", [])
    ty = phi.get("type")

    if has_backedge:
        vn = ctx.new_vn()
        bind(dest, vn, ty, ctx, env, val2rep)
        return vn

    arg_vns = []
    for arg in args:
        arg_vns.append(ctx.get_vn(arg).n)

    if arg_vns and all(v == arg_vns[0] for v in arg_vns):
        vn = VN(arg_vns[0])
        bind(dest, vn, ty, ctx, env, val2rep)
        return vn

    phi_key = ("phi", type_key(ty), tuple(arg_vns))
    existing = ctx.lookup_expr(phi_key)
    if existing:
        vn = existing
    else:
        vn = ctx.new_vn()
        ctx.insert_expr(phi_key, vn)

    bind(dest, vn, ty, ctx, env, val2rep)
    return vn


def process_const(instr: dict, ctx: GVNContext, env: Dict[str, VN],
                  val2rep: Dict[Tuple, str]) -> dict:
    dest = instr.get("dest")
    ty = instr.get("type")

    key = expr_key(instr, [])
    existing = ctx.lookup_expr(key)

    if existing:
        rep = val2rep.get(rep_key(existing, ty))
        if rep and env.get(rep) == existing:
            return {
                "op": "id",
                "dest": dest,
                "type": ty,
                "args": [rep]
            }
        else:
            bind(dest, existing, ty, ctx, env, val2rep)
            return instr
    else:
        vn = ctx.new_vn()
        ctx.insert_expr(key, vn)
        bind(dest, vn, ty, ctx, env, val2rep)
        return instr


def process_id(instr: dict, ctx: GVNContext, env: Dict[str, VN],
               val2rep: Dict[Tuple, str]) -> dict:
    dest = instr.get("dest")
    args = instr.get("args", [])
    ty = instr.get("type")

    if not args:
        vn = ctx.new_vn()
        bind(dest, vn, ty, ctx, env, val2rep)
        return instr

    src = args[0]
    vn = ctx.get_vn(src)

    ctx.vn_of[dest] = vn
    env[dest] = vn

    rk = rep_key(vn, ty)
    if rk not in val2rep or env.get(val2rep.get(rk, "")) != vn:
        val2rep[rk] = dest

    return instr


def process_pure_op(instr: dict, ctx: GVNContext, env: Dict[str, VN],
                    val2rep: Dict[Tuple, str]) -> dict:
    dest = instr.get("dest")
    args = instr.get("args", [])
    ty = instr.get("type")

    arg_vns = [ctx.get_vn(a).n for a in args]

    key = expr_key(instr, arg_vns)
    if key is None:
        vn = ctx.new_vn()
        bind(dest, vn, ty, ctx, env, val2rep)
        return instr

    existing = ctx.lookup_expr(key)
    if existing:
        rep = val2rep.get(rep_key(existing, ty))
        if rep and env.get(rep) == existing:
            return {
                "op": "id",
                "dest": dest,
                "type": ty,
                "args": [rep]
            }
        else:
            bind(dest, existing, ty, ctx, env, val2rep)
            return instr
    else:
        vn = ctx.new_vn()
        ctx.insert_expr(key, vn)
        bind(dest, vn, ty, ctx, env, val2rep)
        return instr


def gvn_walk(block_name: str, ctx: GVNContext, env: Dict[str, VN],
             val2rep: Dict[Tuple, str], bmap: Dict[str, dict],
             dom_tree: Dict[str, List[str]], dom_sets: Dict[str, Set[str]],
             next_map: Dict[str, List[str]], prev_map: Dict[str, List[str]]):
    block = bmap[block_name]
    instrs = block.get("instrs", [])
    new_instrs = []

    ctx.push_scope()

    preds = prev_map.get(block_name, [])
    has_backedge = any(
        block_name in dom_sets.get(pred, set())
        for pred in preds
    )

    i = 0
    while i < len(instrs) and instrs[i].get("op") == "phi":
        phi = instrs[i]
        process_phi(phi, ctx, env, val2rep, has_backedge)
        new_instrs.append(phi)
        i += 1

    for instr in instrs[i:]:
        op = instr.get("op")
        if "label" in instr:
            new_instrs.append(instr)
            continue
        if op in TERMINATORS:
            new_instrs.append(instr)
            continue
        if op == "undef":
            dest = instr.get("dest")
            if dest:
                vn = ctx.new_vn()
                bind(dest, vn, instr.get("type"), ctx, env, val2rep)
            new_instrs.append(instr)
            continue
        if not is_pure(instr):
            dest = instr.get("dest")
            if dest:
                vn = ctx.new_vn()
                bind(dest, vn, instr.get("type"), ctx, env, val2rep)
            ctx.clear_all_scopes()
            new_instrs.append(instr)
            continue

        if op == "const":
            new_instrs.append(process_const(instr, ctx, env, val2rep))
        elif op == "id":
            new_instrs.append(process_id(instr, ctx, env, val2rep))
        else:
            new_instrs.append(process_pure_op(instr, ctx, env, val2rep))

    block["instrs"] = new_instrs

    for child in dom_tree.get(block_name, []):
        gvn_walk(child, ctx, env.copy(), val2rep.copy(), bmap,
                 dom_tree, dom_sets, next_map, prev_map)

    ctx.pop_scope()


def invert_to_prev(blocks: List[dict], next_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for u, succs in next_map.items():
        for v in succs:
            if v in prev_map:
                prev_map[v].append(u)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx.get(n, 0))
    return prev_map


def eliminate_copies(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    rename: Dict[str, str] = {}

    def canon(x: str) -> str:
        visited = set()
        while x in rename and rename[x] != x and x not in visited:
            visited.add(x)
            x = rename[x]
        return x

    new_instrs = []
    for instr in instrs:
        op = instr.get("op")
        if "label" in instr:
            new_instrs.append(instr)
            continue
        if "args" in instr:
            instr["args"] = [canon(a) for a in instr["args"]]
        if op == "id":
            args = instr.get("args", [])
            if args:
                src = canon(args[0])
                dest = instr.get("dest")
                if dest and src != dest:
                    rename[dest] = src
                    continue

        if "dest" in instr:
            rename[instr["dest"]] = instr["dest"]
        new_instrs.append(instr)

    func["instrs"] = new_instrs


def gvn_on_function(func: dict):
    blocks = cfg.form_blocks(func.get("instrs", []))
    if not blocks:
        return

    next_map = cfg.form_cfg(blocks)
    prev_map = invert_to_prev(blocks, next_map)
    dom_sets = df.get_dom_sets(blocks, next_map)
    _, dom_tree = df.get_immediate_doms_and_tree(dom_sets, next_map)

    bmap = {b["name"]: b for b in blocks}
    entry = blocks[0]["name"]

    ctx = GVNContext()
    for arg in func.get("args", []):
        name = arg["name"]
        ctx.vn_of[name] = ctx.new_vn()

    gvn_walk(entry, ctx, {}, {}, bmap, dom_tree, dom_sets, next_map, prev_map)

    func["instrs"] = flatten_blocks(blocks)


def ssa_copy_propagation(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    phi_dests: Set[str] = set()
    for instr in instrs:
        if instr.get("op") == "phi":
            dest = instr.get("dest")
            if dest:
                phi_dests.add(dest)

    copy_map: Dict[str, str] = {}
    for instr in instrs:
        if instr.get("op") == "id" and "args" in instr and instr["args"]:
            dest = instr.get("dest")
            src = instr["args"][0]
            if dest and src and dest != src:
                if dest not in phi_dests:
                    copy_map[dest] = src

    if not copy_map:
        return

    def ultimate_src(v: str) -> str:
        visited = set()
        while v in copy_map and v not in visited:
            visited.add(v)
            v = copy_map[v]
        return v

    new_instrs = []
    for instr in instrs:
        op = instr.get("op")

        if "label" in instr:
            new_instrs.append(instr)
            continue

        if "args" in instr:
            instr["args"] = [ultimate_src(a) for a in instr["args"]]

        if op == "id" and "args" in instr and instr["args"]:
            dest = instr.get("dest")
            src = instr["args"][0]
            if dest == src:
                continue
            if dest in copy_map:
                continue

        new_instrs.append(instr)

    func["instrs"] = new_instrs


def dom_tree_copy_elimination(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    phi_dests: Set[str] = set()
    for instr in instrs:
        if instr.get("op") == "phi":
            dest = instr.get("dest")
            if dest:
                phi_dests.add(dest)

    blocks = cfg.form_blocks(instrs)
    if not blocks:
        return

    next_map = cfg.form_cfg(blocks)
    dom_sets = df.get_dom_sets(blocks, next_map)
    _, dom_tree = df.get_immediate_doms_and_tree(dom_sets, next_map)

    bmap = {b["name"]: b for b in blocks}
    entry = blocks[0]["name"]

    def canon(x: str, rename: Dict[str, str]) -> str:
        visited = set()
        while x in rename and rename[x] != x and x not in visited:
            visited.add(x)
            x = rename[x]
        return x

    def walk(block_name: str, rename: Dict[str, str]):
        block = bmap[block_name]
        new_instrs = []

        for instr in block.get("instrs", []):
            op = instr.get("op")

            if "label" in instr:
                new_instrs.append(instr)
                continue

            if "args" in instr and isinstance(instr["args"], list) and op != "phi":
                instr["args"] = [canon(a, rename) for a in instr["args"]]

            if op == "id":
                args = instr.get("args", [])
                dest = instr.get("dest")
                if args and dest:
                    if dest in phi_dests:
                        new_instrs.append(instr)
                        continue
                    src = canon(args[0], rename)
                    rename[dest] = src
                    continue

            dest = instr.get("dest")
            if dest is not None:
                rename[dest] = dest
            new_instrs.append(instr)

        block["instrs"] = new_instrs

        for child in dom_tree.get(block_name, []):
            walk(child, rename.copy())

    walk(entry, {})

    func["instrs"] = flatten_blocks(blocks)


def compute_instruction_liveness(func: dict) -> Tuple[Dict[int, Set[str]], Dict[int, Set[str]]]:
    instrs = func.get("instrs", [])
    n = len(instrs)

    live_in: Dict[int, Set[str]] = {i: set() for i in range(n)}
    live_out: Dict[int, Set[str]] = {i: set() for i in range(n)}

    successor: Dict[int, List[int]] = {i: [] for i in range(n)}
    label_to_idx: Dict[str, int] = {}

    for i, instr in enumerate(instrs):
        if "label" in instr:
            label_to_idx[instr["label"]] = i

    for i, instr in enumerate(instrs):
        op = instr.get("op")
        if op == "jmp":
            target = instr.get("labels", [None])[0]
            if target and target in label_to_idx:
                successor[i].append(label_to_idx[target])
        elif op == "br":
            labels = instr.get("labels", [])
            for lbl in labels:
                if lbl in label_to_idx:
                    successor[i].append(label_to_idx[lbl])
        elif op == "ret":
            pass
        elif i + 1 < n:
            successor[i].append(i + 1)

    use_i: Dict[int, Set[str]] = {}
    def_i: Dict[int, Set[str]] = {}
    for i, instr in enumerate(instrs):
        uses = set()
        defs = set()
        for arg in instr.get("args", []):
            if isinstance(arg, str):
                uses.add(arg)
        if "dest" in instr:
            defs.add(instr["dest"])
        use_i[i] = uses
        def_i[i] = defs

    changed = True
    iterations = 0
    max_iterations = n * 10
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for i in range(n - 1, -1, -1):
            new_out: Set[str] = set()
            for s in successor[i]:
                new_out |= live_in[s]

            new_in = use_i[i] | (new_out - def_i[i])

            if new_in != live_in[i] or new_out != live_out[i]:
                changed = True
                live_in[i] = new_in
                live_out[i] = new_out

    return live_in, live_out


def trivial_copy_coalescing(func: dict):
    instrs = func.get("instrs", [])
    if len(instrs) < 2:
        return

    use_count: Dict[str, int] = {}
    for instr in instrs:
        for arg in instr.get("args", []):
            if isinstance(arg, str):
                use_count[arg] = use_count.get(arg, 0) + 1

    changed = True
    while changed:
        changed = False
        instrs = func.get("instrs", [])
        new_instrs = []
        skip_next = False

        for i, instr in enumerate(instrs):
            if skip_next:
                skip_next = False
                continue

            if i + 1 < len(instrs):
                next_instr = instrs[i + 1]
                if (next_instr.get("op") == "id" and
                    "dest" in instr and
                    next_instr.get("args", [None])[0] == instr["dest"]):

                    src = instr["dest"]
                    dest = next_instr["dest"]

                    src_uses = sum(1 for ins in instrs
                                   for arg in ins.get("args", [])
                                   if arg == src)

                    if src_uses == 1:
                        instr["dest"] = dest
                        new_instrs.append(instr)
                        skip_next = True
                        changed = True
                        continue

            new_instrs.append(instr)

        func["instrs"] = new_instrs


def full_copy_coalescing(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    param_names = {arg["name"] for arg in func.get("args", [])}

    live_in, live_out = compute_instruction_liveness(func)

    copies: List[Tuple[int, str, str]] = []
    for i, instr in enumerate(instrs):
        if instr.get("op") == "id" and "dest" in instr:
            args = instr.get("args", [])
            if args and isinstance(args[0], str):
                copies.append((i, instr["dest"], args[0]))

    def interferes(v1: str, v2: str) -> bool:
        for i in range(len(instrs)):
            if v1 in live_in[i] and v2 in live_in[i]:
                return True
            if v1 in live_out[i] and v2 in live_out[i]:
                return True
            if instrs[i].get("dest") == v1 and v2 in live_out[i]:
                return True
            if instrs[i].get("dest") == v2 and v1 in live_out[i]:
                return True
        return False

    rename_map: Dict[str, str] = {}

    for idx, dest, src in copies:
        if src in rename_map or dest in rename_map:
            continue
        if src in param_names:
            continue
        if not interferes(dest, src):
            rename_map[dest] = src

    if not rename_map:
        return

    def apply_rename(v: str) -> str:
        visited = set()
        while v in rename_map and v not in visited:
            visited.add(v)
            v = rename_map[v]
        return v

    new_instrs = []
    for instr in instrs:
        if "dest" in instr:
            instr["dest"] = apply_rename(instr["dest"])
        if "args" in instr:
            instr["args"] = [apply_rename(a) if isinstance(a, str) else a
                           for a in instr["args"]]
        if instr.get("op") == "id":
            args = instr.get("args", [])
            if args and args[0] == instr.get("dest"):
                continue

        new_instrs.append(instr)

    func["instrs"] = new_instrs


def aggressive_copy_coalescing(func: dict):
    prev_count = -1
    curr_count = len([i for i in func.get("instrs", []) if i.get("op") == "id"])

    max_rounds = 10
    rounds = 0
    while curr_count != prev_count and rounds < max_rounds:
        prev_count = curr_count
        rounds += 1

        trivial_copy_coalescing(func)
        full_copy_coalescing(func)

        curr_count = len([i for i in func.get("instrs", []) if i.get("op") == "id"])


def parameter_renaming(func: dict):
    instrs = func.get("instrs", [])
    args = func.get("args", [])
    if not args:
        return

    param_names = {arg["name"] for arg in args}

    param_use_count: Dict[str, int] = {name: 0 for name in param_names}
    for instr in instrs:
        for arg in instr.get("args", []):
            if arg in param_names:
                param_use_count[arg] += 1

    first_label_idx = len(instrs)
    for i, instr in enumerate(instrs):
        if "label" in instr:
            first_label_idx = i
            break

    param_renames: Dict[str, str] = {}
    copies_to_remove: Set[int] = set()

    for i, instr in enumerate(instrs):
        if i >= first_label_idx:
            break
        if instr.get("op") == "id" and "dest" in instr:
            instr_args = instr.get("args", [])
            if instr_args and instr_args[0] in param_names:
                src = instr_args[0]
                dest = instr["dest"]
                if (param_use_count.get(src, 0) == 1 and
                    src not in param_renames):
                    param_renames[src] = dest
                    copies_to_remove.add(i)

    if not param_renames:
        return

    for arg in args:
        if arg["name"] in param_renames:
            arg["name"] = param_renames[arg["name"]]
    for instr in instrs:
        if "args" in instr:
            instr["args"] = [param_renames.get(a, a) if isinstance(a, str) else a
                           for a in instr["args"]]
    func["instrs"] = [instr for i, instr in enumerate(instrs) if i not in copies_to_remove]


def const_copy_fusion(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    use_count: Dict[str, int] = {}
    for instr in instrs:
        for arg in instr.get("args", []):
            if isinstance(arg, str):
                use_count[arg] = use_count.get(arg, 0) + 1

    const_defs: Dict[str, Tuple[int, Any, str]] = {}
    for i, instr in enumerate(instrs):
        if instr.get("op") == "const" and "dest" in instr:
            const_defs[instr["dest"]] = (i, instr.get("value"), instr.get("type"))

    fusions: List[Tuple[int, int, str, Any, str]] = []
    for i, instr in enumerate(instrs):
        if instr.get("op") == "id" and "dest" in instr:
            instr_args = instr.get("args", [])
            if instr_args and instr_args[0] in const_defs:
                src = instr_args[0]
                if use_count.get(src, 0) == 1:
                    const_idx, value, ty = const_defs[src]
                    fusions.append((const_idx, i, instr["dest"], value, ty))

    if not fusions:
        return

    remove_indices = set()
    for const_idx, copy_idx, new_dest, value, ty in fusions:
        instrs[const_idx]["dest"] = new_dest
        remove_indices.add(copy_idx)

    func["instrs"] = [instr for i, instr in enumerate(instrs) if i not in remove_indices]


def phi_destination_propagation(func: dict):
    instrs = func.get("instrs", [])
    if not instrs:
        return

    param_names = {arg["name"] for arg in func.get("args", [])}

    changed = True
    max_iterations = 10
    iteration = 0

    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        instrs = func.get("instrs", [])

        use_count: Dict[str, int] = {}
        for instr in instrs:
            for arg in instr.get("args", []):
                if isinstance(arg, str):
                    use_count[arg] = use_count.get(arg, 0) + 1

        def_count: Dict[str, int] = {}
        for instr in instrs:
            if "dest" in instr:
                dest = instr["dest"]
                def_count[dest] = def_count.get(dest, 0) + 1

        def_idx: Dict[str, int] = {}
        for i, instr in enumerate(instrs):
            if "dest" in instr and instr["dest"] not in def_idx:
                def_idx[instr["dest"]] = i

        propagations: List[Tuple[int, int, str, str]] = []

        for i, instr in enumerate(instrs):
            if instr.get("op") == "id" and "dest" in instr:
                instr_args = instr.get("args", [])
                if not instr_args:
                    continue
                src = instr_args[0]
                dest = instr["dest"]
                if src in param_names:
                    continue
                if use_count.get(src, 0) != 1:
                    continue
                if def_count.get(dest, 0) != 1:
                    continue
                if src not in def_idx:
                    continue

                src_def_i = def_idx[src]
                src_def = instrs[src_def_i]
                if src_def.get("op") in ("phi", "call"):
                    continue

                propagations.append((src_def_i, i, src, dest))

        if not propagations:
            break

        def_i, copy_i, old_name, new_name = propagations[0]
        instrs[def_i]["dest"] = new_name
        new_instrs = [instr for j, instr in enumerate(instrs) if j != copy_i]
        func["instrs"] = new_instrs
        changed = True


def run(func: dict, convert_ssa: bool = True):
    if convert_ssa:
        to_ssa.run(func)

    gvn_on_function(func)
    ssa_copy_propagation(func)
    dom_tree_copy_elimination(func)
    if convert_ssa:
        from_ssa.lower_phis(func)

    aggressive_copy_coalescing(func)
    parameter_renaming(func)
    const_copy_fusion(func)
    phi_destination_propagation(func)
    aggressive_copy_coalescing(func)
    tdce.trivial_dce_func(func)


def main(prog: dict, use_ssa: bool = True) -> dict:
    for func in prog.get("functions", []):
        run(func, convert_ssa=use_ssa)
    return prog


if __name__ == "__main__":
    prog = json.load(sys.stdin)
    result = main(prog, use_ssa=True)
    json.dump(result, sys.stdout, indent=None, separators=(",", ":"))
