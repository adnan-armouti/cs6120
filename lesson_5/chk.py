# ---------- utilities (next/prev, RPO) ----------

def invert_next_to_prev(blocks, next_map):
    idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for prev_block, next_blocks in next_map.items():
        for next_block in next_blocks:
            prev_map[next_block].append(prev_block)
    for k in prev_map:
        prev_map[k].sort(key=lambda n: idx[n])
    return prev_map

def rpo_order(blocks, next_map):
    """Reverse postorder numbering over the CFG."""
    names = [b["name"] for b in blocks]
    if not names: return {}, []
    entry = names[0]
    seen, post = set(), []
    def dfs(u):
        if u in seen: return
        seen.add(u)
        for v in next_map.get(u, []):
            dfs(v)
        post.append(u)
    dfs(entry)
    order = list(reversed(post))
    num = {n: i for i, n in enumerate(order)}
    return num, order

# ---------- CHK: idom via reverse postorder + intersect ----------

def intersect(u, v, idom, rpo_num):
    while u != v:
        while rpo_num[u] > rpo_num[v]:
            u = idom[u]
        while rpo_num[v] > rpo_num[u]:
            v = idom[v]
    return u

def idom_chk(blocks, next_map):
    names = [b["name"] for b in blocks]
    if not names: return {}
    entry = names[0]
    prv = invert_next_to_prev(blocks, next_map)
    rpo_num, _ = rpo_order(blocks, next_map)

    # seed
    idom = {n: None for n in names}
    idom[entry] = entry  # CHK uses self-idom during iteration

    changed = True
    while changed:
        changed = False
        # process in RPO, skip entry
        for n in sorted(names, key=lambda x: rpo_num.get(x, 10**9)):
            if n == entry: continue
            # pick a processed predecessor to start
            preds = [p for p in prv.get(n, []) if idom[p] is not None]
            if not preds:
                continue  # unreachable
            new_idom = preds[0]
            for p in preds[1:]:
                new_idom = intersect(new_idom, p, idom, rpo_num)
            if idom[n] != new_idom:
                idom[n] = new_idom
                changed = True

    # normalize: entry has no idom (None)
    idom[entry] = None
    return idom

# ---------- dom tree + dominance frontier ----------

def build_tree(idom):
    tree = {n: [] for n in idom}
    for b, d in idom.items():
        if d is not None:
            tree[d].append(b)
    for k in tree: tree[k].sort()
    return tree

def dom_frontier(blocks, next_map, idom, tree):
    names = [b["name"] for b in blocks]
    DF = {n: set() for n in names}

    # local
    for b in names:
        for s in next_map.get(b, []):
            if idom.get(s) != b:
                DF[b].add(s)

    # postorder over dom tree
    post = []
    def walk(u):
        for c in tree.get(u, []): walk(c)
        post.append(u)
    for r, d in idom.items():
        if d is None:
            walk(r)

    # upward
    for b in post:
        for c in tree.get(b, []):
            for n in DF[c]:
                if idom.get(n) != b:
                    DF[b].add(n)

    return DF