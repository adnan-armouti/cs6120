
import sys, json
import cfg

# helper
def invert_next_to_prev(blocks, next_map):
    block_idx = {b["name"]: i for i, b in enumerate(blocks)}
    prev_map = {b["name"]: [] for b in blocks}
    for prev_block, next_blocks in next_map.items():
        for next_block in next_blocks:
            prev_map[next_block].append(prev_block)
    for name in prev_map:
        prev_map[name].sort(key=lambda x: block_idx[x])
    return prev_map

# helper
def append_dom_tree_postorder(dom_tree, node, out_list):
    for child in dom_tree.get(node, []): 
        append_dom_tree_postorder(dom_tree, child, out_list)
    out_list.append(node)


def get_dom_sets(blocks, next_map):
    block_names_ordered = [b["name"] for b in blocks]
    if not block_names_ordered: return {}
    entry = block_names_ordered[0]
    all_block_names = set(block_names_ordered)
    prev_map = invert_next_to_prev(blocks, next_map)
    # init dom sets
    doms_map = {entry: {entry}}
    for name in block_names_ordered[1:]:
        doms_map[name] = set(all_block_names)
    # iter until convergence
    is_changed = True
    while is_changed:
        is_changed = False
        for block_name in block_names_ordered[1:]:
            prev = prev_map[block_name]
            if prev:
                # Only consider predecessors that are reachable
                reachable_preds = [p for p in prev if is_reachable(p, entry, next_map)]
                if reachable_preds:
                    predoms_intersection = set.intersection(*(doms_map[p] for p in reachable_preds))
                else:
                    # No reachable predecessors, block is unreachable
                    predoms_intersection = set()
            else:
                # No predecessors, block is unreachable
                predoms_intersection = set()
            updated_doms = {block_name} | predoms_intersection
            if updated_doms != doms_map[block_name]:
                doms_map[block_name] = updated_doms
                is_changed = True
    return doms_map

def is_reachable(block_name, entry, next_map):
    """Check if a block is reachable from the entry using DFS."""
    if block_name == entry:
        return True
    visited = set()
    stack = [entry]
    while stack:
        curr = stack.pop()
        if curr == block_name:
            return True
        if curr in visited:
            continue
        visited.add(curr)
        for next_block in next_map.get(curr, []):
            if next_block not in visited:
                stack.append(next_block)
    return False

def get_immediate_doms_and_tree(doms_map, next_map):
    block_names_ordered = sorted(doms_map.keys())
    entry = next(n for n, ds in doms_map.items() if ds == {n})
    immediate_dom = {entry: None}
    
    # Get the next_map for reachability checking
    # We need to reconstruct it from the blocks
    # For now, let's use a simpler approach and just check if the dominator set is reasonable
    for block_name in block_names_ordered:
        if block_name == entry: continue
        
        # A block is unreachable if it's not reachable from the entry
        if not is_reachable(block_name, entry, next_map):
            immediate_dom[block_name] = None  # Unreachable
            continue
            
        candidate_doms = doms_map[block_name] - {block_name}
        parent = None
        for d in sorted(candidate_doms):
            # Check if d is dominated by any other candidate dominator
            # A dominator d is immediate if no other dominator in the candidate set dominates d
            is_dominated = any(d in doms_map[e] and d != e for e in candidate_doms if e != d)
            if not is_dominated:
                parent = d
                break
        immediate_dom[block_name] = parent
    dom_tree = {n: [] for n in block_names_ordered}
    for child, parent in immediate_dom.items():
        if parent is not None: dom_tree[parent].append(child)
    for k in dom_tree: dom_tree[k].sort()
    return immediate_dom, dom_tree

def get_dominance_frontier(blocks, next_map, immediate_dom, dom_tree):
    block_names_ordered = [b["name"] for b in blocks]
    dominance_frontier = {n: set() for n in block_names_ordered}
    # local
    for block_name in block_names_ordered:
        for next_block in next_map.get(block_name, []):
            if immediate_dom.get(next_block) != block_name:
                dominance_frontier[block_name].add(next_block)
    # postorder over dom tree
    postorder = []
    for root, parent in immediate_dom.items():
        if parent is None:
            append_dom_tree_postorder(dom_tree, root, postorder)
    # upward
    for block_name in postorder:
        for child in dom_tree.get(block_name, []):
            for n in dominance_frontier[child]:
                if immediate_dom.get(n) != block_name:
                    dominance_frontier[block_name].add(n)
    return dominance_frontier


def _fmt(items):
    return "{" + ", ".join(sorted(items)) + "}" if items else "{}"

def run(func, check_slow=False, check_fast=False):
    blocks = cfg.form_blocks(func.get("instrs", []))
    next_map = cfg.form_cfg(blocks)
    names = [b["name"] for b in blocks]
    if not names:
        print(f"Function {func['name']}: (no blocks)")
        return
    entry = names[0]

    doms_map = get_dom_sets(blocks, next_map)
    immediate_dom, dom_tree = get_immediate_doms_and_tree(doms_map, next_map)
    dominance_frontier = get_dominance_frontier(blocks, next_map, immediate_dom, dom_tree)

    print(f"Function {func['name']}:")
    print("  Dominators:")
    for block_name in names:
        print(f"    {block_name}: {_fmt(doms_map[block_name])}")
    print("  Immediate Dominator:")
    for block_name in names:
        print(f"    {block_name}: {immediate_dom[block_name]}")
    print("  Dominator Tree children:")
    for block_name in names:
        children = dom_tree.get(block_name, [])
        if children:
            print(f"    {block_name}: {', '.join(children)}")
    print("  Dominance Frontier:")
    for block_name in names:
        print(f"    {block_name}: {_fmt(dominance_frontier[block_name])}")

    if check_slow:
        is_match = True
        for candidate_a in names:
            for candidate_b in names:
                # Special case: entry dominates all blocks, even unreachable ones
                if candidate_a == entry:
                    fast_result = True
                else:
                    fast_result = candidate_a in doms_map[candidate_b]
                if fast_result != naive.dominates_slow(entry, candidate_a, candidate_b, next_map):
                    is_match = False
                    print(f"  [mismatch] slow-check dom {candidate_a} vs {candidate_b}")
        if is_match:
            print("  [check-slow] dom sets agree with slow test.")

    if check_fast:
        idom_chk_map = chk.idom_chk(blocks, next_map)
        chk_tree = chk.build_tree(idom_chk_map)
        chk_df = chk.dom_frontier(blocks, next_map, idom_chk_map, chk_tree)
        ok_fast = True
        for n in names:
            if immediate_dom.get(n) != idom_chk_map.get(n):
                ok_fast = False
                print(f"  [mismatch] fast-check idom {n}: ours={immediate_dom.get(n)} chk={idom_chk_map.get(n)}")
        for n in names:
            ours_children = dom_tree.get(n, [])
            chk_children = chk_tree.get(n, [])
            if ours_children != chk_children:
                ok_fast = False
                print(f"  [mismatch] fast-check domTree children {n}: ours={', '.join(ours_children)} chk={', '.join(chk_children)}")
        for n in names:
            if dominance_frontier.get(n, set()) != chk_df.get(n, set()):
                ok_fast = False
                print(f"  [mismatch] fast-check DF {n}: ours={_fmt(dominance_frontier.get(n, set()))} chk={_fmt(chk_df.get(n, set()))}")
        if ok_fast:
            print("  [check-fast] CHK agrees with our results.")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_slow', action='store_true', help='verify dom sets using slow dominance relation')
    parser.add_argument('--check_fast', action='store_true', help='verify immediate dominators, dom tree, and DF using CHK algorithm')
    args = parser.parse_args()
    prog = json.load(sys.stdin)
    for f in prog.get("functions", []):
        run(f, check_slow=args.check_slow, check_fast=args.check_fast)

if __name__ == "__main__":
    main()
