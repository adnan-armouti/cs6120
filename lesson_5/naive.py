def dominates_slow(entry, dom_block, target_block, next_map):
    if dom_block == entry: return True
    if dom_block == target_block: return True  # Self-dominance
    # First check if target is reachable
    target_reachable = False
    seen, stack = set(), [entry]
    while stack:
        curr_block = stack.pop()
        if curr_block == target_block: 
            target_reachable = True
            break
        if curr_block in seen: continue
        seen.add(curr_block)
        for next_block in next_map.get(curr_block, []):
            if next_block not in seen:
                stack.append(next_block)
    # If target is unreachable, dominance is undefined - return False
    if not target_reachable:
        return False
    # Now check if dom_block dominates target_block
    seen, stack = set(), [entry]
    while stack:
        curr_block = stack.pop()
        if curr_block == dom_block: continue
        if curr_block == target_block: return False
        if curr_block in seen: continue
        seen.add(curr_block)
        for next_block in next_map.get(curr_block, []):
            if next_block not in seen:
                stack.append(next_block)
    return True

def in_frontier_slow(dom_block, node_block, entry, next_map):
    prev_map = {}
    for prev_block, next_blocks in next_map.items():
        for next_block in next_blocks:
            prev_map.setdefault(next_block, []).append(prev_block)
    has_dom_pred = any(dominates_slow(entry, dom_block, p, next_map) for p in prev_map.get(node_block, []))
    if not has_dom_pred: return False
    if dom_block == node_block: return True
    return not dominates_slow(entry, dom_block, node_block, next_map)