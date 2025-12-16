import json
import sys

TERMINATORS = ('jmp', 'br', 'ret')

def yield_block(state, blocks):
    if not state['instrs'] and state['label'] is None:
        return
    name = state['label'] if state['label'] is not None else f"B{state['next_id']}"
    if state['label'] is None:
        state['next_id'] += 1
    blocks.append({"name": name, "label": state['label'], "instrs": state['instrs']})
    state['label'] = None
    state['instrs'] = []

def form_blocks(instrs):
    blocks = []
    state = {'label': None, 'instrs': [], 'next_id': 0}

    if instrs and 'label' in instrs[0]:
        blocks.append({"name": "B0", "label": None, "instrs": []})
        state['next_id'] = 1

    for i in instrs:
        if 'label' in i:
            yield_block(state, blocks)
            state['label'] = i['label']
        else:
            state['instrs'].append(i)
            if i.get('op') in TERMINATORS:
                yield_block(state, blocks)
    yield_block(state, blocks)
    return blocks

def form_cfg(blocks):
    label_to_block = {b["label"]: b["name"] for b in blocks if b["label"] is not None}
    cfg = {b["name"]: [] for b in blocks}
    for idx, b in enumerate(blocks):
        if not b["instrs"]:
            if idx+1 < len(blocks):
                cfg[b["name"]].append(blocks[idx+1]["name"])
            continue
        last_i = b["instrs"][-1]
        op = last_i.get("op")
        if op == "jmp":
            target_label = last_i.get("labels", [None])[0]
            if target_label is not None:
                cfg[b["name"]].append(label_to_block.get(target_label, target_label))
        elif op == "br":
            for target_label in last_i.get("labels", []):
                if target_label is not None:
                    cfg[b["name"]].append(label_to_block.get(target_label, target_label))
        elif op != "ret" and idx+1 < len(blocks):
            cfg[b["name"]].append(blocks[idx+1]["name"])
    return cfg

def main():
    json_file = json.load(sys.stdin)
    for func in json_file.get("functions", []):
        print(f"Function {func['name']}:")
        blocks = form_blocks(func.get("instrs", []))
        cfg = form_cfg(blocks)
        print("Blocks:")
        for b in blocks:
            print(f"Block {b['name']}: {len(b['instrs'])} instructions")
        print("CFG:")
        for b in blocks:
            next_block = cfg[b["name"]]
            if next_block:
                print(f"{b['name']} -> {', '.join(next_block)}")
            else:
                print(f"{b['name']} -> (exit)")
        print("")

if __name__ == "__main__":
    main()
