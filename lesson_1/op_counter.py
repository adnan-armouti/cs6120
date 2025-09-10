import json
import sys

def count_ops(json_file):
    counter = {}
    for func in json_file.get('functions', []):
        for i in func.get('instrs', []):
            op = i.get('op')
            if op:
                counter[op] = counter.get(op, 0) + 1
    return counter

def main():
    json_file = json.load(sys.stdin)
    counter = count_ops(json_file)
    for op in sorted(counter):
        print(f"{op}: {counter[op]}")

if __name__ == "__main__":
    main()