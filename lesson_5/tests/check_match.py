import os
import sys


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def list_bril_files(dir_path: str):
    return sorted([f for f in os.listdir(dir_path) if f.endswith(".bril")])


def evaluate_small(dir_path: str):
    print("small:")
    for bril_name in list_bril_files(dir_path):
        stem = os.path.splitext(bril_name)[0]
        out_path = os.path.join(dir_path, f"{stem}.df_slow.out")
        content = read_text(out_path)
        if not content:
            print(f"  {bril_name}: missing output")
            continue
        slow_ok = ("[check-slow]" in content) and ("[mismatch]" not in content)
        if slow_ok:
            print(f"  {bril_name}: matches naive!")
        else:
            print(f"  {bril_name}: DOES NOT match naive!")
    print()


def evaluate_medium(dir_path: str):
    print("medium:")
    for bril_name in list_bril_files(dir_path):
        stem = os.path.splitext(bril_name)[0]
        out_path = os.path.join(dir_path, f"{stem}.df_both.out")
        content = read_text(out_path)
        if not content:
            print(f"  {bril_name}: missing output")
            continue
        slow_ok = ("[check-slow]" in content)
        fast_ok = ("[check-fast]" in content)
        has_mismatch = ("[mismatch]" in content)
        if slow_ok and fast_ok and not has_mismatch:
            print(f"  {bril_name}: matches naive and fast!")
        elif slow_ok and not has_mismatch:
            print(f"  {bril_name}: matches naive only")
        elif fast_ok and not has_mismatch:
            print(f"  {bril_name}: matches fast only")
        else:
            print(f"  {bril_name}: DOES NOT match naive or fast")
    print()


def evaluate_large(dir_path: str):
    print("large:")
    for bril_name in list_bril_files(dir_path):
        stem = os.path.splitext(bril_name)[0]
        out_path = os.path.join(dir_path, f"{stem}.df_fast.out")
        content = read_text(out_path)
        if not content:
            print(f"  {bril_name}: missing output")
            continue
        fast_ok = ("[check-fast]" in content) and ("[mismatch]" not in content)
        if fast_ok:
            print(f"  {bril_name}: matches fast!")
        else:
            print(f"  {bril_name}: DOES NOT match fast!")
    print()


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    evaluate_small(os.path.join(base_dir, "small"))
    evaluate_medium(os.path.join(base_dir, "medium"))
    evaluate_large(os.path.join(base_dir, "large"))


if __name__ == "__main__":
    main()


