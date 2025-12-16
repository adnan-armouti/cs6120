from __future__ import annotations

import os
import sys
import json
import math
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

import gvn
import to_ssa


def bril_txt_to_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        json_str = subprocess.check_output(["bril2json"], stdin=f, text=True)
    return json.loads(json_str)


def bril_json_to_txt(prog: dict) -> str:
    return subprocess.check_output(
        ["bril2txt"],
        input=json.dumps(prog),
        text=True
    )


def run_bril(bril_path: str, args: List[str] = None) -> Tuple[str, int, Any]:
    if args is None:
        args = []

    with open(bril_path, "r", encoding="utf-8") as f:
        try:
            bril_json_str = subprocess.check_output(["bril2json"], stdin=f, text=True)
        except subprocess.CalledProcessError:
            return "N/A", 0, "N/A"

    program = json.loads(bril_json_str)
    static_count = sum(
        len(func.get("instrs", []))
        for func in program.get("functions", [])
    )

    try:
        result = subprocess.run(
            ["brili", "-p", *args],
            input=bril_json_str,
            capture_output=True,
            text=True,
            check=True,
            timeout=20
        )
    except subprocess.CalledProcessError:
        return "N/A", static_count, "N/A"
    except subprocess.TimeoutExpired:
        return "T/O", static_count, "T/O"

    if result.stderr.startswith("total_dyn_inst: "):
        dyn_count = int(result.stderr.split()[1])
    else:
        dyn_count = "N/A"

    return result.stdout, static_count, dyn_count


def run_bril_json(prog: dict, args: List[str] = None) -> Tuple[str, int, Any]:
    if args is None:
        args = []

    bril_json_str = json.dumps(prog)
    static_count = sum(
        len(func.get("instrs", []))
        for func in prog.get("functions", [])
    )

    try:
        result = subprocess.run(
            ["brili", "-p", *args],
            input=bril_json_str,
            capture_output=True,
            text=True,
            check=True,
            timeout=20
        )
    except subprocess.CalledProcessError:
        return "N/A", static_count, "N/A"
    except subprocess.TimeoutExpired:
        return "T/O", static_count, "T/O"

    if result.stderr.startswith("total_dyn_inst: "):
        dyn_count = int(result.stderr.split()[1])
    else:
        dyn_count = "N/A"

    return result.stdout, static_count, dyn_count


def extract_args(bril_file: Path) -> List[str]:
    try:
        with open(bril_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.replace(" ", "").startswith("#ARGS:"):
                    return line[line.find(":") + 1:].split()
    except (OSError, UnicodeDecodeError):
        pass
    return []


def geometric_mean(nums: List[float]) -> float:
    if not nums:
        return float("nan")
    return math.exp(sum(math.log(x) for x in nums) / len(nums))


def gvn_wrapper(before_path: Path, after_path: Path):
    program = bril_txt_to_json(str(before_path))
    optimized = gvn.main(program, use_ssa=True)
    bril_text = bril_json_to_txt(optimized)
    with open(after_path, "w", encoding="utf-8") as f:
        f.write(bril_text)


def ssa_wrapper(before_path: Path, after_path: Path):
    program = bril_txt_to_json(str(before_path))
    for func in program.get("functions", []):
        to_ssa.run(func)
    bril_text = bril_json_to_txt(program)
    with open(after_path, "w", encoding="utf-8") as f:
        f.write(bril_text)


def collect_targets(input_paths: List[str]) -> List[Path]:
    targets: List[Path] = []
    for p in input_paths:
        path = Path(p)
        if path.is_file() and path.name.endswith(".bril"):
            targets.append(path)
        elif path.is_dir():
            for child in path.rglob("*.bril"):
                if child.is_file():
                    targets.append(child)
    targets.sort()
    return targets


def eval_results(results: List[Dict[str, Any]]):
    total = len(results)
    good = [r for r in results if r["verdict"] == "Good!"]

    print(f"\nSuccessful optimizations: {len(good)}/{total}")

    if good:
        valid = [r for r in good
                 if isinstance(r["static_before"], int) and r["static_before"] > 0
                 and isinstance(r["static_after"], int) and r["static_after"] > 0
                 and isinstance(r["dyn_before"], int) and r["dyn_before"] > 0
                 and isinstance(r["dyn_after"], int) and r["dyn_after"] > 0]

        if valid:
            static_ratios = [r["static_after"] / r["static_before"] for r in valid]
            dyn_ratios = [r["dyn_after"] / r["dyn_before"] for r in valid]

            print(f"Static Instr Ratio (GM): {geometric_mean(static_ratios):.4f}x")
            print(f"Dynamic Instr Ratio (GM): {geometric_mean(dyn_ratios):.4f}x")

            by_static = sorted(valid, key=lambda r: r["static_after"] / r["static_before"])
            by_dyn = sorted(valid, key=lambda r: r["dyn_after"] / r["dyn_before"])

            print("\nTop 5 static improvements:")
            for r in by_static[:5]:
                ratio = r["static_after"] / r["static_before"]
                print(f"  {Path(r['file']).name}: {ratio:.3f}x ({r['static_before']} -> {r['static_after']})")

            print("\nTop 5 dynamic improvements:")
            for r in by_dyn[:5]:
                ratio = r["dyn_after"] / r["dyn_before"]
                print(f"  {Path(r['file']).name}: {ratio:.3f}x ({r['dyn_before']} -> {r['dyn_after']})")
    else:
        print("No successful optimizations.")

    failures = [r for r in results if r["verdict"] != "Good!"]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures[:10]:
            print(f"  {Path(r['file']).name}: {r['verdict']}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")


def main(argv: List[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=["tests/core"])
    parser.add_argument("--out", type=str, default="results_gvn.json")
    parser.add_argument("--ssa", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    targets = collect_targets(args.paths)
    if not targets:
        print("No .bril files found.")
        return 1

    print(f"Found {len(targets)} .bril files")
    os.makedirs("./tmp", exist_ok=True)

    results: List[Dict[str, Any]] = []

    for t in tqdm(targets, disable=args.verbose):
        if args.verbose:
            print(f"Processing {t}...")

        prog_args = extract_args(t)

        out_before, static_before, dyn_before = run_bril(str(t), prog_args)

        ssa_data = {}
        if args.ssa:
            ssa_file = Path("./tmp") / (str(t).replace("/", "__") + ".ssa")
            try:
                ssa_wrapper(t, ssa_file)
                out_ssa, static_ssa, dyn_ssa = run_bril(str(ssa_file), prog_args)
                ssa_data = {
                    "output_ssa": out_ssa,
                    "static_ssa": static_ssa,
                    "dyn_ssa": dyn_ssa,
                }
            except Exception as e:
                ssa_data = {
                    "output_ssa": f"ERROR: {e}",
                    "static_ssa": "N/A",
                    "dyn_ssa": "N/A",
                }

        gvn_file = Path("./tmp") / (str(t).replace("/", "__") + ".gvn")
        try:
            gvn_wrapper(t, gvn_file)
            out_after, static_after, dyn_after = run_bril(str(gvn_file), prog_args)
        except Exception as e:
            out_after = f"ERROR: {e}"
            static_after = "N/A"
            dyn_after = "N/A"

        if out_before == "N/A":
            verdict = "BAD: original program fails"
        elif out_before == "T/O":
            verdict = "BAD: original program times out"
        elif isinstance(out_after, str) and out_after.startswith("ERROR"):
            verdict = f"BAD: GVN error - {out_after}"
        elif out_after == "N/A":
            verdict = "BAD: optimized program fails"
        elif out_after == "T/O":
            verdict = "BAD: optimized program times out"
        elif out_before != out_after:
            verdict = "BAD: output mismatch"
        else:
            verdict = "Good!"

        rec = {
            "file": str(t),
            "verdict": verdict,
            "output_before": out_before[:500] if isinstance(out_before, str) else out_before,
            "output_after": out_after[:500] if isinstance(out_after, str) else out_after,
            "static_before": static_before,
            "static_after": static_after,
            "dyn_before": dyn_before,
            "dyn_after": dyn_after,
        }
        rec.update(ssa_data)
        results.append(rec)

    eval_results(results)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote detailed results to {args.out}")

    good = sum(1 for r in results if r["verdict"] == "Good!")
    return 0 if good == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
