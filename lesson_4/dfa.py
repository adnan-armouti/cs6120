from __future__ import annotations
from typing import Dict, List, Set

class DataFlowAnalysis:
    def __init__(self) -> None:
        self.direction: str = "forward"

    def prepare(self, func: dict, blocks: List[dict]) -> None:
        return None

    def merge(self, values: List[Set[str]]) -> Set[str]:
        raise NotImplementedError

    def transfer(self, block_name: str, in_or_out: Set[str]) -> Set[str]:
        raise NotImplementedError

    def initial_in(self, block_name: str) -> Set[str]:
        return set()

    def initial_out(self, block_name: str) -> Set[str]:
        return set()

    @staticmethod
    def _prev_map(blocks: List[dict], next_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
        order = {b["name"]: i for i, b in enumerate(blocks)}
        prev: Dict[str, List[str]] = {b["name"]: [] for b in blocks}
        for src, dsts in next_map.items():
            for d in dsts:
                prev[d].append(src)
        for k in prev:
            prev[k].sort(key=lambda n: order[n])
        return prev

    def solve(self, func: dict, blocks: List[dict], next_map: Dict[str, List[str]]):
        if not blocks:
            return {}, {}
        self.prepare(func, blocks)
        names = [b["name"] for b in blocks]
        entry = names[0]
        prev = self._prev_map(blocks, next_map)

        in_dict  = {n: set(self.initial_in(n))  for n in names}
        out_dict = {n: set(self.initial_out(n)) for n in names}
        order = {n: i for i, n in enumerate(names)}
        work = set(names)

        if self.direction == "forward":
            while work:
                # pick earliest by program order (no nested helper)
                b = min(work, key=lambda k: order[k])
                work.remove(b)
                if b != entry:
                    in_dict[b] = self.merge([out_dict[p] for p in prev[b]])
                new_out = self.transfer(b, in_dict[b])
                if new_out != out_dict[b]:
                    out_dict[b] = new_out
                    for n in next_map.get(b, []):
                        work.add(n)
        else:
            while work:
                b = min(work, key=lambda k: order[k])
                work.remove(b)
                next = next_map.get(b, [])
                if next:
                    out_dict[b] = self.merge([in_dict[s] for s in next])
                new_in = self.transfer(b, out_dict[b])
                if new_in != in_dict[b]:
                    in_dict[b] = new_in
                    for p in prev[b]:
                        work.add(p)

        return in_dict, out_dict