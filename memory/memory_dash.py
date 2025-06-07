"""Minimal CLI to inspect memory state."""

from __future__ import annotations

import argparse
from pprint import pprint

from .memory_core import MemoryCore


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect XAlgo memory stores")
    parser.add_argument("--save", action="store_true", help="Force save on exit")
    args = parser.parse_args()

    mem = MemoryCore()
    print("\nGhost Heatmap:")
    pprint(mem.ghost_heatmap)

    print("\nSignal Memory (last 5):")
    for item in list(mem.signal_memory)[-5:]:
        pprint(item)

    print("\nRegime Profile:")
    pprint(mem.regime_profile)

    if args.save:
        mem.save_all()
        print("\nMemory saved.")


if __name__ == "__main__":
    main()
