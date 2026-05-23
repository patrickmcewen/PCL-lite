#!/usr/bin/env python3
"""Token usage roll-up for PCL-lite run directories.

Walks a run dir produced by ``run_stepdb.sh`` (e.g.
``/workspace/pcl_run/benchcard_stepdb_transformer_subparts``) and aggregates
every ``usage.json`` along the directory hierarchy:

    <run_root>/{single,iter}/<category>/<task>/<timestamp>/<model>/usage.json

Each ``usage.json`` is a JSON list of per-API-call dicts with
``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``; the script sums
those across the list and rolls the result up through every parent directory.

Usage:
    python scripts/token_summary.py <run_dir> [--depth N] [--json]
    python scripts/token_summary.py <run_dir> --mode iter --depth 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _zero() -> dict[str, int]:
    return {"calls": 0, "files": 0, "prompt": 0, "completion": 0, "total": 0}


def _add(into: dict[str, int], other: dict[str, int]) -> None:
    for k in ("calls", "files", "prompt", "completion", "total"):
        into[k] += other.get(k, 0)


def _read_usage(path: Path) -> dict[str, int]:
    """Sum the per-call entries in a single ``usage.json`` into one record."""
    raw = json.loads(path.read_text())
    assert isinstance(raw, list), (
        f"{path}: expected a JSON list of per-call usage dicts, got "
        f"{type(raw).__name__}"
    )
    rec = _zero()
    rec["files"] = 1
    for entry in raw:
        rec["calls"] += 1
        rec["prompt"] += int(entry.get("prompt_tokens") or 0)
        rec["completion"] += int(entry.get("completion_tokens") or 0)
        rec["total"] += int(entry.get("total_tokens") or 0)
    return rec


def _summarize_dir(d: Path) -> dict[str, Any] | None:
    """Recursively summarize a directory.

    Returns a node dict ``{"total": {...}, "<child>": {...}, ...}`` or ``None``
    when the subtree contains no ``usage.json`` records (so the caller can
    prune empty intermediate dirs from the output).
    """
    total = _zero()
    children: dict[str, Any] = {}

    own = d / "usage.json"
    if own.is_file():
        rec = _read_usage(own)
        _add(total, rec)

    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        child = _summarize_dir(sub)
        if child is None:
            continue
        children[sub.name] = child
        _add(total, child["total"])

    if total["files"] == 0 and not children:
        return None

    return {"total": total, **children}


def summarize(run_root: Path) -> dict[str, Any]:
    """Public entry point — always returns a dict with at least a ``total``."""
    summary = _summarize_dir(run_root)
    if summary is None:
        return {"total": _zero()}
    return summary


def _format_int(n: int) -> str:
    return f"{n:,}"


def _print_tree(node: dict, name: str, depth: int, max_depth: int | None) -> None:
    indent = "  " * depth
    t = node["total"]
    print(
        f"{indent}{name}: {_format_int(t['total'])} "
        f"(prompt={_format_int(t['prompt'])}, "
        f"completion={_format_int(t['completion'])}, "
        f"calls={t['calls']}, files={t['files']})"
    )
    if max_depth is not None and depth >= max_depth:
        return
    for child_name, child in node.items():
        if child_name == "total":
            continue
        _print_tree(child, child_name, depth + 1, max_depth)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", help="PCL-lite run directory (or any subdir)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Max nesting depth to print (default: unlimited)")
    parser.add_argument("--mode", choices=("single", "iter"), default=None,
                        help="Restrict to a single mode subtree (default: both)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of an indented tree")
    args = parser.parse_args()

    root = Path(args.run_dir)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    target = root / args.mode if args.mode else root
    if args.mode and not target.is_dir():
        print(f"error: --mode {args.mode!r} but {target} does not exist",
              file=sys.stderr)
        sys.exit(2)

    summary = summarize(target)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_tree(summary, target.name or str(target), 0, args.depth)


if __name__ == "__main__":
    main()
