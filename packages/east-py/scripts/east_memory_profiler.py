#!/usr/bin/env python3
#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Memory profiling script for East-py.

Profiles memory usage across the East IR pipeline (deserialization, compilation,
execution) using tracemalloc for allocation tracking and gc for object counting.

Usage:
    # Profile using test IR files
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py

    # Profile specific files
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py /tmp/east-profile-ir/*.beast2

    # Show top N allocation sites
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py --top 30

    # Compare memory between phases (snapshot diff)
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py --diff

    # Check for leaks by running multiple iterations
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py --leak-check --iterations 5

    # Save results to JSON
    uv run --package east-py python packages/east-py/scripts/east_memory_profiler.py -o memory-results.json
"""

import argparse
import asyncio
import gc
import json
import sys
import tracemalloc
from collections import Counter
from pathlib import Path
from typing import Any

# Add east-py to path if running directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from east.runtime.compiler import compile, compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.beast2 import (
    BEAST2_MAGIC_BYTES,
    decode_beast2_for,
    decode_beast2_with_header_for,
)
from east.serialization.json import decode_json_for
from east.types.type_of_type import IRType
from east.types.types import FunctionType, NullType, StringType
from east.types.values import (
    EastArray,
    EastDict,
    EastMatrix,
    EastNull,
    EastRef,
    EastSet,
    EastStruct,
    EastVariant,
    EastVector,
)

# Default test IR directories
TEST_IR_DIR = Path("/tmp/east-test-ir")
PROFILE_IR_DIR = Path("/tmp/east-profile-ir")

# East types to track in object counts
EAST_TYPES = (
    EastArray, EastDict, EastSet, EastStruct, EastVariant,
    EastNull, EastRef, EastVector, EastMatrix,
)


def format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if abs(n) < 1024:
        return f"{n} B"
    if abs(n) < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.1f} MiB"


def create_test_platforms() -> tuple[list[PlatformFunction], dict[str, Any]]:
    """Create minimal platform functions for test execution."""
    test_results: dict[str, Any] = {"passed": 0, "failed": 0, "errors": []}

    async def describe_impl(_name: str, test_fn: Any) -> None:
        if callable(test_fn):
            result = test_fn()
            if asyncio.iscoroutine(result):
                await result

    async def test_impl(name: str, test_fn: Any) -> None:
        try:
            if callable(test_fn):
                result = test_fn()
                if asyncio.iscoroutine(result):
                    await result
            test_results["passed"] += 1
        except Exception as e:
            test_results["failed"] += 1
            test_results["errors"].append((name, str(e)))

    def test_pass_impl() -> None:
        pass

    def test_fail_impl(message: str) -> None:
        raise AssertionError(message)

    return [
        PlatformFunction(
            name="describe",
            inputs=[StringType, FunctionType([], NullType)],
            output=NullType,
            type="async",
            fn=describe_impl,
        ),
        PlatformFunction(
            name="test",
            inputs=[StringType, FunctionType([], NullType)],
            output=NullType,
            type="async",
            fn=test_impl,
        ),
        PlatformFunction(
            name="testPass",
            inputs=[],
            output=NullType,
            type="sync",
            fn=test_pass_impl,
        ),
        PlatformFunction(
            name="testFail",
            inputs=[StringType],
            output=NullType,
            type="sync",
            fn=test_fail_impl,
        ),
    ], test_results


def get_test_ir_files() -> list[Path]:
    """Get available test IR files, preferring the profile IR directory."""
    for d in [PROFILE_IR_DIR, TEST_IR_DIR]:
        if d.exists():
            files = list(d.glob("*.beast2")) + list(d.glob("*.json"))
            if files:
                return sorted(files)
    return []


def count_east_objects() -> dict[str, int]:
    """Count live East value objects using gc."""
    counts: dict[str, int] = {}
    for obj in gc.get_objects():
        for east_type in EAST_TYPES:
            if isinstance(obj, east_type):
                name = type(obj).__name__
                counts[name] = counts.get(name, 0) + 1
                break
    return counts


def count_east_objects_fast() -> dict[str, int]:
    """Count live East value objects by type name (faster than isinstance)."""
    counts: Counter = Counter()
    type_names = {t.__name__ for t in EAST_TYPES}
    for obj in gc.get_objects():
        name = type(obj).__name__
        if name in type_names:
            counts[name] += 1
    return dict(counts)


def take_memory_snapshot() -> tuple[tracemalloc.Snapshot, dict[str, int]]:
    """Take a tracemalloc snapshot and count East objects."""
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    obj_counts = count_east_objects_fast()
    return snapshot, obj_counts


def format_snapshot_stats(
    snapshot: tracemalloc.Snapshot,
    top_n: int = 20,
    key_type: str = "lineno",
) -> list[dict]:
    """Extract top allocation sites from a snapshot."""
    stats = snapshot.statistics(key_type)
    results = []
    for stat in stats[:top_n]:
        results.append({
            "location": str(stat.traceback),
            "size": stat.size,
            "size_human": format_bytes(stat.size),
            "count": stat.count,
        })
    return results


def format_snapshot_diff(
    old: tracemalloc.Snapshot,
    new: tracemalloc.Snapshot,
    top_n: int = 20,
    key_type: str = "lineno",
) -> list[dict]:
    """Compare two snapshots and return top differences."""
    stats = new.compare_to(old, key_type)
    results = []
    for stat in stats[:top_n]:
        results.append({
            "location": str(stat.traceback),
            "size_diff": stat.size_diff,
            "size_diff_human": format_bytes(stat.size_diff),
            "size": stat.size,
            "size_human": format_bytes(stat.size),
            "count_diff": stat.count_diff,
            "count": stat.count,
        })
    return results


def format_obj_counts(counts: dict[str, int], title: str = "") -> str:
    """Format East object counts as a table."""
    lines = []
    if title:
        lines.append(f"  {title}:")
    total = 0
    for name, count in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {name:<20} {count:>8,}")
        total += count
    lines.append(f"    {'TOTAL':<20} {total:>8,}")
    return "\n".join(lines)


def format_obj_diff(before: dict[str, int], after: dict[str, int]) -> str:
    """Format the difference in object counts."""
    all_names = sorted(set(before) | set(after))
    lines = []
    total_diff = 0
    for name in all_names:
        b = before.get(name, 0)
        a = after.get(name, 0)
        diff = a - b
        if diff != 0:
            sign = "+" if diff > 0 else ""
            lines.append(f"    {name:<20} {b:>8,} -> {a:>8,}  ({sign}{diff:,})")
            total_diff += diff
    if not lines:
        lines.append("    (no change)")
    sign = "+" if total_diff > 0 else ""
    lines.append(f"    {'NET CHANGE':<20} {' ':>19}  ({sign}{total_diff:,})")
    return "\n".join(lines)


def profile_memory_for_file(
    ir_file: Path,
    platforms: list[PlatformFunction],
    compile_only: bool = False,
    top_n: int = 20,
    show_diff: bool = False,
) -> dict:
    """Profile memory usage for a single IR file across all phases."""
    is_beast2 = ir_file.suffix == ".beast2"
    format_name = "BEAST2" if is_beast2 else "JSON"

    print(f"\n{'=' * 72}")
    print(f"  {ir_file.name} ({format_name}, {ir_file.stat().st_size / 1024:.1f} KiB)")
    print(f"{'=' * 72}")

    result: dict[str, Any] = {
        "file": ir_file.name,
        "format": format_name,
        "file_size": ir_file.stat().st_size,
        "phases": {},
    }

    # Load file data
    with open(ir_file, "rb") as f:
        file_data = f.read()

    # Baseline snapshot
    gc.collect()
    snap_baseline, obj_baseline = take_memory_snapshot()
    tracemalloc.get_traced_memory()  # baseline checkpoint

    # -- Phase 1: Deserialization --
    if is_beast2:
        has_header = file_data[: len(BEAST2_MAGIC_BYTES)] == BEAST2_MAGIC_BYTES
        if has_header:
            decoder = decode_beast2_with_header_for(IRType)
        else:
            decoder = decode_beast2_for(IRType)
        ir = decoder(file_data)
    else:
        decoder = decode_json_for(IRType)
        ir = decoder(file_data)

    snap_deser, obj_deser = take_memory_snapshot()
    mem_deser = tracemalloc.get_traced_memory()

    phase_info = {
        "current_mem": format_bytes(mem_deser[0]),
        "peak_mem": format_bytes(mem_deser[1]),
        "current_mem_bytes": mem_deser[0],
        "peak_mem_bytes": mem_deser[1],
        "objects": obj_deser,
    }
    result["phases"]["deserialize"] = phase_info

    print("\n  After deserialization:")
    print(f"    Current: {format_bytes(mem_deser[0])}, Peak: {format_bytes(mem_deser[1])}")
    print(format_obj_counts(obj_deser, "Live East objects"))

    if show_diff:
        print(f"\n    Allocations (top {top_n}):")
        diff = format_snapshot_diff(snap_baseline, snap_deser, top_n)
        for d in diff:
            if d["size_diff"] > 0:
                print(f"      {d['size_diff_human']:>10}  ({d['count_diff']:>+6} allocs)  {d['location']}")

    # -- Phase 2: Compilation --
    is_async = ir.type == "AsyncFunction"
    if is_async:
        compiled = compile_async(ir, platforms)
    else:
        compiled = compile(ir, platforms)

    snap_compile, obj_compile = take_memory_snapshot()
    mem_compile = tracemalloc.get_traced_memory()

    phase_info = {
        "current_mem": format_bytes(mem_compile[0]),
        "peak_mem": format_bytes(mem_compile[1]),
        "current_mem_bytes": mem_compile[0],
        "peak_mem_bytes": mem_compile[1],
        "objects": obj_compile,
    }
    result["phases"]["compile"] = phase_info

    print("\n  After compilation:")
    print(f"    Current: {format_bytes(mem_compile[0])}, Peak: {format_bytes(mem_compile[1])}")
    print("  Object changes (deserialize -> compile):")
    print(format_obj_diff(obj_deser, obj_compile))

    if show_diff:
        print(f"\n    Allocations (top {top_n}):")
        diff = format_snapshot_diff(snap_deser, snap_compile, top_n)
        for d in diff:
            if d["size_diff"] > 0:
                print(f"      {d['size_diff_human']:>10}  ({d['count_diff']:>+6} allocs)  {d['location']}")

    # -- Phase 3: Execution --
    if not compile_only:
        if is_async:
            asyncio.run(compiled())
        else:
            compiled()

        snap_exec, obj_exec = take_memory_snapshot()
        mem_exec = tracemalloc.get_traced_memory()

        phase_info = {
            "current_mem": format_bytes(mem_exec[0]),
            "peak_mem": format_bytes(mem_exec[1]),
            "current_mem_bytes": mem_exec[0],
            "peak_mem_bytes": mem_exec[1],
            "objects": obj_exec,
        }
        result["phases"]["execute"] = phase_info

        print("\n  After execution:")
        print(f"    Current: {format_bytes(mem_exec[0])}, Peak: {format_bytes(mem_exec[1])}")
        print("  Object changes (compile -> execute):")
        print(format_obj_diff(obj_compile, obj_exec))

        if show_diff:
            print(f"\n    Allocations (top {top_n}):")
            diff = format_snapshot_diff(snap_compile, snap_exec, top_n)
            for d in diff:
                if d["size_diff"] > 0:
                    print(f"      {d['size_diff_human']:>10}  ({d['count_diff']:>+6} allocs)  {d['location']}")

    # -- Top allocation sites overall --
    final_snap = snap_exec if not compile_only else snap_compile
    print(f"\n  Top {top_n} allocation sites (cumulative):")
    top_allocs = format_snapshot_stats(final_snap, top_n)
    result["top_allocations"] = top_allocs
    for a in top_allocs:
        print(f"    {a['size_human']:>10}  ({a['count']:>6} allocs)  {a['location']}")

    return result


def check_for_leaks(
    ir_file: Path,
    platforms: list[PlatformFunction],
    iterations: int = 5,
) -> dict:
    """Run multiple iterations and check if memory grows unboundedly."""
    is_beast2 = ir_file.suffix == ".beast2"
    format_name = "BEAST2" if is_beast2 else "JSON"

    print(f"\n{'=' * 72}")
    print(f"  LEAK CHECK: {ir_file.name} ({format_name})")
    print(f"  Running {iterations} iterations...")
    print(f"{'=' * 72}")

    with open(ir_file, "rb") as f:
        file_data = f.read()

    if is_beast2:
        has_header = file_data[: len(BEAST2_MAGIC_BYTES)] == BEAST2_MAGIC_BYTES
        if has_header:
            decoder = decode_beast2_with_header_for(IRType)
        else:
            decoder = decode_beast2_for(IRType)
    else:
        decoder = decode_json_for(IRType)

    mem_per_iter = []
    obj_per_iter = []

    for i in range(iterations):
        gc.collect()
        tracemalloc.clear_traces()

        # Full pipeline: deserialize -> compile -> execute
        ir = decoder(file_data)
        is_async = ir.type == "AsyncFunction"
        if is_async:
            compiled = compile_async(ir, platforms)
            asyncio.run(compiled())
        else:
            compiled = compile(ir, platforms)
            compiled()

        # Clean up this iteration's results
        del ir, compiled
        gc.collect()

        current, peak = tracemalloc.get_traced_memory()
        obj_counts = count_east_objects_fast()
        mem_per_iter.append(current)
        obj_per_iter.append(sum(obj_counts.values()))

        print(f"  Iteration {i + 1}: {format_bytes(current)} current, "
              f"{sum(obj_counts.values()):,} East objects")

    # Analyze growth
    if len(mem_per_iter) >= 3:
        # Check if memory is growing linearly
        first_half_avg = sum(mem_per_iter[: len(mem_per_iter) // 2]) / (len(mem_per_iter) // 2)
        second_half_avg = sum(mem_per_iter[len(mem_per_iter) // 2 :]) / (
            len(mem_per_iter) - len(mem_per_iter) // 2
        )
        growth_ratio = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0

        first_obj_avg = sum(obj_per_iter[: len(obj_per_iter) // 2]) / (len(obj_per_iter) // 2)
        second_obj_avg = sum(obj_per_iter[len(obj_per_iter) // 2 :]) / (
            len(obj_per_iter) - len(obj_per_iter) // 2
        )
        obj_growth_ratio = second_obj_avg / first_obj_avg if first_obj_avg > 0 else 1.0

        print(f"\n  Memory growth ratio (2nd half / 1st half): {growth_ratio:.2f}x")
        print(f"  Object growth ratio (2nd half / 1st half): {obj_growth_ratio:.2f}x")

        if growth_ratio > 1.2:
            print(f"  WARNING: Memory appears to be growing ({growth_ratio:.2f}x). Possible leak.")
        elif growth_ratio > 1.05:
            print(f"  NOTICE: Slight memory growth ({growth_ratio:.2f}x). May be cache warming.")
        else:
            print(f"  OK: Memory stable ({growth_ratio:.2f}x).")

    return {
        "file": ir_file.name,
        "iterations": iterations,
        "memory_per_iteration": mem_per_iter,
        "objects_per_iteration": obj_per_iter,
    }


def print_global_cache_sizes() -> None:
    """Report sizes of known global caches in east-py."""
    print(f"\n{'=' * 72}")
    print("  GLOBAL CACHES")
    print(f"{'=' * 72}")

    # _key_cache in values.py (struct key interning)
    from east.types.values import _key_cache
    print(f"  values._key_cache (struct key interning): {len(_key_cache)} entries")

    # Builtin registry
    try:
        from east.builtins.registry import _BUILTINS
        print(f"  builtins._BUILTINS (builtin registry):    {len(_BUILTINS)} entries")
    except ImportError:
        pass

    # _cached_make_east_key
    from east.types.values import _cached_make_east_key
    print(f"  values._cached_make_east_key:              {'loaded' if _cached_make_east_key else 'not loaded'}")


def main():
    parser = argparse.ArgumentParser(description="Memory profiler for East-py")
    parser.add_argument(
        "ir_files",
        nargs="*",
        help="IR files to profile (default: auto-discover test files)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Profile deserialization and compilation only, skip execution",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top allocation sites to show (default: 20)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show allocation diffs between phases",
    )
    parser.add_argument(
        "--leak-check",
        action="store_true",
        help="Run multiple iterations to check for memory leaks",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations for leak check (default: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--traceback-depth",
        type=int,
        default=1,
        help="tracemalloc frame depth (default: 1, higher = more detail but slower)",
    )

    args = parser.parse_args()

    # Get files
    if args.ir_files:
        ir_files = [Path(f) for f in args.ir_files]
        for f in ir_files:
            if not f.exists():
                print(f"Error: File not found: {f}")
                sys.exit(1)
    else:
        ir_files = get_test_ir_files()
        if not ir_files:
            print(f"Error: No test IR files found in {PROFILE_IR_DIR} or {TEST_IR_DIR}")
            print("Run the profile generator or 'make test-export' first.")
            sys.exit(1)

    # Start tracemalloc
    tracemalloc.start(args.traceback_depth)

    print("East-py Memory Profiler")
    print("=======================")
    print(f"Files: {len(ir_files)}")
    print(f"Mode: {'leak check' if args.leak_check else 'profile'}")
    print(f"tracemalloc depth: {args.traceback_depth}")

    platforms, _ = create_test_platforms()
    all_results = []

    if args.leak_check:
        for ir_file in ir_files:
            try:
                result = check_for_leaks(ir_file, platforms, args.iterations)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback as tb
                tb.print_exc()
    else:
        for ir_file in ir_files:
            try:
                result = profile_memory_for_file(
                    ir_file, platforms, args.compile_only, args.top, args.diff,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback as tb
                tb.print_exc()

    # Global cache report
    print_global_cache_sizes()

    # Summary
    print(f"\n{'=' * 72}")
    print("  SUMMARY")
    print(f"{'=' * 72}")
    current, peak = tracemalloc.get_traced_memory()
    print(f"  Final traced memory:  {format_bytes(current)}")
    print(f"  Peak traced memory:   {format_bytes(peak)}")

    tracemalloc.stop()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
