#!/usr/bin/env python3
#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Profiling script for East-py compilation and execution.

This script profiles the East IR compilation and execution pipeline to identify
performance bottlenecks. It can profile:
1. Deserialization (JSON or BEAST2 binary format)
2. IR compilation (compile/compile_async)
3. Runtime execution

Usage:
    # Profile using test IR files (requires `make test-export` first)
    uv run --package east-py python packages/east-py/scripts/east_profiler.py

    # Profile a specific IR file (JSON or BEAST2)
    uv run --package east-py python packages/east-py/scripts/east_profiler.py /tmp/east-profile-ir/array_operations_medium.json
    uv run --package east-py python packages/east-py/scripts/east_profiler.py /tmp/east-profile-ir/array_operations_medium.beast2

    # Profile BEAST2 files to see compilation/execution without JSON overhead
    uv run --package east-py python packages/east-py/scripts/east_profiler.py /tmp/east-profile-ir/*.beast2

    # Profile with flame graph output (requires py-spy)
    py-spy record -o profile.svg -- uv run --package east-py python packages/east-py/scripts/east_profiler.py

    # Profile compilation only (skip execution)
    uv run --package east-py python packages/east-py/scripts/east_profiler.py --compile-only

    # Show top N functions
    uv run --package east-py python packages/east-py/scripts/east_profiler.py --top 50

    # Save detailed results to JSON
    uv run --package east-py python packages/east-py/scripts/east_profiler.py -o profile-results.json
"""

import argparse
import asyncio
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
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

# Default test IR directory
TEST_IR_DIR = Path("/tmp/east-test-ir")


def create_test_platforms() -> list[PlatformFunction]:
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
    """Get available test IR files (JSON and BEAST2)."""
    if not TEST_IR_DIR.exists():
        return []
    json_files = list(TEST_IR_DIR.glob("*.json"))
    beast2_files = list(TEST_IR_DIR.glob("*.beast2"))
    return sorted(json_files + beast2_files)


def profile_phase(_name: str, func, *args, **kwargs) -> tuple[Any, float, pstats.Stats]:
    """Profile a single phase and return result, duration, and stats."""
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    duration = time.perf_counter() - start

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumulative")

    return result, duration, stats


def extract_top_functions(stats: pstats.Stats, n: int = 20) -> list[dict]:
    """Extract top N functions from profile stats."""
    # Get stats as list of tuples
    stats_list = []
    for func, (_cc, nc, tt, ct, _callers) in stats.stats.items():
        filename, line, name = func
        stats_list.append({
            "function": f"{filename}:{line}({name})",
            "name": name,
            "file": filename,
            "line": line,
            "calls": nc,
            "tottime": tt,
            "cumtime": ct,
            "percall_tot": tt / nc if nc > 0 else 0,
            "percall_cum": ct / nc if nc > 0 else 0,
        })

    # Sort by cumulative time
    stats_list.sort(key=lambda x: x["cumtime"], reverse=True)
    return stats_list[:n]


def format_stats_table(stats_list: list[dict], title: str) -> str:
    """Format stats as a readable table."""
    lines = [
        f"\n{'=' * 80}",
        f" {title}",
        f"{'=' * 80}",
        f"{'Calls':>10} {'TotTime':>10} {'PerCall':>10} {'CumTime':>10} {'PerCall':>10}  Function",
        f"{'-' * 80}",
    ]

    for s in stats_list:
        calls = s["calls"]
        percall_tot = s.get("percall_tot", s["tottime"] / calls if calls > 0 else 0)
        percall_cum = s.get("percall_cum", s["cumtime"] / calls if calls > 0 else 0)
        lines.append(
            f"{calls:>10} {s['tottime']:>10.4f} {percall_tot:>10.6f} "
            f"{s['cumtime']:>10.4f} {percall_cum:>10.6f}  {s['name']}"
        )
        # Show file location for top entries
        if s == stats_list[0] or s["cumtime"] > stats_list[0]["cumtime"] * 0.1:
            lines.append(f"{'':>56}  └─ {s['file']}:{s['line']}")

    return "\n".join(lines)


def analyze_hotspots(all_stats: dict[str, list[dict]]) -> dict:
    """Analyze hotspots across all phases."""
    # Aggregate by function name across phases
    func_totals = defaultdict(lambda: {"calls": 0, "tottime": 0, "cumtime": 0, "phases": []})

    for phase, stats in all_stats.items():
        for s in stats:
            key = s["name"]
            func_totals[key]["calls"] += s["calls"]
            func_totals[key]["tottime"] += s["tottime"]
            func_totals[key]["cumtime"] += s["cumtime"]
            func_totals[key]["phases"].append(phase)
            func_totals[key]["file"] = s["file"]
            func_totals[key]["line"] = s["line"]

    # Sort by total cumulative time
    sorted_funcs = sorted(func_totals.items(), key=lambda x: x[1]["cumtime"], reverse=True)

    return {
        "top_functions": sorted_funcs[:30],
        "by_phase": all_stats,
    }


def identify_optimization_opportunities(analysis: dict) -> list[str]:
    """Identify potential optimization opportunities."""
    suggestions = []
    top_funcs = analysis["top_functions"]

    if not top_funcs:
        return ["No profiling data collected"]

    total_time = sum(f[1]["cumtime"] for f in top_funcs)

    for name, stats in top_funcs[:15]:
        pct = (stats["cumtime"] / total_time * 100) if total_time > 0 else 0
        calls = stats["calls"]
        avg_time = stats["cumtime"] / calls if calls > 0 else 0

        # Identify specific patterns
        if "decode" in name.lower() or "json" in name.lower():
            if pct > 10:
                suggestions.append(
                    f"JSON DESERIALIZATION ({pct:.1f}%): {name} called {calls:,} times. "
                    "Consider caching decoded IR or using a faster JSON parser (orjson/ujson)."
                )

        elif "_compile" in name:
            if pct > 5:
                suggestions.append(
                    f"COMPILATION ({pct:.1f}%): {name} called {calls:,} times. "
                    f"Avg {avg_time*1000:.3f}ms per call. "
                    "Consider caching compiled functions by IR hash."
                )

        elif name in ("__getitem__", "__setitem__", "__contains__"):
            if calls > 10000:
                suggestions.append(
                    f"DICT ACCESS ({pct:.1f}%): {name} called {calls:,} times. "
                    "Environment lookups may be a bottleneck. Consider slot-based access."
                )

        elif "EastArray" in name or "EastDict" in name or "EastSet" in name:
            if pct > 5:
                suggestions.append(
                    f"CONTAINER OPS ({pct:.1f}%): {name} called {calls:,} times. "
                    "Consider batch operations or native Python types for hot paths."
                )

        elif name == "make" or "Factory" in name:
            if calls > 1000:
                suggestions.append(
                    f"FUNCTION FACTORY ({pct:.1f}%): {name} called {calls:,} times. "
                    "Consider caching factories by (builtin_name, type_params) tuple."
                )

        elif ("match" in name.lower() or "variant" in name.lower()) and pct > 3:
            suggestions.append(
                f"PATTERN MATCHING ({pct:.1f}%): {name} called {calls:,} times. "
                "Consider optimizing variant dispatch with direct case lookup."
            )

    # Add general observations
    if "deserialize" in analysis["by_phase"]:
        deser_time = sum(s["cumtime"] for s in analysis["by_phase"]["deserialize"][:5])
        compile_time = sum(s["cumtime"] for s in analysis["by_phase"].get("compile", [])[:5])
        if deser_time > compile_time * 2:
            suggestions.append(
                f"PHASE IMBALANCE: Deserialization ({deser_time:.2f}s) >> Compilation ({compile_time:.2f}s). "
                "Binary serialization (BEAST) would significantly reduce load time."
            )

    return suggestions


def profile_ir_file(
    ir_file: Path,
    platforms: list[PlatformFunction],
    compile_only: bool = False,
    top_n: int = 20,
) -> dict:
    """Profile a single IR file through all phases."""
    is_beast2 = ir_file.suffix == ".beast2"
    format_name = "BEAST2" if is_beast2 else "JSON"

    print(f"\nProfiling: {ir_file.name} ({format_name})")
    print(f"  Size: {ir_file.stat().st_size / 1024:.1f} KB")

    results = {"file": ir_file.name, "format": format_name, "phases": {}, "top_functions": {}}

    # Phase 1: Load file
    with open(ir_file, "rb") as f:
        file_data = f.read()
    print(f"  Loaded {len(file_data):,} bytes")

    # Phase 2: Deserialization (JSON or BEAST2)
    if is_beast2:
        # Check if file has magic header
        has_header = file_data[:len(BEAST2_MAGIC_BYTES)] == BEAST2_MAGIC_BYTES
        if has_header:
            def deserialize():
                decoder = decode_beast2_with_header_for(IRType)
                return decoder(file_data)
        else:
            def deserialize():
                decoder = decode_beast2_for(IRType)
                return decoder(file_data)
    else:
        def deserialize():
            decoder = decode_json_for(IRType)
            return decoder(file_data)

    ir, deser_time, deser_stats = profile_phase("deserialize", deserialize)
    results["phases"]["deserialize"] = deser_time
    results["top_functions"]["deserialize"] = extract_top_functions(deser_stats, top_n)
    print(f"  Deserialized ({format_name}) in {deser_time:.3f}s")

    # Phase 3: Compilation
    is_async = ir.type == "AsyncFunction"

    def compile_ir():
        if is_async:
            return compile_async(ir, platforms)
        return compile(ir, platforms)

    compiled, compile_time, compile_stats = profile_phase("compile", compile_ir)
    results["phases"]["compile"] = compile_time
    results["top_functions"]["compile"] = extract_top_functions(compile_stats, top_n)
    print(f"  Compiled in {compile_time:.3f}s (async={is_async})")

    # Phase 4: Execution (optional)
    if not compile_only:
        def execute():
            if is_async:
                asyncio.run(compiled())
            else:
                compiled()

        _, exec_time, exec_stats = profile_phase("execute", execute)
        results["phases"]["execute"] = exec_time
        results["top_functions"]["execute"] = extract_top_functions(exec_stats, top_n)
        print(f"  Executed in {exec_time:.3f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Profile East-py compilation and execution")
    parser.add_argument(
        "ir_files",
        nargs="*",
        help="IR files to profile (default: profile all test files). Supports JSON and BEAST2 formats.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Profile compilation only, skip execution",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top functions to show per phase (default: 20)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for detailed results",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate flamegraph (requires py-spy to be installed separately)",
    )

    args = parser.parse_args()

    # Get files to profile
    if args.ir_files:
        ir_files = [Path(f) for f in args.ir_files]
        for f in ir_files:
            if not f.exists():
                print(f"Error: File not found: {f}")
                sys.exit(1)
    else:
        ir_files = get_test_ir_files()
        if not ir_files:
            print(f"Error: No test IR files found in {TEST_IR_DIR}")
            print("Run 'make test-export' in the east-py-datascience package first.")
            sys.exit(1)

    print("East-py Profiler")
    print("================")
    print(f"Files to profile: {len(ir_files)}")
    print(f"Compile only: {args.compile_only}")

    # Create platforms
    platforms, test_results = create_test_platforms()

    # Profile each file
    all_results = []
    all_stats: dict[str, list[dict]] = defaultdict(list)

    for ir_file in ir_files:
        try:
            result = profile_ir_file(
                ir_file, platforms, args.compile_only, args.top
            )
            all_results.append(result)

            # Aggregate stats
            for phase, stats in result["top_functions"].items():
                all_stats[phase].extend(stats)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Analyze aggregate results
    print("\n" + "=" * 80)
    print(" AGGREGATE ANALYSIS")
    print("=" * 80)

    # Phase timing summary
    phase_totals = defaultdict(float)
    for result in all_results:
        for phase, duration in result["phases"].items():
            phase_totals[phase] += duration

    total_time = sum(phase_totals.values())
    print(f"\nPhase Breakdown (total: {total_time:.2f}s):")
    for phase, ptime in sorted(phase_totals.items(), key=lambda x: -x[1]):
        pct = ptime / total_time * 100 if total_time > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {phase:<15} {ptime:>8.3f}s ({pct:>5.1f}%) {bar}")

    # Analyze hotspots
    analysis = analyze_hotspots(dict(all_stats))

    # Show top functions per phase
    for phase in ["deserialize", "compile", "execute"]:
        if phase in all_stats:
            # Deduplicate and sum
            func_map = defaultdict(lambda: {"calls": 0, "tottime": 0, "cumtime": 0})
            for s in all_stats[phase]:
                key = s["name"]
                func_map[key]["calls"] += s["calls"]
                func_map[key]["tottime"] += s["tottime"]
                func_map[key]["cumtime"] += s["cumtime"]
                func_map[key]["file"] = s["file"]
                func_map[key]["line"] = s["line"]
                func_map[key]["name"] = s["name"]

            sorted_funcs = sorted(func_map.values(), key=lambda x: x["cumtime"], reverse=True)
            print(format_stats_table(sorted_funcs[:args.top], f"TOP FUNCTIONS: {phase.upper()}"))

    # Optimization suggestions
    suggestions = identify_optimization_opportunities(analysis)
    if suggestions:
        print("\n" + "=" * 80)
        print(" OPTIMIZATION OPPORTUNITIES")
        print("=" * 80)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. {suggestion}")

    # Save detailed results
    if args.output:
        output_data = {
            "files": all_results,
            "phase_totals": dict(phase_totals),
            "suggestions": suggestions,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")

    # Flamegraph hint
    if args.flamegraph:
        print("\n" + "=" * 80)
        print(" FLAMEGRAPH")
        print("=" * 80)
        print("To generate a flamegraph, run:")
        print(f"  py-spy record -o profile.svg -- uv run --package east-py python {__file__}")
        print("Then open profile.svg in a browser.")


if __name__ == "__main__":
    main()
