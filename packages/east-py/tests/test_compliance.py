#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Tests that run TypeScript-exported IR tests.

This module loads IR test files exported from the TypeScript ../East repository
via `npm run test:export` and executes them in Python to verify cross-implementation
compatibility.

To generate the test IR files:
    cd ../East && npm run test:export

This will create JSON files in /tmp/east-test-ir/ containing compiled IR for each test.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from east.runtime.compiler import compile, compile_async
from east.runtime.platform import PlatformFunction
from east.serialization.json import decode_json_for
from east.types.type_of_type import IRType
from east.types.types import FunctionType, NullType, StringType

# Path where TypeScript exports test IR
TEST_IR_DIR = Path("/tmp/east-test-ir")

# Log file for detailed errors
ERROR_LOG_DIR = Path("/tmp/east-py-test-errors")

# Profiling output directory
PROFILING_DIR = Path("/tmp/east-py-profiling")


def get_test_ir_files():
    """Get list of exported test IR JSON files.

    Returns:
        List of Path objects for test IR files

    Raises:
        FileNotFoundError: If test IR directory doesn't exist
    """
    if not TEST_IR_DIR.exists():
        raise FileNotFoundError(
            f"Test IR directory {TEST_IR_DIR} not found. "
            "Run 'cd ../East && npm run test:export' to generate test files."
        )

    # Get all JSON files, excluding ones with "___" (sub-test files)
    files = [
        f
        for f in TEST_IR_DIR.glob("*.json")
        if "___" not in f.name  # Skip sub-test files for now
    ]

    if not files:
        raise FileNotFoundError(
            f"No test IR files found in {TEST_IR_DIR}. "
            "Run 'cd ../East && npm run test:export' to generate test files."
        )

    return sorted(files)


@pytest.fixture
def test_platforms(subtests):
    """Platform functions for 'describe', 'test', 'testPass', 'testFail' - used by test IR."""
    # Track test execution and failures with context
    executed_tests = []
    failures = []
    current_test_stack = []  # Stack of describe/test names

    # Track timing for profiling
    describe_timings = []  # List of (name, duration) tuples
    test_timings = []  # List of (path, duration) tuples

    async def describe_impl(name: str, test_fn: Any) -> None:
        """Execute a test suite described by name."""
        import asyncio

        start_time = time.time()
        executed_tests.append(("describe", name))
        current_test_stack.append(("describe", name))

        # Use pytest subtest for describe block
        with subtests.test(msg=f"[{name}]"):
            try:
                # Execute the test function (may be async)
                if callable(test_fn):
                    result = test_fn()
                    if asyncio.iscoroutine(result):
                        await result
            finally:
                duration = time.time() - start_time
                describe_timings.append((name, duration))
                current_test_stack.pop()

    async def test_impl(name: str, test_fn: Any) -> None:
        """Execute a single test described by name."""
        import asyncio

        start_time = time.time()

        # Build test path before execution
        test_path = " > ".join(name for _, name in current_test_stack) + f" > {name}"

        executed_tests.append(("test", name, test_path))
        current_test_stack.append(("test", name))

        # Use pytest subtest for each test
        with subtests.test(msg=test_path):
            try:
                # Execute the test function (may be async)
                if callable(test_fn):
                    result = test_fn()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
                # Get full traceback
                import traceback

                error_detail = traceback.format_exc()

                failures.append(
                    {
                        "path": test_path,
                        "error": str(e),
                        "error_detail": error_detail,
                        "test_name": name,
                    }
                )
                # Re-raise to fail the subtest
                raise
            finally:
                duration = time.time() - start_time
                test_timings.append((test_path, duration))
                current_test_stack.pop()

    def test_pass_impl() -> None:
        """Assertion passed - do nothing."""
        pass

    def test_fail_impl(message: str) -> None:
        """Assertion failed - raise exception to fail the test."""
        # Raise exception to fail the test (matches TypeScript behavior)
        raise AssertionError(message)

    describe_fn = PlatformFunction(
        name="describe",
        inputs=[StringType, FunctionType([], NullType)],
        output=NullType,
        type="async",
        fn=describe_impl,
    )

    test_fn = PlatformFunction(
        name="test",
        inputs=[StringType, FunctionType([], NullType)],
        output=NullType,
        type="async",
        fn=test_impl,
    )

    test_pass_fn = PlatformFunction(
        name="testPass",
        inputs=[],
        output=NullType,
        type="sync",
        fn=test_pass_impl,
    )

    test_fail_fn = PlatformFunction(
        name="testFail",
        inputs=[StringType],
        output=NullType,
        type="sync",
        fn=test_fail_impl,
    )

    return (
        [describe_fn, test_fn, test_pass_fn, test_fail_fn],
        executed_tests,
        failures,
        describe_timings,
        test_timings,
    )


def load_and_compile_test_ir(ir_file: Path, platform_fns: list[PlatformFunction]):
    """Load a test IR JSON file and compile it.

    Args:
        ir_file: Path to JSON file containing IR
        platform_fns: List of platform functions to use during compilation

    Returns:
        Compiled function ready to execute
    """
    # Read JSON file
    with open(ir_file, "rb") as f:
        json_data = f.read()

    # Decode JSON to IR
    decoder = decode_json_for(IRType)
    ir = decoder(json_data)

    # Compile IR
    compiled = compile(ir, platform_fns)

    return compiled


@pytest.mark.parametrize(
    "test_file",
    get_test_ir_files(),
    ids=lambda p: p.stem,
)
def test_typescript_exported_ir(test_file, test_platforms):
    """Test that TypeScript-exported IR executes correctly in Python.

    This test loads IR exported from the TypeScript implementation and verifies
    it can be decoded, compiled, and executed in Python without errors.
    """
    platform_fns, executed_tests, failures, describe_timings, test_timings = test_platforms

    # Create error log and profiling directories
    ERROR_LOG_DIR.mkdir(exist_ok=True)
    PROFILING_DIR.mkdir(exist_ok=True)

    # Track timing for each stage
    stage_timings = {}

    # Stage 1: Load file
    load_start = time.time()
    with open(test_file, "rb") as f:
        json_data = f.read()
    stage_timings["load_file"] = time.time() - load_start
    stage_timings["file_size_mb"] = len(json_data) / (1024 * 1024)

    # Stage 2: Deserialize JSON
    deserialize_start = time.time()
    decoder = decode_json_for(IRType)
    ir = decoder(json_data)
    stage_timings["deserialize"] = time.time() - deserialize_start

    # Stage 3: Compile IR (use async compiler if IR is AsyncFunction)
    import asyncio

    compile_start = time.time()
    is_async_ir = ir.type == "AsyncFunction"
    if is_async_ir:
        compiled_test = compile_async(ir, platform_fns)
    else:
        compiled_test = compile(ir, platform_fns)
    stage_timings["compile"] = time.time() - compile_start

    # Track overall test duration
    start_time = time.time()

    try:
        # Stage 4: Execute the compiled test
        print(f"\n{test_file.stem} test cases:", flush=True)
        execute_start = time.time()
        if is_async_ir:
            asyncio.run(compiled_test())
        else:
            compiled_test()
        stage_timings["execute"] = time.time() - execute_start

        duration = time.time() - start_time

        # Verify test executed (should have called describe at least once)
        assert len(executed_tests) > 0, f"Test {test_file.stem} didn't execute any test cases"

        # Count test cases (not describe blocks)
        test_count = sum(1 for t in executed_tests if t[0] == "test")
        passed_count = test_count - len(failures)

        print(f"\n  Summary: {passed_count}/{test_count} passed ({duration:.2f}s)")

        # Write profiling data
        profiling_data = {
            "test_file": test_file.stem,
            "timestamp": datetime.now().isoformat(),
            "stage_timings": stage_timings,
            "test_count": test_count,
            "passed_count": passed_count,
            "failed_count": len(failures),
            "describe_timings": [{"name": name, "duration": dur} for name, dur in describe_timings],
            "test_timings": [{"path": path, "duration": dur} for path, dur in test_timings],
            "total_duration": duration,
            "avg_test_duration": stage_timings["execute"] / test_count if test_count > 0 else 0,
        }

        profile_file = (
            PROFILING_DIR / f"{test_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(profile_file, "w") as f:
            json.dump(profiling_data, f, indent=2)

        print(f"  Profiling: {profile_file}")

        # Check for test failures
        if failures:
            # Write detailed errors to log file
            log_file = (
                ERROR_LOG_DIR / f"{test_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            with open(log_file, "w") as f:
                f.write(f"TypeScript Test: {test_file.stem}\n")
                f.write(f"Failures: {len(failures)}/{test_count}\n")
                f.write("=" * 80 + "\n\n")

                for failure in failures:
                    f.write(f"Test: {failure['path']}\n")
                    f.write(f"Error: {failure['error']}\n")
                    if failure.get("error_detail"):
                        f.write(f"\nDetails:\n{failure['error_detail']}\n")
                    f.write("=" * 80 + "\n\n")

            # Show condensed error in pytest output
            failure_summary = []
            for f in failures:
                failure_summary.append(f"  ✗ {f['path']}: {f['error']}")

            pytest.fail(
                f"\n{len(failures)}/{test_count} test(s) failed. "
                f"Details logged to: {log_file}\n\n"
                + "\n".join(failure_summary[:10])  # Show first 10
                + (f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else "")
            )

    except Exception as e:
        duration = time.time() - start_time

        # Write compilation/execution error to log file
        import traceback

        tb = traceback.format_exc()

        log_file = (
            ERROR_LOG_DIR / f"{test_file.stem}_CRASH_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        with open(log_file, "w") as f:
            f.write(f"TypeScript Test: {test_file.stem}\n")
            f.write("COMPILATION/EXECUTION ERROR\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Error: {e}\n\n")
            f.write(f"Traceback:\n{tb}\n")

        # Write profiling data even on crash
        profiling_data = {
            "test_file": test_file.stem,
            "timestamp": datetime.now().isoformat(),
            "stage_timings": stage_timings,
            "error": str(e),
            "error_type": "CRASH",
            "total_duration": duration,
        }

        profile_file = (
            PROFILING_DIR
            / f"{test_file.stem}_CRASH_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(profile_file, "w") as f:
            json.dump(profiling_data, f, indent=2)

        pytest.fail(
            f"Failed to execute TypeScript test IR from {test_file.name} ({duration:.2f}s)\n"
            f"Error: {e}\n"
            f"Details logged to: {log_file}\n"
            f"Profiling: {profile_file}"
        )


def test_typescript_test_ir_directory_exists():
    """Verify that TypeScript test IR directory exists."""
    if not TEST_IR_DIR.exists():
        pytest.fail(
            f"Test IR directory {TEST_IR_DIR} not found. "
            "Run 'cd ../East && npm run test:export' to generate test files."
        )

    # Verify it has JSON files
    files = get_test_ir_files()
    assert len(files) > 0, f"No test IR files found in {TEST_IR_DIR}"


if __name__ == "__main__":
    # When run directly, show available test files
    try:
        files = get_test_ir_files()
        print(f"Found {len(files)} TypeScript test IR files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    except FileNotFoundError as e:
        print(str(e))
