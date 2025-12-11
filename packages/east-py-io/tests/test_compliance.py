"""Tests that run TypeScript-exported IR tests for east-py-io.

This module loads IR test files exported from the TypeScript east-node-io package
via `npm run test:export` and executes them in Python to verify cross-implementation
compatibility.

To generate the test IR files:
    cd ../east-node/packages/east-node-io && npm run test:export

Note: Many I/O tests require external services (databases, S3, etc.) to be running.
Use `make test:integration` to run tests with Docker services.
"""

import asyncio
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
from east_py_std import console_impl

from east_py_io import platform

# Path where TypeScript exports test IR
TEST_IR_DIR = Path("/tmp/east-node-io")

# Log file for detailed errors
ERROR_LOG_DIR = Path("/tmp/east-py-io-test-errors")

# Profiling output directory
PROFILING_DIR = Path("/tmp/east-py-io-profiling")


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
            "Run 'cd ../east-node/packages/east-node-io && npm run test:export' to generate test files."
        )

    # Get all JSON files
    files = list(TEST_IR_DIR.glob("*.json"))

    if not files:
        raise FileNotFoundError(
            f"No test IR files found in {TEST_IR_DIR}. "
            "Run 'cd ../east-node/packages/east-node-io && npm run test:export' to generate test files."
        )

    return sorted(files)


@pytest.fixture
def test_platforms(subtests):
    """Platform functions for tests - combines platform with test tracking."""
    # Track test execution and failures with context
    executed_tests = []
    failures = []
    current_test_stack = []

    # Track timing for profiling
    describe_timings = []
    test_timings = []

    async def describe_impl(name: str, test_fn: Any) -> None:
        """Execute a test suite described by name."""
        start_time = time.time()
        executed_tests.append(("describe", name))
        current_test_stack.append(("describe", name))

        with subtests.test(msg=f"[{name}]"):
            try:
                if callable(test_fn):
                    result = test_fn()
                    if asyncio.iscoroutine(result):
                        await result
            finally:
                duration = time.time() - start_time
                describe_timings.append((name, duration))
                current_test_stack.pop()

    async def test_impl_fn(name: str, test_fn: Any) -> None:
        """Execute a single test described by name."""
        start_time = time.time()
        test_path = " > ".join(name for _, name in current_test_stack) + f" > {name}"

        executed_tests.append(("test", name, test_path))
        current_test_stack.append(("test", name))

        with subtests.test(msg=test_path):
            try:
                if callable(test_fn):
                    result = test_fn()
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as e:
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
                raise
            finally:
                duration = time.time() - start_time
                test_timings.append((test_path, duration))
                current_test_stack.pop()

    def test_pass_impl() -> None:
        """Assertion passed - do nothing."""
        pass

    def test_fail_impl(message: str) -> None:
        """Assertion failed - raise exception."""
        raise AssertionError(message)

    # Create test platform functions with tracking
    test_platform_fns = [
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
            fn=test_impl_fn,
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
    ]

    # Combine with platform
    # Combine IO platform, console (from std), and test platform functions
    platform_fns = list(platform) + list(console_impl) + test_platform_fns

    return (
        platform_fns,
        executed_tests,
        failures,
        describe_timings,
        test_timings,
    )


@pytest.mark.parametrize(
    "test_file",
    get_test_ir_files(),
    ids=lambda p: p.stem,
)
def test_typescript_exported_ir(test_file, test_platforms):
    """Test that TypeScript-exported IR executes correctly in Python."""
    platform_fns, executed_tests, failures, describe_timings, test_timings = test_platforms

    ERROR_LOG_DIR.mkdir(exist_ok=True)
    PROFILING_DIR.mkdir(exist_ok=True)

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

    # Stage 3: Compile IR
    compile_start = time.time()
    is_async_ir = ir.type == "AsyncFunction"
    compiled_test = compile_async(ir, platform_fns) if is_async_ir else compile(ir, platform_fns)
    stage_timings["compile"] = time.time() - compile_start

    start_time = time.time()

    try:
        print(f"\n{test_file.stem} test cases:", flush=True)
        execute_start = time.time()
        if is_async_ir:
            asyncio.run(compiled_test())
        else:
            compiled_test()
        stage_timings["execute"] = time.time() - execute_start

        duration = time.time() - start_time

        assert len(executed_tests) > 0, f"Test {test_file.stem} didn't execute any test cases"

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
            "total_duration": duration,
        }

        profile_file = (
            PROFILING_DIR / f"{test_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(profile_file, "w") as f:
            json.dump(profiling_data, f, indent=2)

        if failures:
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

            failure_summary = [f"  ✗ {f['path']}: {f['error']}" for f in failures[:10]]
            pytest.fail(
                f"\n{len(failures)}/{test_count} test(s) failed. Details logged to: {log_file}\n\n"
                + "\n".join(failure_summary)
                + (f"\n  ... and {len(failures) - 10} more" if len(failures) > 10 else "")
            )

    except Exception as e:
        duration = time.time() - start_time
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

        pytest.fail(
            f"Failed to execute TypeScript test IR from {test_file.name} ({duration:.2f}s)\n"
            f"Error: {e}\n"
            f"Details logged to: {log_file}"
        )


def test_typescript_test_ir_directory_exists():
    """Verify that TypeScript test IR directory exists."""
    if not TEST_IR_DIR.exists():
        pytest.skip(
            f"Test IR directory {TEST_IR_DIR} not found. "
            "Run 'cd ../east-node/packages/east-node-io && npm run test:export' to generate test files."
        )

    files = get_test_ir_files()
    assert len(files) > 0, f"No test IR files found in {TEST_IR_DIR}"


if __name__ == "__main__":
    try:
        files = get_test_ir_files()
        print(f"Found {len(files)} TypeScript test IR files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    except FileNotFoundError as e:
        print(str(e))
