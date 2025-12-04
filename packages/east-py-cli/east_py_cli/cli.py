"""CLI argument parsing and main entry point."""

import argparse
import sys
from pathlib import Path

from east_py_cli.loader import load_ir, load_runtime
from east_py_cli.runner import run_program


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="east-py",
        description="Run East IR programs with Python platform functions",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run an East IR program")
    run_parser.add_argument(
        "ir_file",
        type=Path,
        help="Path to IR file (.beast2, .beast, .east, or .json)",
    )
    run_parser.add_argument(
        "-r",
        "--runtime",
        action="append",
        default=[],
        metavar="PACKAGE",
        help="Python package providing platform functions (can be repeated)",
    )
    run_parser.add_argument(
        "--std",
        action="store_true",
        help="Shorthand for --runtime east-py-std",
    )
    run_parser.add_argument(
        "--io",
        action="store_true",
        help="Shorthand for --runtime east-py-io",
    )
    run_parser.add_argument(
        "-i",
        "--input",
        action="append",
        default=[],
        type=Path,
        metavar="FILE",
        help="Input data file (can be repeated, order matches function parameters)",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Output file path for result",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # version command
    subparsers.add_parser("version", help="Show version information")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the run command."""
    # Expand shorthand runtime flags
    runtimes = list(args.runtime)
    if args.std:
        runtimes.append("east-py-std")
    if args.io:
        runtimes.append("east-py-io")

    # Validate IR file exists
    if not args.ir_file.exists():
        print(f"Error: IR file not found: {args.ir_file}", file=sys.stderr)
        return 1

    # Validate input files exist
    for input_file in args.input:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            return 1

    try:
        # Load IR
        if args.verbose:
            print(f"Loading IR from {args.ir_file}...")
        ir = load_ir(args.ir_file)

        # Load platform functions from runtimes
        platform_fns = []
        for runtime in runtimes:
            if args.verbose:
                print(f"Loading runtime: {runtime}")
            fns = load_runtime(runtime)
            platform_fns.extend(fns)
            if args.verbose:
                print(f"  Loaded {len(fns)} platform functions")

        # Run the program
        if args.verbose:
            print(f"Running program with {len(args.input)} inputs...")

        run_program(
            ir=ir,
            platform_fns=platform_fns,
            input_files=args.input,
            output_file=args.output,
            verbose=args.verbose,
        )

        if args.verbose:
            print("Done.")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Execute the version command."""
    from east_py_cli import __version__ as cli_version

    print(f"east-py-cli {cli_version}")

    # Try to get east-py version
    try:
        import east

        print(f"east-py {getattr(east, '__version__', 'unknown')}")
    except ImportError:
        print("east-py: not installed")

    # Check for available runtimes
    print("\nRuntimes available:")
    for runtime in ["east-py-std", "east-py-io"]:
        try:
            fns = load_runtime(runtime)
            module_name = runtime.replace("-", "_")
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {runtime} {version} ({len(fns)} platform functions)")
        except ImportError:
            print(f"  {runtime}: not installed")
        except Exception as e:
            print(f"  {runtime}: error ({e})")

    return 0


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "version":
        sys.exit(cmd_version(args))
    else:
        parser.print_help()
        sys.exit(1)
