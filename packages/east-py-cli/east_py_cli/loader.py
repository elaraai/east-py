"""IR and value loading utilities."""

import importlib
from pathlib import Path
from typing import Any

from east.runtime.platform import PlatformFunction
from east.types.type_of_type import IRType
from east.types.types import EastType


def detect_format(file_path: Path) -> str:
    """Detect file format from extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'beast2', 'east', or 'json'

    Raises:
        ValueError: If extension is not recognized
    """
    ext = file_path.suffix.lower()
    if ext in (".beast2", ".beast"):
        return "beast2"
    elif ext == ".east":
        return "east"
    elif ext == ".json":
        return "json"
    else:
        raise ValueError(
            f"Unknown file extension: {ext}. " f"Supported: .beast2, .beast, .east, .json"
        )


def load_ir(file_path: Path) -> Any:
    """Load IR from a file.

    Auto-detects format from file extension.

    Args:
        file_path: Path to IR file

    Returns:
        Parsed IR value

    Raises:
        ValueError: If format is not recognized or parsing fails
    """
    fmt = detect_format(file_path)

    if fmt == "beast2":
        from east.serialization.beast2 import decode_beast2_for

        with open(file_path, "rb") as f:
            data = f.read()
        decoder = decode_beast2_for(IRType)
        return decoder(data)

    elif fmt == "east":
        from east.serialization.east_parser import parse_east

        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return parse_east(IRType, text)

    elif fmt == "json":
        from east.serialization.json import decode_json_for

        with open(file_path, "rb") as f:
            data = f.read()
        decoder = decode_json_for(IRType)
        return decoder(data)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def load_value(file_path: Path, value_type: EastType) -> Any:
    """Load a value from a file with type-directed parsing.

    Args:
        file_path: Path to data file
        value_type: Expected East type for parsing

    Returns:
        Parsed value

    Raises:
        ValueError: If format is not recognized or parsing fails
    """
    fmt = detect_format(file_path)

    if fmt == "beast2":
        from east.serialization.beast2 import decode_beast2_for

        with open(file_path, "rb") as f:
            data = f.read()
        decoder = decode_beast2_for(value_type)
        return decoder(data)

    elif fmt == "east":
        from east.serialization.east_parser import parse_east

        with open(file_path, encoding="utf-8") as f:
            text = f.read()
        return parse_east(value_type, text)

    elif fmt == "json":
        from east.serialization.json import decode_json_for

        with open(file_path, "rb") as f:
            data = f.read()
        decoder = decode_json_for(value_type)
        return decoder(data)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_value(file_path: Path, value: Any, value_type: EastType) -> None:
    """Save a value to a file with type-directed serialization.

    Args:
        file_path: Path to write to
        value: Value to serialize
        value_type: East type for serialization

    Raises:
        ValueError: If format is not recognized
    """
    fmt = detect_format(file_path)

    if fmt == "beast2":
        from east.serialization.beast2 import encode_beast2_for

        encoder = encode_beast2_for(value_type)
        data = encoder(value)
        with open(file_path, "wb") as f:
            f.write(data)

    elif fmt == "east":
        from east.serialization.east_printer import print_east

        text = print_east(value_type, value)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    elif fmt == "json":
        from east.serialization.json import encode_json_for

        encoder = encode_json_for(value_type)
        data = encoder(value)
        with open(file_path, "wb") as f:
            f.write(data)

    else:
        raise ValueError(f"Unknown format: {fmt}")


def load_runtime(package_name: str) -> list[PlatformFunction]:
    """Load platform functions from a runtime package.

    Args:
        package_name: Package name (e.g., 'east-py-std')

    Returns:
        List of platform functions

    Raises:
        ImportError: If package is not installed
        ValueError: If package doesn't export platform functions
    """
    # Convert package name to module name (east-py-std -> east_py_std)
    module_name = package_name.replace("-", "_")

    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Runtime package '{package_name}' not found. "
            f"Install it with: uv add {package_name}"
        ) from e

    # Try known attribute names
    for attr in ["platform", "python_platform", "python_io_platform"]:
        if hasattr(mod, attr):
            fns = getattr(mod, attr)
            if isinstance(fns, list):
                return fns

    raise ValueError(
        f"Runtime package '{package_name}' has no 'platform' export. "
        f"The package must export a list of PlatformFunction objects."
    )
