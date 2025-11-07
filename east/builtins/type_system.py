"""Type system builtin functions."""

from typing import Any

from east.builtins.registry import register_builtin
from east.types.type_system import EastType, type_of


def builtin_type_of(value: Any) -> EastType:
    """Get the East type of a value.

    Args:
        value: Any value

    Returns:
        EastType representing the type of the value
    """
    return type_of(value)


def string_print_east(value: Any, value_type: EastType) -> str:
    """Print value to East text format.

    Args:
        value: Value to print
        value_type: Type of the value

    Returns:
        East text format representation
    """
    from east.serialization.east_printer import print_east

    return print_east(value, value_type)


def string_parse_east(text: str, target_type: EastType) -> Any:
    """Parse East text format to value.

    Args:
        text: East text format string
        target_type: Expected type of the value

    Returns:
        Parsed value
    """
    from east.serialization.east_parser import parse_east

    return parse_east(target_type, text)


# Register all type system builtins
register_builtin("TypeOf", builtin_type_of)
register_builtin("Print", string_print_east)  # Renamed from StringPrintEast
register_builtin("Parse", string_parse_east)  # Renamed from StringParseEast


__all__ = [
    "builtin_type_of",
    "string_print_east",
    "string_parse_east",
]
