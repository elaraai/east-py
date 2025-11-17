"""Type system builtin functions.

NOTE: Print and Parse are registered in string.py with factory patterns.
This module is kept for backwards compatibility but registers to the same names.
"""

from collections.abc import Callable
from typing import Any

from east.builtins.registry import register_builtin
from east.types.types import EastType


def string_print_east_for(value_type: EastType) -> Callable[[Any], str]:
    """Factory for printing value to East text format.

    Args:
        value_type: Type of the value

    Returns:
        Function that prints values of this type to East text format
    """
    from east.serialization.east_printer import print_east

    def string_print_east(value: Any) -> str:
        return print_east(value, value_type)

    return string_print_east


def string_parse_east_for(target_type: EastType) -> Callable[[str], Any]:
    """Factory for parsing East text format to value.

    Args:
        target_type: Expected type of the value

    Returns:
        Function that parses East text format to values of this type
    """
    from east.serialization.east_parser import parse_east

    def string_parse_east(text: str) -> Any:
        # parse_east returns the value directly or raises ParseError
        return parse_east(target_type, text)

    return string_parse_east


# Register all type system builtins as factories
# These override the ones registered in string.py
register_builtin("Print", string_print_east_for)
register_builtin("Parse", string_parse_east_for)


__all__ = [
    "string_print_east_for",
    "string_parse_east_for",
]
