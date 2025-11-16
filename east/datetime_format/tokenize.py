r"""Parser/tokenizer for datetime format strings.

Converts Day.js-style format strings into structured token arrays.
The parser handles escape sequences and groups consecutive literal
characters into single tokens for efficiency.

Escaping: Use backslash (\) to escape any character. \x produces
literal x for any character. A terminating backslash is treated as
a literal backslash.

Unicode: The parser correctly handles Unicode codepoints including
surrogate pairs (but not grapheme clusters).
"""

from typing import Any

from east.types.types import VariantType

# Map token types to their format patterns
TOKEN_PATTERNS: dict[str, str] = {
    "year4": "YYYY",
    "year2": "YY",
    "month1": "M",
    "month2": "MM",
    "monthNameShort": "MMM",
    "monthNameFull": "MMMM",
    "day1": "D",
    "day2": "DD",
    "weekdayNameMin": "dd",
    "weekdayNameShort": "ddd",
    "weekdayNameFull": "dddd",
    "hour24_1": "H",
    "hour24_2": "HH",
    "hour12_1": "h",
    "hour12_2": "hh",
    "minute1": "m",
    "minute2": "mm",
    "second1": "s",
    "second2": "ss",
    "millisecond3": "SSS",
    "ampmUpper": "A",
    "ampmLower": "a",
}

# All format patterns in descending length order for greedy matching
FORMAT_PATTERNS = [
    "YYYY",
    "MMMM",
    "dddd",
    "SSS",
    "HH",
    "MM",
    "DD",
    "hh",
    "mm",
    "ss",
    "YY",
    "MMM",
    "ddd",
    "dd",
    "H",
    "M",
    "D",
    "h",
    "m",
    "s",
    "A",
    "a",
]


def tokenize_datetime_format(format_str: str) -> list[Any]:
    """Parses a datetime format string into structured tokens.

    Args:
        format_str: The format string to parse (e.g., "YYYY-MM-DD HH:mm:ss")

    Returns:
        Array of format tokens (EastVariant objects)

    Example:
        >>> tokenize_datetime_format("YYYY-MM-DD")
        # Returns:
        # [
        #   EastVariant("year4", None),
        #   EastVariant("literal", "-"),
        #   EastVariant("month2", None),
        #   EastVariant("literal", "-"),
        #   EastVariant("day2", None)
        # ]

    Example:
        >>> # Escaping
        >>> tokenize_datetime_format("\\\\YYYY-MM-DD")  # literal "YYYY-MM-DD"
        >>> tokenize_datetime_format("YYYY\\\\-MM")      # year4, literal "-", month2

    Example:
        >>> # Unicode support
        >>> tokenize_datetime_format("YYYY年MM月DD日")
        # Returns: year4, literal("年"), month2, literal("月"), day2, literal("日")
    """
    from east.types.types import NullType, StringType

    # Build runtime variant type for creating tokens
    cases = [
        ("literal", StringType),
        ("year4", NullType),
        ("year2", NullType),
        ("month1", NullType),
        ("month2", NullType),
        ("monthNameShort", NullType),
        ("monthNameFull", NullType),
        ("day1", NullType),
        ("day2", NullType),
        ("weekdayNameMin", NullType),
        ("weekdayNameShort", NullType),
        ("weekdayNameFull", NullType),
        ("hour24_1", NullType),
        ("hour24_2", NullType),
        ("hour12_1", NullType),
        ("hour12_2", NullType),
        ("minute1", NullType),
        ("minute2", NullType),
        ("second1", NullType),
        ("second2", NullType),
        ("millisecond3", NullType),
        ("ampmUpper", NullType),
        ("ampmLower", NullType),
    ]
    VariantType(cases)

    tokens: list[Any] = []

    # Convert to list of characters to handle Unicode correctly
    code_points = list(format_str)
    i = 0
    literal = ""

    def flush_literal() -> None:
        """Flushes accumulated literal characters as a single token."""
        nonlocal literal
        if literal:
            tokens.append({"type": "literal", "value": literal})
            literal = ""

    def try_match(pattern: str, token_type: str) -> bool:
        """Attempts to match a format token pattern at current position.

        Args:
            pattern: The pattern to match (e.g., "YYYY", "MM")
            token_type: The token type to emit if matched

        Returns:
            True if pattern matched and consumed, False otherwise
        """
        nonlocal i

        # Check if we have enough characters left
        if i + len(pattern) > len(code_points):
            return False

        # Check if the pattern matches
        slice_str = "".join(code_points[i : i + len(pattern)])
        if slice_str == pattern:
            flush_literal()
            tokens.append({"type": token_type, "value": None})
            i += len(pattern)
            return True

        return False

    while i < len(code_points):
        char = code_points[i]

        # Handle escaping: \x produces literal x
        if char == "\\":
            if i + 1 < len(code_points):
                # Escape next character
                literal += code_points[i + 1]
                i += 2
                continue
            # Terminating backslash - treat as literal
            literal += "\\"
            i += 1
            continue

        # Try to match format tokens (longest patterns first)
        # Year
        if try_match("YYYY", "year4") or try_match("YY", "year2"):
            continue

        # Month (MMMM before MMM before MM before M)
        if (
            try_match("MMMM", "monthNameFull")
            or try_match("MMM", "monthNameShort")
            or try_match("MM", "month2")
            or try_match("M", "month1")
        ):
            continue

        # Day of month
        if try_match("DD", "day2") or try_match("D", "day1"):
            continue

        # Day of week (dddd before ddd before dd)
        if (
            try_match("dddd", "weekdayNameFull")
            or try_match("ddd", "weekdayNameShort")
            or try_match("dd", "weekdayNameMin")
        ):
            continue

        # Hour 24h
        if try_match("HH", "hour24_2") or try_match("H", "hour24_1"):
            continue

        # Hour 12h
        if try_match("hh", "hour12_2") or try_match("h", "hour12_1"):
            continue

        # Minute
        if try_match("mm", "minute2") or try_match("m", "minute1"):
            continue

        # Second
        if try_match("ss", "second2") or try_match("s", "second1"):
            continue

        # Millisecond
        if try_match("SSS", "millisecond3"):
            continue

        # AM/PM
        if try_match("A", "ampmUpper") or try_match("a", "ampmLower"):
            continue

        # Not a format token - accumulate as literal
        literal += char
        i += 1

    # Flush any remaining literal
    flush_literal()

    return tokens


def format_tokens_to_string(tokens: list[Any], colorize: bool = False) -> str:
    """Converts a token array back to a canonical format string.

    This produces a format string with minimal escaping - only escaping
    characters when they would otherwise be parsed as format tokens.

    Args:
        tokens: Array of format tokens to stringify (EastVariant objects or dicts)
        colorize: Whether to colorize format tokens with ANSI cyan (default: False)

    Returns:
        A canonical format string that parses to the same tokens

    Remarks:
        This is useful for debugging and error messages. When users write
        format strings with accidental format codes, showing the canonical
        version helps them understand what was parsed.

        When colorize is True, format tokens (e.g., "YYYY", "MM", "DD") are
        wrapped in ANSI escape codes to display in cyan, making them visually
        distinct from literal characters in terminal output.

    Example:
        >>> # Show user what their format string was interpreted as
        >>> tokens = tokenize_datetime_format("Today is YYYY")
        >>> print(f'Parsed as: "{format_tokens_to_string(tokens)}"')
        # Output: Parsed as: "Tod\\ay i\\s YYYY"

    Example:
        >>> # Colorized output for terminal
        >>> tokens = tokenize_datetime_format("YYYY-MM-DD")
        >>> print(format_tokens_to_string(tokens, True))
        # Output: format tokens in cyan, literals in default color
    """
    # ANSI escape codes for colorization
    CYAN = "\x1b[36m"
    RESET = "\x1b[0m"

    result_parts = []

    for token in tokens:
        # Handle both dict and EastVariant formats
        token_type = token["type"]
        token_value = token.get("value") if isinstance(token, dict) else token.value

        if token_type == "literal":
            assert isinstance(
                token_value, str
            ), f"Literal token must have string value, got {type(token_value)}"
            chars = list(token_value)
            result = ""
            i = 0

            while i < len(chars):
                remaining = "".join(chars[i:])

                # Check if remaining string starts with a format pattern
                matched_pattern = None
                for pattern in FORMAT_PATTERNS:
                    if remaining.startswith(pattern):
                        matched_pattern = pattern
                        break

                if matched_pattern:
                    # Escape first character to prevent token recognition
                    result += "\\" + chars[i]
                    i += 1
                elif chars[i] == "\\":
                    # Always escape backslashes
                    result += "\\\\"
                    i += 1
                else:
                    # Safe to emit as-is
                    result += chars[i]
                    i += 1

            result_parts.append(result)
        else:
            # Emit the format pattern for this token type
            pattern = TOKEN_PATTERNS[token_type]
            if colorize:
                result_parts.append(f"{CYAN}{pattern}{RESET}")
            else:
                result_parts.append(pattern)

    return "".join(result_parts)


__all__ = ["tokenize_datetime_format", "format_tokens_to_string"]
