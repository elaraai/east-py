"""String builtin functions."""

from east.builtins.registry import register_builtin
from east.types.containers import EastArray
from east.types.type_system import StringType


def string_concat(a: str, b: str) -> str:
    """Concatenate two strings.

    Args:
        a: First string
        b: Second string

    Returns:
        a + b
    """
    return a + b


def string_length(s: str) -> int:
    """Get length of string.

    Args:
        s: String

    Returns:
        Number of characters in s
    """
    return len(s)


def string_get(s: str, index: int) -> str:
    """Get character at index.

    Args:
        s: String
        index: Character index (0-based)

    Returns:
        Character at index

    Raises:
        IndexError: If index out of bounds
    """
    return s[index]


def string_slice(s: str, start: int, end: int) -> str:
    """Get substring slice.

    Args:
        s: String
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        s[start:end]
    """
    return s[start:end]


def string_index_of(s: str, substring: str) -> int:
    """Find first occurrence of substring.

    Args:
        s: String to search in
        substring: Substring to find

    Returns:
        Index of first occurrence, or -1 if not found
    """
    return s.find(substring)


def string_last_index_of(s: str, substring: str) -> int:
    """Find last occurrence of substring.

    Args:
        s: String to search in
        substring: Substring to find

    Returns:
        Index of last occurrence, or -1 if not found
    """
    return s.rfind(substring)


def string_split(s: str, delimiter: str) -> EastArray:
    """Split string by delimiter.

    Args:
        s: String to split
        delimiter: Delimiter string

    Returns:
        Array of substrings
    """
    parts = s.split(delimiter)
    return EastArray(StringType, parts)


def string_join(arr: EastArray, delimiter: str) -> str:
    """Join array of strings with delimiter.

    Args:
        arr: Array of strings
        delimiter: Delimiter to join with

    Returns:
        Joined string
    """
    return delimiter.join(str(item) for item in arr)


def string_trim(s: str) -> str:
    """Trim whitespace from both ends.

    Args:
        s: String

    Returns:
        String with whitespace removed from both ends
    """
    return s.strip()


def string_trim_start(s: str) -> str:
    """Trim whitespace from start.

    Args:
        s: String

    Returns:
        String with whitespace removed from start
    """
    return s.lstrip()


def string_trim_end(s: str) -> str:
    """Trim whitespace from end.

    Args:
        s: String

    Returns:
        String with whitespace removed from end
    """
    return s.rstrip()


def string_to_lower_case(s: str) -> str:
    """Convert string to lowercase.

    Args:
        s: String

    Returns:
        Lowercase version of s
    """
    return s.lower()


def string_to_upper_case(s: str) -> str:
    """Convert string to uppercase.

    Args:
        s: String

    Returns:
        Uppercase version of s
    """
    return s.upper()


def string_replace(s: str, old: str, new: str) -> str:
    """Replace all occurrences of substring.

    Args:
        s: String
        old: Substring to replace
        new: Replacement string

    Returns:
        String with all occurrences replaced
    """
    return s.replace(old, new)


def string_starts_with(s: str, prefix: str) -> bool:
    """Check if string starts with prefix.

    Args:
        s: String
        prefix: Prefix to check

    Returns:
        True if s starts with prefix
    """
    return s.startswith(prefix)


def string_ends_with(s: str, suffix: str) -> bool:
    """Check if string ends with suffix.

    Args:
        s: String
        suffix: Suffix to check

    Returns:
        True if s ends with suffix
    """
    return s.endswith(suffix)


def string_contains(s: str, substring: str) -> bool:
    """Check if string contains substring.

    Args:
        s: String
        substring: Substring to check

    Returns:
        True if s contains substring
    """
    return substring in s


def string_to_integer(s: str) -> int:
    """Parse string as integer.

    Args:
        s: String to parse

    Returns:
        Parsed integer

    Raises:
        ValueError: If string is not a valid integer
    """
    return int(s)


def string_to_float(s: str) -> float:
    """Parse string as float.

    Args:
        s: String to parse

    Returns:
        Parsed float

    Raises:
        ValueError: If string is not a valid float
    """
    # Handle special values
    if s == "NaN":
        return float("nan")
    if s == "Infinity":
        return float("inf")
    if s == "-Infinity":
        return float("-inf")
    return float(s)


# Register all string builtins
register_builtin("StringConcat", string_concat)
register_builtin("StringLength", string_length)
register_builtin("StringGet", string_get)
register_builtin("StringSlice", string_slice)
register_builtin("StringIndexOf", string_index_of)
register_builtin("StringLastIndexOf", string_last_index_of)
register_builtin("StringSplit", string_split)
register_builtin("StringJoin", string_join)
register_builtin("StringTrim", string_trim)
register_builtin("StringTrimStart", string_trim_start)
register_builtin("StringTrimEnd", string_trim_end)
register_builtin("StringLowerCase", string_to_lower_case)  # Renamed from StringToLowerCase
register_builtin("StringUpperCase", string_to_upper_case)  # Renamed from StringToUpperCase
register_builtin("StringReplace", string_replace)
register_builtin("StringStartsWith", string_starts_with)
register_builtin("StringEndsWith", string_ends_with)
register_builtin("StringContains", string_contains)
register_builtin("StringToInteger", string_to_integer)
register_builtin("StringToFloat", string_to_float)


__all__ = [
    "string_concat",
    "string_length",
    "string_get",
    "string_slice",
    "string_index_of",
    "string_last_index_of",
    "string_split",
    "string_join",
    "string_trim",
    "string_trim_start",
    "string_trim_end",
    "string_to_lower_case",
    "string_to_upper_case",
    "string_replace",
    "string_starts_with",
    "string_ends_with",
    "string_contains",
    "string_to_integer",
    "string_to_float",
]
