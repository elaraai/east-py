"""String builtin functions."""

import json
from collections.abc import Callable

from east.builtins.registry import register_builtin
from east.types.types import EastType, StringType
from east.types.values import EastArray, EastValue


def string_concat(a: str, b: str) -> str:
    """Concatenate two strings.

    Args:
        a: First string
        b: Second string

    Returns:
        a + b
    """
    return a + b


def string_repeat(s: str, count: int) -> str:
    """Repeat string n times.

    Args:
        s: String to repeat
        count: Number of repetitions

    Returns:
        s repeated count times
    """
    return s * count


def string_length(s: str) -> int:
    """Get length of string.

    Args:
        s: String

    Returns:
        Number of characters in s
    """
    return len(s)


def string_substring(s: str, start: int, end: int) -> str:
    """Get substring between two indices (JavaScript semantics).

    Args:
        s: String
        start: Start index (0-based, inclusive)
        end: End index (0-based, exclusive)

    Returns:
        Substring from start to end (exclusive)

    Note:
        Matches JavaScript substring behavior:
        - Negative values are treated as 0
        - Values > length are clamped to length
        - If start > end (negative length), returns empty string
    """
    length = len(s)
    # Clamp negative values to 0, values > length to length
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    # If start > end, return empty string (negative length becomes 0)
    if start > end:
        return ""
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


def string_split(s: str, delimiter: str) -> EastArray:
    """Split string by delimiter (JavaScript semantics).

    Args:
        s: String to split
        delimiter: Delimiter string

    Returns:
        Array of substrings

    Note:
        When delimiter is empty, splits into individual characters (matching JavaScript behavior)
    """
    if delimiter == "":
        # Empty delimiter: split into individual characters
        # Special case: empty string returns array with one empty string
        if s == "":
            parts = [""]
        else:
            parts = list(s)
    else:
        parts = s.split(delimiter)
    return EastArray(StringType, parts)


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


def regex_contains(text: str, pattern: str, flags: str) -> bool:
    """Check if regex pattern matches text.

    Args:
        text: Text to search
        pattern: Regex pattern
        flags: Regex flags (e.g., "i" for case-insensitive, "m" for multiline)

    Returns:
        True if pattern matches text

    Example:
        >>> regex_contains("Hello World", r"world", "i")
        True
        >>> regex_contains("Hello World", r"world", "")
        False
    """
    import re

    # Parse flags
    re_flags = 0
    if "i" in flags:
        re_flags |= re.IGNORECASE
    if "m" in flags:
        re_flags |= re.MULTILINE
    if "s" in flags:
        re_flags |= re.DOTALL

    return re.search(pattern, text, re_flags) is not None


def regex_index_of(text: str, pattern: str, flags: str) -> int:
    """Find first regex match position.

    Args:
        text: Text to search
        pattern: Regex pattern
        flags: Regex flags

    Returns:
        Index of first match, or -1 if not found

    Example:
        >>> regex_index_of("Hello World", r"\\w+", "")
        0
        >>> regex_index_of("Hello World", r"World", "")
        6
        >>> regex_index_of("Hello World", r"xyz", "")
        -1
    """
    import re

    # Parse flags
    re_flags = 0
    if "i" in flags:
        re_flags |= re.IGNORECASE
    if "m" in flags:
        re_flags |= re.MULTILINE
    if "s" in flags:
        re_flags |= re.DOTALL

    match = re.search(pattern, text, re_flags)
    return match.start() if match else -1


def regex_replace(text: str, pattern: str, flags: str, replacement: str) -> str:
    r"""Replace all regex matches (JavaScript replaceAll semantics).

    Args:
        text: Text to search
        pattern: Regex pattern
        flags: Regex flags (i=case insensitive, m=multiline, s=dotall)
        replacement: Replacement string (supports $1, $2, $&, $`, $' syntax)

    Returns:
        Text with all matches replaced

    Note:
        Always replaces all matches (replaceAll semantics).
        The 'g' flag is automatically added if not present to match TypeScript behavior.
        Supports JavaScript replacement patterns: $1-$9, $&, $`, $', $<name>.

    Example:
        >>> regex_replace("hello123world456", r"\d+", "g", "X")
        'helloXworldX'
        >>> regex_replace("hello world", r"(\w+) (\w+)", "", "$2 $1")
        'world hello'
    """
    import re

    # Ensure global flag is set for replaceAll semantics (matching TypeScript)
    global_flags = flags if "g" in flags else flags + "g"

    # Parse flags
    re_flags = 0
    if "i" in global_flags:
        re_flags |= re.IGNORECASE
    if "m" in global_flags:
        re_flags |= re.MULTILINE
    if "s" in global_flags:
        re_flags |= re.DOTALL

    # Convert JavaScript pattern syntax to Python syntax
    # (?<name>...) -> (?P<name>...) (named groups)
    python_pattern = re.sub(r"\(\?<(\w+)>", r"(?P<\1>", pattern)

    # Check if replacement uses $` or $' which require special handling
    needs_custom_replacement = "$`" in replacement or "$'" in replacement

    if needs_custom_replacement:
        # Use a custom replacement function to handle $` and $'
        # Process all $-sequences in a single pass to avoid interference
        def replacer(match):
            def replace_dollar_sequence(m):
                seq = m.group(0)
                if seq == "$`":
                    return text[: match.start()]
                if seq == "$'":
                    return text[match.end() :]
                if seq == "$&":
                    return match.group(0)
                if m.group(1):  # $<name> - named group
                    try:
                        return match.group(m.group(1))
                    except (IndexError, KeyError):
                        return seq
                elif m.group(2):  # $1, $2, ... - numbered group
                    try:
                        return match.group(int(m.group(2)))
                    except IndexError:
                        return seq
                elif seq == "$$":  # Literal $
                    return "$"
                return seq

            # Match all JavaScript replacement patterns in one pass
            return re.sub(
                r"\$`|\$\'|\$&|\$<(\w+)>|\$(\d+)|\$\$", replace_dollar_sequence, replacement
            )

        return re.sub(python_pattern, replacer, text, count=0, flags=re_flags)
    # Use simple string replacement (faster path when $` and $' not needed)
    python_replacement = replacement

    # Replace $$ with placeholder first (to avoid interference with other $ replacements)
    python_replacement = python_replacement.replace("$$", "\x00")

    # Replace $<name> with \g<name> (named groups)
    python_replacement = re.sub(r"\$<(\w+)>", r"\\g<\1>", python_replacement)

    # Replace $& with \g<0> (entire match)
    python_replacement = python_replacement.replace("$&", r"\g<0>")

    # Replace $1, $2, ... with \1, \2, ...
    python_replacement = re.sub(r"\$(\d+)", r"\\\1", python_replacement)

    # Replace placeholder back to literal $
    python_replacement = python_replacement.replace("\x00", "$")

    return re.sub(python_pattern, python_replacement, text, count=0, flags=re_flags)


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


def string_print_json_for(T: EastType) -> Callable[[EastValue], str]:
    """Factory for converting East value to JSON string.

    Args:
        T: EastType of the value

    Returns:
        Function that converts values of this type to JSON strings
    """
    from east.serialization.json import to_json_for

    encoder = to_json_for(T)

    def string_print_json(value: EastValue) -> str:
        json_value = encoder(value)
        return json.dumps(json_value, separators=(",", ":"), ensure_ascii=False)

    return string_print_json


def string_parse_json_for(T: EastType) -> Callable[[str], EastValue]:
    """Factory for parsing JSON string to East value.

    Args:
        T: Expected EastType

    Returns:
        Function that parses JSON strings to values of this type
    """
    from east.serialization.json import from_json_for

    decoder = from_json_for(T)

    def string_parse_json(s: str) -> EastValue:
        parsed = json.loads(s)
        return decoder(parsed)

    return string_parse_json


def print_east_for(T: EastType) -> Callable[[EastValue], str]:
    """Factory for printing East value to East text format.

    Args:
        T: EastType of the value

    Returns:
        Function that prints values of this type to East text format
    """
    from east.serialization.east_printer import print_east as printer

    def print_east(value: EastValue) -> str:
        return printer(value, T)

    return print_east


def parse_east_for(T: EastType) -> Callable[[str], EastValue]:
    """Factory for parsing East text format to value.

    Args:
        T: Expected EastType

    Returns:
        Function that parses East text format to values of this type
    """
    from east.serialization.east_parser import parse_east as parser

    def parse_east(s: str) -> EastValue:
        # parser returns the value directly or raises ParseError
        return parser(T, s)

    return parse_east


# Register all string builtins as factories
register_builtin("StringConcat", lambda: string_concat)
register_builtin("StringRepeat", lambda: string_repeat)
register_builtin("StringLength", lambda: string_length)
register_builtin("StringSubstring", lambda: string_substring)
register_builtin("StringIndexOf", lambda: string_index_of)
register_builtin("StringSplit", lambda: string_split)
register_builtin("StringTrim", lambda: string_trim)
register_builtin("StringTrimStart", lambda: string_trim_start)
register_builtin("StringTrimEnd", lambda: string_trim_end)
register_builtin("StringLowerCase", lambda: string_to_lower_case)  # Renamed from StringToLowerCase
register_builtin("StringUpperCase", lambda: string_to_upper_case)  # Renamed from StringToUpperCase
register_builtin("StringReplace", lambda: string_replace)
register_builtin("RegexContains", lambda: regex_contains)
register_builtin("RegexIndexOf", lambda: regex_index_of)
register_builtin("RegexReplace", lambda: regex_replace)
register_builtin("StringStartsWith", lambda: string_starts_with)
register_builtin("StringEndsWith", lambda: string_ends_with)
register_builtin("StringContains", lambda: string_contains)


def string_print_error(message: str, stack: EastArray) -> str:
    """Format an error message with stack trace.

    Args:
        message: Error message
        stack: Stack trace (array of structs with filename, line, column fields)

    Returns:
        Formatted error string with "Error: " prefix and stack trace
    """
    # Format: "Error: {message}\n    [{index}] {filename} {line}:{column}"
    lines = [f"Error: {message}"]
    for i, frame in enumerate(stack):
        filename = frame.filename
        line = frame.line
        column = frame.column
        lines.append(f"    [{i}] {filename} {line}:{column}")
    return "\n".join(lines)


register_builtin("Print", print_east_for)
register_builtin("Parse", parse_east_for)
register_builtin("StringPrintJSON", string_print_json_for)
register_builtin("StringParseJSON", string_parse_json_for)
register_builtin("StringPrintError", lambda: string_print_error)


__all__ = [
    "string_concat",
    "string_repeat",
    "string_length",
    "string_substring",
    "string_index_of",
    "string_split",
    "string_trim",
    "string_trim_start",
    "string_trim_end",
    "string_to_lower_case",
    "string_to_upper_case",
    "string_replace",
    "regex_contains",
    "regex_index_of",
    "regex_replace",
    "string_starts_with",
    "string_ends_with",
    "string_contains",
    "print_east_for",
    "parse_east_for",
    "string_print_json_for",
    "string_parse_json_for",
    "string_print_error",
]
