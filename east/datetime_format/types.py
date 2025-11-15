"""Datetime format token types.

This module defines the structured representation of datetime format strings.
Format strings are parsed once and serialized as structured tokens, ensuring
consistent formatting behavior across all East backends.
"""

from east.types.types import NullType, StringType, VariantType

# Structured representation of a datetime format string.
#
# Format strings are parsed into an array of tokens at AST construction time.
# Each token represents either a datetime component (year, month, day, etc.)
# or a literal string to be included verbatim in the output.
#
# This structured representation serves as East's "narrow waist" for datetime
# formatting - the TypeScript SDK parses format strings once, and all backends
# implement formatting from the same token structure, guaranteeing identical
# behavior across JavaScript, Julia, and other future backends.
#
# Formats do not support timezones or locales.
#
# Example:
#     Format string "YYYY-MM-DD" parses to:
#     [
#         {"type": "year4", "value": None},
#         {"type": "literal", "value": "-"},
#         {"type": "month2", "value": None},
#         {"type": "literal", "value": "-"},
#         {"type": "day2", "value": None}
#     ]
DateTimeFormatTokenType = VariantType(
    [
        # Literal text to include verbatim in formatted output.
        # Any characters in the format string that are not recognized as format
        # codes are treated as literals.
        ("literal", StringType),
        # Year
        ("year4", NullType),  # Four-digit year (e.g., "2025") - Format: YYYY
        ("year2", NullType),  # Two-digit year (e.g., "25") - Format: YY
        # Month
        ("month1", NullType),  # Month as 1-12 without zero-padding (e.g., "1", "12") - Format: M
        ("month2", NullType),  # Month as 01-12 with zero-padding (e.g., "01", "12") - Format: MM
        ("monthNameShort", NullType),  # Short month name (e.g., "Jan", "Feb") - Format: MMM
        ("monthNameFull", NullType),  # Full month name (e.g., "January", "February") - Format: MMMM
        # Day of month
        (
            "day1",
            NullType,
        ),  # Day of month as 1-31 without zero-padding (e.g., "1", "31") - Format: D
        (
            "day2",
            NullType,
        ),  # Day of month as 01-31 with zero-padding (e.g., "01", "31") - Format: DD
        # Day of week
        ("weekdayNameMin", NullType),  # Minimal weekday name (e.g., "Su", "Mo") - Format: dd
        ("weekdayNameShort", NullType),  # Short weekday name (e.g., "Sun", "Mon") - Format: ddd
        (
            "weekdayNameFull",
            NullType,
        ),  # Full weekday name (e.g., "Sunday", "Monday") - Format: dddd
        # Hour (24-hour)
        ("hour24_1", NullType),  # Hour 0-23 without zero-padding (e.g., "0", "23") - Format: H
        ("hour24_2", NullType),  # Hour 00-23 with zero-padding (e.g., "00", "23") - Format: HH
        # Hour (12-hour)
        ("hour12_1", NullType),  # Hour 1-12 without zero-padding (e.g., "1", "12") - Format: h
        ("hour12_2", NullType),  # Hour 01-12 with zero-padding (e.g., "01", "12") - Format: hh
        # Minute
        ("minute1", NullType),  # Minute 0-59 without zero-padding (e.g., "0", "59") - Format: m
        ("minute2", NullType),  # Minute 00-59 with zero-padding (e.g., "00", "59") - Format: mm
        # Second
        ("second1", NullType),  # Second 0-59 without zero-padding (e.g., "0", "59") - Format: s
        ("second2", NullType),  # Second 00-59 with zero-padding (e.g., "00", "59") - Format: ss
        # Millisecond
        (
            "millisecond3",
            NullType,
        ),  # Millisecond 000-999 with zero-padding (e.g., "000", "999") - Format: SSS
        # AM/PM
        ("ampmUpper", NullType),  # Uppercase AM/PM (e.g., "AM", "PM") - Format: A
        ("ampmLower", NullType),  # Lowercase am/pm (e.g., "am", "pm") - Format: a
    ]
)

__all__ = ["DateTimeFormatTokenType"]
