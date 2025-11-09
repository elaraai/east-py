"""Datetime formatting runtime implementation.

This module provides the runtime formatting logic for converting datetime objects
to strings according to parsed format token arrays.
"""

from datetime import datetime
from typing import Any

# Month names in English
MONTH_NAMES_FULL = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Abbreviated month names in English
MONTH_NAMES_SHORT = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# Full weekday names in English, starting with Sunday
WEEKDAY_NAMES_FULL = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Abbreviated weekday names in English, starting with Sunday
WEEKDAY_NAMES_SHORT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# Minimal weekday names (2 characters), starting with Sunday
WEEKDAY_NAMES_MIN = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]


def format_datetime(dt: datetime, tokens: list[Any]) -> str:
    """Formats a datetime according to an array of format tokens.

    Args:
        dt: The datetime object to format
        tokens: Array of format tokens specifying the output format (EastVariant objects or dicts)

    Returns:
        The formatted date string

    Remarks:
        This function implements the runtime formatting logic for East's datetime
        formatting.

        All dates are treated as UTC (naive datetimes with no timezone information).
        The datetime object's UTC time components are used for formatting.

    Example:
        >>> from datetime import datetime, timezone
        >>> dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=timezone.utc)
        >>> tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")
        >>> format_datetime(dt, tokens)
        "2025-01-15 14:30:45.123"
    """
    parts: list[str] = []

    for token in tokens:
        # Handle both dict and EastVariant formats
        token_type = token["type"] if isinstance(token, dict) else token.tag
        token_value = token.get("value") if isinstance(token, dict) else token.value

        # Year
        if token_type == "year4":
            part = str(dt.year)
        elif token_type == "year2":
            part = str(dt.year % 100).zfill(2)

        # Month
        elif token_type == "month1":
            part = str(dt.month)
        elif token_type == "month2":
            part = str(dt.month).zfill(2)
        elif token_type == "monthNameShort":
            part = MONTH_NAMES_SHORT[dt.month - 1]
        elif token_type == "monthNameFull":
            part = MONTH_NAMES_FULL[dt.month - 1]

        # Day of month
        elif token_type == "day1":
            part = str(dt.day)
        elif token_type == "day2":
            part = str(dt.day).zfill(2)

        # Day of week
        # Python weekday(): 0=Monday, 1=Tuesday, ..., 6=Sunday
        # JavaScript getUTCDay(): 0=Sunday, 1=Monday, ..., 6=Saturday
        # Convert Python weekday to JavaScript getUTCDay format
        elif token_type == "weekdayNameMin":
            weekday = (dt.weekday() + 1) % 7  # Convert Mon=0 to Sun=0 format
            part = WEEKDAY_NAMES_MIN[weekday]
        elif token_type == "weekdayNameShort":
            weekday = (dt.weekday() + 1) % 7
            part = WEEKDAY_NAMES_SHORT[weekday]
        elif token_type == "weekdayNameFull":
            weekday = (dt.weekday() + 1) % 7
            part = WEEKDAY_NAMES_FULL[weekday]

        # Hour (24-hour)
        elif token_type == "hour24_1":
            part = str(dt.hour)
        elif token_type == "hour24_2":
            part = str(dt.hour).zfill(2)

        # Hour (12-hour)
        elif token_type == "hour12_1":
            hour12 = dt.hour % 12 or 12
            part = str(hour12)
        elif token_type == "hour12_2":
            hour12 = dt.hour % 12 or 12
            part = str(hour12).zfill(2)

        # Minute
        elif token_type == "minute1":
            part = str(dt.minute)
        elif token_type == "minute2":
            part = str(dt.minute).zfill(2)

        # Second
        elif token_type == "second1":
            part = str(dt.second)
        elif token_type == "second2":
            part = str(dt.second).zfill(2)

        # Millisecond
        elif token_type == "millisecond3":
            # dt.microsecond is in microseconds, convert to milliseconds
            part = str(dt.microsecond // 1000).zfill(3)

        # AM/PM
        elif token_type == "ampmUpper":
            part = "AM" if dt.hour < 12 else "PM"
        elif token_type == "ampmLower":
            part = "am" if dt.hour < 12 else "pm"

        # Literal
        elif token_type == "literal":
            part = token_value

        else:
            raise ValueError(f"Unknown datetime format token type: {token_type}")

        parts.append(part)

    return "".join(parts)


__all__ = ["format_datetime"]
