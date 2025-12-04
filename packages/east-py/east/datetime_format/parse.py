"""Datetime parsing runtime implementation.

This module provides the runtime parsing logic for converting strings
to datetime objects according to parsed format token arrays. This is called by the
DateTimeParseFormat builtin.
"""

import re
from datetime import UTC, datetime
from typing import Any, TypedDict


class ParseSuccess(TypedDict):
    """Successful parse result."""

    success: bool  # Always True
    value: datetime


class ParseError(TypedDict):
    """Failed parse result."""

    success: bool  # Always False
    error: str
    position: int


DateTimeParseResult = ParseSuccess | ParseError


# Month names in English for parsing
MONTH_NAMES_FULL = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

MONTH_NAMES_SHORT = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

# Weekday names in English for parsing (currently ignored during parsing)
WEEKDAY_NAMES_FULL = ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]

WEEKDAY_NAMES_SHORT = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]

WEEKDAY_NAMES_MIN = ["su", "mo", "tu", "we", "th", "fr", "sa"]


def parse_datetime_formatted(input_str: str, tokens: list[Any]) -> DateTimeParseResult:
    """Parses a datetime string according to format tokens.

    Args:
        input_str: The string to parse
        tokens: Array of format tokens specifying the expected format

    Returns:
        Parse result containing either the datetime or an error with position

    Remarks:
        All dates are treated as UTC (naive datetimes with no timezone information).
        The parsed components are used to construct a datetime using timezone.utc.

        Weekday tokens (dd, ddd, dddd) are currently ignored during parsing - they
        are consumed from the input but not validated against the actual weekday.

    Example:
        >>> tokens = tokenize_datetime_format("YYYY-MM-DD")
        >>> result = parse_datetime_formatted("2025-01-15", tokens)
        >>> if result["success"]:
        ...     print(result["value"])  # datetime object for 2025-01-15T00:00:00.000Z
        ... else:
        ...     print(f"Parse error at position {result['position']}: {result['error']}")
    """
    position = 0

    # Datetime components - will validate redundancy and check for gaps
    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    second: int | None = None
    millisecond: int | None = None
    is_pm: bool | None = None
    hour12: int | None = None  # Track 12-hour format separately
    parsed_weekday: int | None = None  # Track parsed weekday for validation

    for token in tokens:
        if position > len(input_str):
            return {"success": False, "error": "Unexpected end of input", "position": position}

        # Handle both dict and EastVariant formats
        token_type = token["type"]
        token_value = (
            token.get("value") if isinstance(token, dict) else getattr(token, "value", None)
        )

        if token_type == "year4":
            match = input_str[position : position + 4]
            if len(match) < 4 or not re.match(r"^\d{4}$", match):
                return {"success": False, "error": "Expected 4-digit year", "position": position}
            parsed_year = int(match)
            if year is not None and year != parsed_year:
                return {
                    "success": False,
                    "error": f"Year specified multiple times with different values: {year} and {parsed_year}",
                    "position": position,
                }
            year = parsed_year
            position += 4

        elif token_type == "year2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {"success": False, "error": "Expected 2-digit year", "position": position}
            yy = int(match)
            # 2-digit year: 00-99 -> 2000-2099 (simple heuristic)
            parsed_year = 2000 + yy
            if year is not None and year != parsed_year:
                return {
                    "success": False,
                    "error": f"Year specified multiple times with different values: {year} and {parsed_year}",
                    "position": position,
                }
            year = parsed_year
            position += 2

        elif token_type == "month2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit month (01-12)",
                    "position": position,
                }
            parsed_month = int(match)
            if parsed_month < 1 or parsed_month > 12:
                return {
                    "success": False,
                    "error": f"Month out of range (got {parsed_month}, expected 01-12)",
                    "position": position,
                }
            if month is not None and month != parsed_month:
                return {
                    "success": False,
                    "error": f"Month specified multiple times with different values: {month} and {parsed_month}",
                    "position": position,
                }
            month = parsed_month
            position += 2

        elif token_type == "month1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_month = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit month",
                        "position": position,
                    }
                parsed_month = int(match)
                position += 1

            if parsed_month < 1 or parsed_month > 12:
                return {
                    "success": False,
                    "error": f"Month out of range (got {parsed_month}, expected 1-12)",
                    "position": position - len(match),
                }
            if month is not None and month != parsed_month:
                return {
                    "success": False,
                    "error": f"Month specified multiple times with different values: {month} and {parsed_month}",
                    "position": position - len(match),
                }
            month = parsed_month

        elif token_type == "monthNameFull":
            matched = False
            parsed_month = None
            for i, name in enumerate(MONTH_NAMES_FULL):
                slice_str = input_str[position : position + len(name)].lower()
                if slice_str == name:
                    parsed_month = i + 1
                    position += len(name)
                    matched = True
                    break

            if not matched:
                return {
                    "success": False,
                    "error": 'Expected full month name (e.g., "January")',
                    "position": position,
                }
            if month is not None and month != parsed_month:
                return {
                    "success": False,
                    "error": f"Month specified multiple times with different values: {month} and {parsed_month}",
                    "position": position,
                }
            month = parsed_month

        elif token_type == "monthNameShort":
            matched = False
            parsed_month = None
            for i, name in enumerate(MONTH_NAMES_SHORT):
                slice_str = input_str[position : position + len(name)].lower()
                if slice_str == name:
                    parsed_month = i + 1
                    position += len(name)
                    matched = True
                    break

            if not matched:
                return {
                    "success": False,
                    "error": 'Expected short month name (e.g., "Jan")',
                    "position": position,
                }
            if month is not None and month != parsed_month:
                return {
                    "success": False,
                    "error": f"Month specified multiple times with different values: {month} and {parsed_month}",
                    "position": position,
                }
            month = parsed_month

        elif token_type == "day2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit day (01-31)",
                    "position": position,
                }
            parsed_day = int(match)
            if parsed_day < 1 or parsed_day > 31:
                return {
                    "success": False,
                    "error": f"Day out of range (got {parsed_day}, expected 01-31)",
                    "position": position,
                }
            if day is not None and day != parsed_day:
                return {
                    "success": False,
                    "error": f"Day specified multiple times with different values: {day} and {parsed_day}",
                    "position": position,
                }
            day = parsed_day
            position += 2

        elif token_type == "day1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_day = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit day",
                        "position": position,
                    }
                parsed_day = int(match)
                position += 1

            if parsed_day < 1 or parsed_day > 31:
                return {
                    "success": False,
                    "error": f"Day out of range (got {parsed_day}, expected 1-31)",
                    "position": position - len(match),
                }
            if day is not None and day != parsed_day:
                return {
                    "success": False,
                    "error": f"Day specified multiple times with different values: {day} and {parsed_day}",
                    "position": position - len(match),
                }
            day = parsed_day

        # Weekday parsing - store for validation after Date construction
        elif token_type == "weekdayNameFull":
            matched = False
            for i, name in enumerate(WEEKDAY_NAMES_FULL):
                slice_str = input_str[position : position + len(name)].lower()
                if slice_str == name:
                    # Map to JavaScript getDay() values: 0=Sunday, 1=Monday, ..., 6=Saturday
                    weekday_value = i
                    if parsed_weekday is not None and parsed_weekday != weekday_value:
                        return {
                            "success": False,
                            "error": "Weekday specified multiple times with different values",
                            "position": position,
                        }
                    parsed_weekday = weekday_value
                    position += len(name)
                    matched = True
                    break

            if not matched:
                return {
                    "success": False,
                    "error": 'Expected full weekday name (e.g., "Monday")',
                    "position": position,
                }

        elif token_type == "weekdayNameShort":
            matched = False
            for i, name in enumerate(WEEKDAY_NAMES_SHORT):
                slice_str = input_str[position : position + len(name)].lower()
                if slice_str == name:
                    # Map to JavaScript getDay() values: 0=Sunday, 1=Monday, ..., 6=Saturday
                    weekday_value = i
                    if parsed_weekday is not None and parsed_weekday != weekday_value:
                        return {
                            "success": False,
                            "error": "Weekday specified multiple times with different values",
                            "position": position,
                        }
                    parsed_weekday = weekday_value
                    position += len(name)
                    matched = True
                    break

            if not matched:
                return {
                    "success": False,
                    "error": 'Expected short weekday name (e.g., "Mon")',
                    "position": position,
                }

        elif token_type == "weekdayNameMin":
            matched = False
            for i, name in enumerate(WEEKDAY_NAMES_MIN):
                slice_str = input_str[position : position + len(name)].lower()
                if slice_str == name:
                    # Map to JavaScript getDay() values: 0=Sunday, 1=Monday, ..., 6=Saturday
                    weekday_value = i
                    if parsed_weekday is not None and parsed_weekday != weekday_value:
                        return {
                            "success": False,
                            "error": "Weekday specified multiple times with different values",
                            "position": position,
                        }
                    parsed_weekday = weekday_value
                    position += len(name)
                    matched = True
                    break

            if not matched:
                return {
                    "success": False,
                    "error": 'Expected minimal weekday name (e.g., "Mo")',
                    "position": position,
                }

        elif token_type == "hour24_2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit hour (00-23)",
                    "position": position,
                }
            parsed_hour = int(match)
            if parsed_hour > 23:
                return {
                    "success": False,
                    "error": f"Hour out of range (got {parsed_hour}, expected 00-23)",
                    "position": position,
                }
            if hour is not None and hour != parsed_hour:
                return {
                    "success": False,
                    "error": f"Hour (24-hour) specified multiple times with different values: {hour} and {parsed_hour}",
                    "position": position,
                }
            hour = parsed_hour
            position += 2

        elif token_type == "hour24_1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_hour = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit hour",
                        "position": position,
                    }
                parsed_hour = int(match)
                position += 1

            if parsed_hour > 23:
                return {
                    "success": False,
                    "error": f"Hour out of range (got {parsed_hour}, expected 0-23)",
                    "position": position - len(match),
                }
            if hour is not None and hour != parsed_hour:
                return {
                    "success": False,
                    "error": f"Hour (24-hour) specified multiple times with different values: {hour} and {parsed_hour}",
                    "position": position - len(match),
                }
            hour = parsed_hour

        elif token_type == "hour12_2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit hour (01-12)",
                    "position": position,
                }
            parsed_hour12 = int(match)
            if parsed_hour12 < 1 or parsed_hour12 > 12:
                return {
                    "success": False,
                    "error": f"Hour out of range (got {parsed_hour12}, expected 01-12)",
                    "position": position,
                }
            if hour12 is not None and hour12 != parsed_hour12:
                return {
                    "success": False,
                    "error": f"Hour (12-hour) specified multiple times with different values: {hour12} and {parsed_hour12}",
                    "position": position,
                }
            hour12 = parsed_hour12
            position += 2

        elif token_type == "hour12_1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_hour12 = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit hour",
                        "position": position,
                    }
                parsed_hour12 = int(match)
                position += 1

            if parsed_hour12 < 1 or parsed_hour12 > 12:
                return {
                    "success": False,
                    "error": f"Hour out of range (got {parsed_hour12}, expected 1-12)",
                    "position": position - len(match),
                }
            if hour12 is not None and hour12 != parsed_hour12:
                return {
                    "success": False,
                    "error": f"Hour (12-hour) specified multiple times with different values: {hour12} and {parsed_hour12}",
                    "position": position - len(match),
                }
            hour12 = parsed_hour12

        elif token_type == "minute2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit minute (00-59)",
                    "position": position,
                }
            parsed_minute = int(match)
            if parsed_minute > 59:
                return {
                    "success": False,
                    "error": f"Minute out of range (got {parsed_minute}, expected 00-59)",
                    "position": position,
                }
            if minute is not None and minute != parsed_minute:
                return {
                    "success": False,
                    "error": f"Minute specified multiple times with different values: {minute} and {parsed_minute}",
                    "position": position,
                }
            minute = parsed_minute
            position += 2

        elif token_type == "minute1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_minute = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit minute",
                        "position": position,
                    }
                parsed_minute = int(match)
                position += 1

            if parsed_minute > 59:
                return {
                    "success": False,
                    "error": f"Minute out of range (got {parsed_minute}, expected 0-59)",
                    "position": position - len(match),
                }
            if minute is not None and minute != parsed_minute:
                return {
                    "success": False,
                    "error": f"Minute specified multiple times with different values: {minute} and {parsed_minute}",
                    "position": position - len(match),
                }
            minute = parsed_minute

        elif token_type == "second2":
            match = input_str[position : position + 2]
            if len(match) < 2 or not re.match(r"^\d{2}$", match):
                return {
                    "success": False,
                    "error": "Expected 2-digit second (00-59)",
                    "position": position,
                }
            parsed_second = int(match)
            if parsed_second > 59:
                return {
                    "success": False,
                    "error": f"Second out of range (got {parsed_second}, expected 00-59)",
                    "position": position,
                }
            if second is not None and second != parsed_second:
                return {
                    "success": False,
                    "error": f"Second specified multiple times with different values: {second} and {parsed_second}",
                    "position": position,
                }
            second = parsed_second
            position += 2

        elif token_type == "second1":
            # Try 2 digits first, then 1
            match = input_str[position : position + 2]
            if re.match(r"^\d{2}$", match):
                parsed_second = int(match)
                position += 2
            else:
                match = input_str[position : position + 1]
                if not re.match(r"^\d$", match):
                    return {
                        "success": False,
                        "error": "Expected 1 or 2-digit second",
                        "position": position,
                    }
                parsed_second = int(match)
                position += 1

            if parsed_second > 59:
                return {
                    "success": False,
                    "error": f"Second out of range (got {parsed_second}, expected 0-59)",
                    "position": position - len(match),
                }
            if second is not None and second != parsed_second:
                return {
                    "success": False,
                    "error": f"Second specified multiple times with different values: {second} and {parsed_second}",
                    "position": position - len(match),
                }
            second = parsed_second

        elif token_type == "millisecond3":
            match = input_str[position : position + 3]
            if len(match) < 3 or not re.match(r"^\d{3}$", match):
                return {
                    "success": False,
                    "error": "Expected 3-digit millisecond (000-999)",
                    "position": position,
                }
            parsed_millisecond = int(match)
            if millisecond is not None and millisecond != parsed_millisecond:
                return {
                    "success": False,
                    "error": f"Millisecond specified multiple times with different values: {millisecond} and {parsed_millisecond}",
                    "position": position,
                }
            millisecond = parsed_millisecond
            position += 3

        elif token_type == "ampmUpper":
            match = input_str[position : position + 2].upper()
            if match == "AM":
                is_pm = False
                position += 2
            elif match == "PM":
                is_pm = True
                position += 2
            else:
                return {"success": False, "error": 'Expected "AM" or "PM"', "position": position}

        elif token_type == "ampmLower":
            match = input_str[position : position + 2].lower()
            if match == "am":
                is_pm = False
                position += 2
            elif match == "pm":
                is_pm = True
                position += 2
            else:
                return {"success": False, "error": 'Expected "am" or "pm"', "position": position}

        elif token_type == "literal":
            expected = token_value
            assert isinstance(expected, str)
            actual = input_str[position : position + len(expected)]
            if actual != expected:
                return {
                    "success": False,
                    "error": f'Expected literal "{expected}", got "{actual}"',
                    "position": position,
                }
            position += len(expected)

        else:
            return {
                "success": False,
                "error": f"Unknown token type: {token_type}",
                "position": position,
            }

    # Check for unconsumed input
    if position < len(input_str):
        return {
            "success": False,
            "error": f'Unexpected trailing characters: "{input_str[position:]}"',
            "position": position,
        }

    # Prefix validation: Fill in defaults while checking for gaps in the component hierarchy
    # The hierarchy is: Year → Month → Day → Hour → Minute → Second → Millisecond
    # We check for gaps and fill in defaults from most significant to least significant

    # Check if we have ANY date components to determine if this is a time-only format
    has_any_date_component = year is not None or month is not None or day is not None

    found_gap = False

    # Date components
    if year is None:
        if has_any_date_component:
            found_gap = True
        year = 1970  # Default epoch year for time-only formats

    if month is None:
        if has_any_date_component:
            found_gap = True
        month = 1
    elif found_gap:
        # Can't have month if we already found a gap (e.g., no year)
        return {
            "success": False,
            "error": "Invalid format: cannot have month without year",
            "position": 0,
        }

    if day is None:
        if has_any_date_component:
            found_gap = True
        day = 1
    elif found_gap:
        # Can't have day if we already found a gap
        return {
            "success": False,
            "error": "Invalid format: cannot have day without year and month",
            "position": 0,
        }

    # Hour validation and conversion: handle hour24 vs hour12+AM/PM redundancy
    if hour is not None and hour12 is not None:
        # Both hour24 and hour12 specified - validate they match
        if is_pm is None:
            # hour12 without AM/PM is ambiguous - can't validate
            return {
                "success": False,
                "error": "12-hour format specified without AM/PM indicator",
                "position": 0,
            }

        # Convert hour12 + is_pm to hour24
        if is_pm:
            expected_hour24 = 12 if hour12 == 12 else hour12 + 12  # 12 PM -> 12, 1 PM -> 13
        else:
            expected_hour24 = 0 if hour12 == 12 else hour12  # 12 AM -> 0, 1 AM -> 1

        if hour != expected_hour24:
            return {
                "success": False,
                "error": f"Hour mismatch: 24-hour format specifies {hour}, but 12-hour format with {'PM' if is_pm else 'AM'} implies {expected_hour24}",
                "position": 0,
            }
    elif hour is None and hour12 is not None:
        # Only hour12 specified - convert to hour24
        if is_pm is None:
            return {
                "success": False,
                "error": "12-hour format specified without AM/PM indicator",
                "position": 0,
            }

        if is_pm:
            hour = 12 if hour12 == 12 else hour12 + 12  # 12 PM -> 12, 1 PM -> 13
        else:
            hour = 0 if hour12 == 12 else hour12  # 12 AM -> 0, 1 AM -> 1

    # Continue with time components
    if hour is None:
        found_gap = True
        hour = 0
    elif found_gap:
        return {
            "success": False,
            "error": "Invalid format: cannot have hour without year, month, and day (or use time-only format)",
            "position": 0,
        }

    if minute is None:
        found_gap = True
        minute = 0
    elif found_gap:
        return {
            "success": False,
            "error": "Invalid format: cannot have minute without hour",
            "position": 0,
        }

    if second is None:
        found_gap = True
        second = 0
    elif found_gap:
        return {
            "success": False,
            "error": "Invalid format: cannot have second without minute",
            "position": 0,
        }

    if millisecond is None:
        millisecond = 0
    elif found_gap:
        return {
            "success": False,
            "error": "Invalid format: cannot have millisecond without second",
            "position": 0,
        }

    # Construct datetime using UTC
    # Note: datetime expects microseconds, not milliseconds
    try:
        dt = datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=UTC)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid date: {year}-{str(month).zfill(2)}-{str(day).zfill(2)} ({str(e)})",
            "position": 0,
        }

    # Validate the date is actually valid (e.g., not Feb 31)
    if dt.year != year or dt.month != month or dt.day != day:
        return {
            "success": False,
            "error": f"Invalid date: {year}-{str(month).zfill(2)}-{str(day).zfill(2)}",
            "position": 0,
        }

    # Validate weekday if it was parsed
    if parsed_weekday is not None:
        # Python weekday(): 0=Monday, 1=Tuesday, ..., 6=Sunday
        # JavaScript getUTCDay(): 0=Sunday, 1=Monday, ..., 6=Saturday
        # Convert Python weekday to JavaScript getUTCDay format
        actual_weekday = (dt.weekday() + 1) % 7  # Convert Mon=0 to Sun=0 format
        if actual_weekday != parsed_weekday:
            weekday_names = [
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
            return {
                "success": False,
                "error": f'Weekday mismatch: parsed "{weekday_names[parsed_weekday]}" but date is actually "{weekday_names[actual_weekday]}"',
                "position": 0,
            }

    return {"success": True, "value": dt}


__all__ = ["parse_datetime_formatted", "DateTimeParseResult", "ParseSuccess", "ParseError"]
