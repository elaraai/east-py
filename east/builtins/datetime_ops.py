"""DateTime builtin functions."""

from datetime import UTC, datetime, timedelta
from typing import Any

from east.builtins.registry import register_builtin
from east.datetime_format.parse import parse_datetime_formatted
from east.datetime_format.print import format_datetime


def datetime_add(dt: datetime, milliseconds: int) -> datetime:
    """Add milliseconds to datetime.

    Args:
        dt: DateTime
        milliseconds: Milliseconds to add (can be negative)

    Returns:
        New datetime
    """
    return dt + timedelta(milliseconds=milliseconds)


def datetime_difference(a: datetime, b: datetime) -> int:
    """Get difference between two datetimes in milliseconds.

    Args:
        a: First datetime
        b: Second datetime

    Returns:
        Difference in milliseconds (a - b)
    """
    delta = a - b
    return int(delta.total_seconds() * 1000)


def datetime_year(dt: datetime) -> int:
    """Get year component.

    Args:
        dt: DateTime

    Returns:
        Year (e.g., 2024)
    """
    return dt.year


def datetime_month(dt: datetime) -> int:
    """Get month component.

    Args:
        dt: DateTime

    Returns:
        Month (1-12)
    """
    return dt.month


def datetime_day(dt: datetime) -> int:
    """Get day component.

    Args:
        dt: DateTime

    Returns:
        Day (1-31)
    """
    return dt.day


def datetime_hour(dt: datetime) -> int:
    """Get hour component.

    Args:
        dt: DateTime

    Returns:
        Hour (0-23)
    """
    return dt.hour


def datetime_minute(dt: datetime) -> int:
    """Get minute component.

    Args:
        dt: DateTime

    Returns:
        Minute (0-59)
    """
    return dt.minute


def datetime_second(dt: datetime) -> int:
    """Get second component.

    Args:
        dt: DateTime

    Returns:
        Second (0-59)
    """
    return dt.second


def datetime_millisecond(dt: datetime) -> int:
    """Get millisecond component.

    Args:
        dt: DateTime

    Returns:
        Millisecond (0-999)
    """
    return dt.microsecond // 1000


def datetime_get_day_of_week(dt: datetime) -> int:
    """Get day of week (ISO 8601 format).

    Args:
        dt: DateTime

    Returns:
        Day of week (1=Monday, 7=Sunday) per ISO 8601
    """
    # Python's weekday() returns 0=Monday, 6=Sunday
    # ISO 8601 expects 1=Monday, 7=Sunday
    return dt.weekday() + 1


def datetime_to_epoch_milliseconds(dt: datetime) -> int:
    """Convert datetime to Unix epoch milliseconds.

    Args:
        dt: DateTime

    Returns:
        Milliseconds since Unix epoch (1970-01-01T00:00:00Z)
    """
    return int(dt.timestamp() * 1000)


def datetime_from_epoch_milliseconds(milliseconds: int) -> datetime:
    """Create datetime from Unix epoch milliseconds.

    Args:
        milliseconds: Milliseconds since Unix epoch

    Returns:
        DateTime in UTC
    """
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def datetime_from_components(
    year: int, month: int, day: int, hour: int, minute: int, second: int, millisecond: int
) -> datetime:
    """Create datetime from components.

    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        day: Day (1-31)
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)
        millisecond: Millisecond (0-999)

    Returns:
        DateTime in UTC
    """
    return datetime(year, month, day, hour, minute, second, millisecond * 1000, tzinfo=UTC)


def datetime_print_format(dt: datetime, tokens: list[Any]) -> str:
    """Format datetime using format token array.

    Args:
        dt: DateTime to format
        tokens: Array of DateTimeFormatToken variants

    Returns:
        Formatted datetime string

    Example:
        >>> from east.datetime_format import tokenize_datetime_format
        >>> dt = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        >>> tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        >>> datetime_print_format(dt, tokens)
        '2025-01-15 14:30:45'
    """
    return format_datetime(dt, tokens)


def datetime_parse_format(text: str, tokens: list[Any]) -> datetime:
    """Parse datetime using format token array.

    Args:
        text: String to parse
        tokens: Array of DateTimeFormatToken variants

    Returns:
        Parsed datetime in UTC

    Raises:
        ValueError: If parsing fails

    Example:
        >>> from east.datetime_format import tokenize_datetime_format
        >>> tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        >>> datetime_parse_format("2025-01-15 14:30:45", tokens)
        datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
    """
    result = parse_datetime_formatted(text, tokens)

    if result["success"]:
        return result["value"]
    # Format error with position
    error_msg = result["error"]
    position = result["position"]
    raise ValueError(f"Failed to parse datetime at position {position}: {error_msg}")


# Register all datetime builtins
register_builtin("DateTimeAddMilliseconds", datetime_add)  # Renamed from DateTimeAdd
register_builtin(
    "DateTimeDurationMilliseconds", datetime_difference
)  # Renamed from DateTimeDifference
register_builtin("DateTimeGetYear", datetime_year)  # Renamed from DateTimeYear
register_builtin("DateTimeGetMonth", datetime_month)  # Renamed from DateTimeMonth
register_builtin("DateTimeGetDayOfMonth", datetime_day)  # Renamed from DateTimeDay
register_builtin("DateTimeGetHour", datetime_hour)  # Renamed from DateTimeHour
register_builtin("DateTimeGetMinute", datetime_minute)  # Renamed from DateTimeMinute
register_builtin("DateTimeGetSecond", datetime_second)  # Renamed from DateTimeSecond
register_builtin("DateTimeGetMillisecond", datetime_millisecond)  # Renamed from DateTimeMillisecond
register_builtin("DateTimeGetDayOfWeek", datetime_get_day_of_week)
register_builtin("DateTimeToEpochMilliseconds", datetime_to_epoch_milliseconds)
register_builtin("DateTimeFromEpochMilliseconds", datetime_from_epoch_milliseconds)
register_builtin("DateTimeFromComponents", datetime_from_components)
register_builtin("DateTimePrintFormat", datetime_print_format)
register_builtin("DateTimeParseFormat", datetime_parse_format)


__all__ = [
    "datetime_add",
    "datetime_difference",
    "datetime_year",
    "datetime_month",
    "datetime_day",
    "datetime_hour",
    "datetime_minute",
    "datetime_second",
    "datetime_millisecond",
    "datetime_get_day_of_week",
    "datetime_to_epoch_milliseconds",
    "datetime_from_epoch_milliseconds",
    "datetime_from_components",
    "datetime_print_format",
    "datetime_parse_format",
]
