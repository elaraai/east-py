"""DateTime builtin functions."""

from datetime import UTC, datetime, timedelta

from east.builtins.registry import register_builtin


def datetime_now() -> datetime:
    """Get current datetime in UTC.

    Returns:
        Current datetime
    """
    return datetime.now(UTC)


def datetime_parse(s: str) -> datetime:
    """Parse ISO 8601 datetime string.

    Args:
        s: ISO 8601 datetime string

    Returns:
        Parsed datetime

    Raises:
        ValueError: If string is not valid ISO 8601
    """
    # Try parsing with different ISO 8601 formats
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    ]:
        try:
            dt = datetime.strptime(s, fmt)
            # Ensure UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue

    raise ValueError(f"Invalid ISO 8601 datetime: {s}")


def datetime_format(dt: datetime) -> str:
    """Format datetime as ISO 8601 string.

    Args:
        dt: DateTime

    Returns:
        ISO 8601 formatted string
    """
    # Convert to UTC and format
    dt_utc = dt.astimezone(UTC)
    # Format with microseconds, removing trailing zeros
    iso = dt_utc.isoformat()
    # Ensure it ends with Z for UTC
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def datetime_add(dt: datetime, seconds: int) -> datetime:
    """Add seconds to datetime.

    Args:
        dt: DateTime
        seconds: Seconds to add (can be negative)

    Returns:
        New datetime
    """
    return dt + timedelta(seconds=seconds)


def datetime_subtract(dt: datetime, seconds: int) -> datetime:
    """Subtract seconds from datetime.

    Args:
        dt: DateTime
        seconds: Seconds to subtract (can be negative)

    Returns:
        New datetime
    """
    return dt - timedelta(seconds=seconds)


def datetime_difference(a: datetime, b: datetime) -> int:
    """Get difference between two datetimes in seconds.

    Args:
        a: First datetime
        b: Second datetime

    Returns:
        Difference in seconds (a - b)
    """
    delta = a - b
    return int(delta.total_seconds())


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


# Register all datetime builtins
register_builtin("DateTimeNow", datetime_now)
register_builtin("DateTimeParse", datetime_parse)
register_builtin("DateTimeFormat", datetime_format)
register_builtin("DateTimeAdd", datetime_add)
register_builtin("DateTimeSubtract", datetime_subtract)
register_builtin("DateTimeDifference", datetime_difference)
register_builtin("DateTimeYear", datetime_year)
register_builtin("DateTimeMonth", datetime_month)
register_builtin("DateTimeDay", datetime_day)
register_builtin("DateTimeHour", datetime_hour)
register_builtin("DateTimeMinute", datetime_minute)
register_builtin("DateTimeSecond", datetime_second)
register_builtin("DateTimeMillisecond", datetime_millisecond)


__all__ = [
    "datetime_now",
    "datetime_parse",
    "datetime_format",
    "datetime_add",
    "datetime_subtract",
    "datetime_difference",
    "datetime_year",
    "datetime_month",
    "datetime_day",
    "datetime_hour",
    "datetime_minute",
    "datetime_second",
    "datetime_millisecond",
]
