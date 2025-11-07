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


def datetime_add(dt: datetime, milliseconds: int) -> datetime:
    """Add milliseconds to datetime.

    Args:
        dt: DateTime
        milliseconds: Milliseconds to add (can be negative)

    Returns:
        New datetime
    """
    return dt + timedelta(milliseconds=milliseconds)


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


def datetime_get_day_of_week(dt: datetime) -> int:
    """Get day of week.

    Args:
        dt: DateTime

    Returns:
        Day of week (0=Monday, 6=Sunday)
    """
    return dt.weekday()


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


# Register all datetime builtins
# Note: DateTimeNow, DateTimeParse, DateTimeFormat, DateTimeSubtract not in spec but kept for convenience
register_builtin("DateTimeNow", datetime_now)
register_builtin("DateTimeParse", datetime_parse)
register_builtin("DateTimeFormat", datetime_format)
register_builtin("DateTimeAddMilliseconds", datetime_add)  # Renamed from DateTimeAdd
register_builtin("DateTimeSubtract", datetime_subtract)  # Not in spec
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
    "datetime_get_day_of_week",
    "datetime_to_epoch_milliseconds",
    "datetime_from_epoch_milliseconds",
    "datetime_from_components",
]
