"""Validation for datetime format tokens.

Ensures tokens form a valid contiguous prefix of components (can't skip from
year to hour without month/day).
"""

from typing import Any, TypedDict


class ValidationSuccess(TypedDict):
    """Successful validation result."""

    valid: bool  # Always True


class ValidationError(TypedDict):
    """Failed validation result with error message."""

    valid: bool  # Always False
    error: str


DateTimeFormatValidationResult = ValidationSuccess | ValidationError


def validate_datetime_format_tokens(tokens: list[Any]) -> DateTimeFormatValidationResult:
    """Validates that datetime format tokens form a valid contiguous prefix.

    The component hierarchy is: Year → Month → Day → Hour → Minute → Second → Millisecond

    Valid prefixes:
    - (empty) - for time-only formats like "HH:mm"
    - Year - e.g., "YYYY"
    - Year, Month - e.g., "YYYY-MM"
    - Year, Month, Day - e.g., "YYYY-MM-DD"
    - Year, Month, Day, Hour - e.g., "YYYY-MM-DD HH"
    - Year, Month, Day, Hour, Minute - e.g., "YYYY-MM-DD HH:mm"
    - etc.

    Invalid combinations:
    - Year, Hour (skipped Month and Day) - e.g., "YYYY HH:mm"
    - Month without Year - e.g., "MM-DD"
    - Day without Year and Month - e.g., "DD"

    Args:
        tokens: Array of format tokens to validate (EastVariant objects or dicts)

    Returns:
        Validation result indicating success or error

    Example:
        >>> tokens = tokenize_datetime_format("YYYY HH:mm")
        >>> result = validate_datetime_format_tokens(tokens)
        >>> if not result["valid"]:
        ...     print(result["error"])  # "Invalid format: cannot have hour without month and day..."
    """
    # Check which component categories are present
    has_year = False
    has_month = False
    has_day = False
    has_hour = False
    has_minute = False
    has_second = False
    has_millisecond = False

    for token in tokens:
        # Handle both dict and EastVariant formats
        token_type = token["type"] if isinstance(token, dict) else token.tag

        if token_type in ("year4", "year2"):
            has_year = True
        elif token_type in ("month2", "month1", "monthNameFull", "monthNameShort"):
            has_month = True
        elif token_type in ("day2", "day1"):
            has_day = True
        elif token_type in ("hour24_2", "hour24_1", "hour12_2", "hour12_1"):
            has_hour = True
        elif token_type in ("minute2", "minute1"):
            has_minute = True
        elif token_type in ("second2", "second1"):
            has_second = True
        elif token_type == "millisecond3":
            has_millisecond = True
        elif token_type in (
            "weekdayNameFull",
            "weekdayNameShort",
            "weekdayNameMin",
            "ampmUpper",
            "ampmLower",
            "literal",
        ):
            # Weekday, AM/PM, and literals don't affect the hierarchy
            pass
        else:
            return {"valid": False, "error": f"Unknown token type: {token_type}"}

    # Validate the "contiguous prefix" invariant
    # The hierarchy is: Year → Month → Day → Hour → Minute → Second → Millisecond

    # If we have month, we must have year
    if has_month and not has_year:
        return {
            "valid": False,
            "error": "Invalid format: cannot have month without year. Use YYYY or YY before month tokens.",
        }

    # If we have day, we must have year and month
    if has_day and not has_year:
        return {
            "valid": False,
            "error": "Invalid format: cannot have day without year. Use YYYY or YY before day tokens.",
        }
    if has_day and not has_month:
        return {
            "valid": False,
            "error": "Invalid format: cannot have day without month. Use MM or M before day tokens.",
        }

    # If we have hour, we must have year, month, and day (or none of them for time-only)
    if has_hour and has_year and not has_month:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year to hour. Include month (MM or M) and day (DD or D) tokens, or remove year for time-only format.",
        }
    if has_hour and has_year and has_month and not has_day:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year/month to hour. Include day (DD or D) token, or remove year/month for time-only format.",
        }

    # If we have minute, we must have hour (and if we have date components, all of them)
    if has_minute and not has_hour:
        return {
            "valid": False,
            "error": "Invalid format: cannot have minute without hour. Use HH or H before minute tokens.",
        }
    if has_minute and has_year and not has_month:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year to minute. Include month and day tokens, or remove year for time-only format.",
        }
    if has_minute and has_year and has_month and not has_day:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year/month to minute. Include day token, or remove year/month for time-only format.",
        }

    # If we have second, we must have hour and minute
    if has_second and not has_hour:
        return {
            "valid": False,
            "error": "Invalid format: cannot have second without hour. Use HH or H before second tokens.",
        }
    if has_second and not has_minute:
        return {
            "valid": False,
            "error": "Invalid format: cannot have second without minute. Use mm or m before second tokens.",
        }
    if has_second and has_year and not has_month:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year to second. Include month and day tokens, or remove year for time-only format.",
        }
    if has_second and has_year and has_month and not has_day:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year/month to second. Include day token, or remove year/month for time-only format.",
        }

    # If we have millisecond, we must have hour, minute, and second
    if has_millisecond and not has_hour:
        return {
            "valid": False,
            "error": "Invalid format: cannot have millisecond without hour. Use HH or H before millisecond tokens.",
        }
    if has_millisecond and not has_minute:
        return {
            "valid": False,
            "error": "Invalid format: cannot have millisecond without minute. Use mm or m before millisecond tokens.",
        }
    if has_millisecond and not has_second:
        return {
            "valid": False,
            "error": "Invalid format: cannot have millisecond without second. Use ss or s before millisecond tokens.",
        }
    if has_millisecond and has_year and not has_month:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year to millisecond. Include month and day tokens, or remove year for time-only format.",
        }
    if has_millisecond and has_year and has_month and not has_day:
        return {
            "valid": False,
            "error": "Invalid format: cannot skip from year/month to millisecond. Include day token, or remove year/month for time-only format.",
        }

    return {"valid": True}


__all__ = ["validate_datetime_format_tokens", "DateTimeFormatValidationResult"]
