"""Tests for datetime parsing.

This module tests the parseDateTimeFormatted function.
"""

from datetime import UTC, datetime

from east.datetime_format import parse_datetime_formatted, tokenize_datetime_format
from east.types.type_system import NullType, StringType, _VariantTypeClass


def v(tag: str, value=None):
    """Helper to create datetime format token variants for testing."""
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
    variant_type = _VariantTypeClass(tuple(cases))
    return variant_type.create(tag, value)


def success(result):
    """Check if parse result is success."""
    return result["success"]


def value(result):
    """Get value from successful parse result."""
    return result["value"]


def error(result):
    """Get error from failed parse result."""
    return result["error"]


def position(result):
    """Get position from failed parse result."""
    return result["position"]


class TestParse:
    """Tests for parseDateTimeFormatted function."""

    def test_basic_iso_8601_date(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-01-15", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1  # January
            assert result["value"].day == 15

    def test_iso_8601_datetime(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        result = parse_datetime_formatted("2025-01-15 14:30:45", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1
            assert result["value"].day == 15
            assert result["value"].hour == 14
            assert result["value"].minute == 30
            assert result["value"].second == 45

    def test_with_milliseconds(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")
        result = parse_datetime_formatted("2025-01-15 14:30:45.123", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].microsecond == 123000  # 123 milliseconds

    def test_12_hour_format_with_pm(self):
        tokens = tokenize_datetime_format("h:mm A")
        result = parse_datetime_formatted("2:30 PM", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].hour == 14  # 2 PM = 14:00
            assert result["value"].minute == 30

    def test_12_hour_format_with_am(self):
        tokens = tokenize_datetime_format("h:mm A")
        result = parse_datetime_formatted("9:30 AM", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].hour == 9
            assert result["value"].minute == 30

    def test_12_am_midnight(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD h A")
        result = parse_datetime_formatted("2025-01-15 12 AM", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].hour == 0  # 12 AM = 00:00

    def test_12_pm_noon(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD h A")
        result = parse_datetime_formatted("2025-01-15 12 PM", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].hour == 12  # 12 PM = 12:00

    def test_month_names_full(self):
        tokens = tokenize_datetime_format("MMMM D, YYYY")
        result = parse_datetime_formatted("January 15, 2025", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1  # January
            assert result["value"].day == 15

    def test_month_names_short(self):
        tokens = tokenize_datetime_format("MMM D, YYYY")
        result = parse_datetime_formatted("Jan 15, 2025", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].month == 1  # January

    def test_month_names_case_insensitive(self):
        tokens = tokenize_datetime_format("MMMM D, YYYY")
        result = parse_datetime_formatted("january 15, 2025", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].month == 1  # January

    def test_weekday_names_ignored_but_consumed(self):
        tokens = tokenize_datetime_format("dddd, MMMM D, YYYY")
        result = parse_datetime_formatted("Wednesday, January 15, 2025", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1
            assert result["value"].day == 15

    def test_unpadded_single_digit_month(self):
        tokens = tokenize_datetime_format("M/D/YY")
        result = parse_datetime_formatted("1/5/25", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1  # January
            assert result["value"].day == 5

    def test_unpadded_double_digit_month(self):
        tokens = tokenize_datetime_format("M/D/YY")
        result = parse_datetime_formatted("12/25/25", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].month == 12  # December
            assert result["value"].day == 25

    def test_2_digit_year(self):
        tokens = tokenize_datetime_format("YY-MM-DD")
        result = parse_datetime_formatted("25-01-15", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025  # 25 -> 2025

    def test_error_missing_year(self):
        tokens = [v("month2"), v("literal", "-"), v("day2")]
        result = parse_datetime_formatted("01-15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "year" in result["error"].lower()

    def test_error_missing_month(self):
        tokens = [v("year4"), v("literal", "-"), v("day2")]
        result = parse_datetime_formatted("2025-15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "month" in result["error"].lower()

    def test_missing_day_defaults_to_1st(self):
        tokens = [v("year4"), v("literal", "-"), v("month2")]
        result = parse_datetime_formatted("2025-01", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1  # January
            assert result["value"].day == 1  # Defaults to 1st

    def test_error_month_out_of_range(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-13-15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()
            assert result["position"] == 5

    def test_error_day_out_of_range(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-01-32", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()

    def test_error_invalid_date_feb_31(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-02-31", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "invalid date" in result["error"].lower()

    def test_valid_leap_year_date(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2024-02-29", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2024
            assert result["value"].month == 2  # February
            assert result["value"].day == 29

    def test_error_invalid_leap_year_date(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-02-29", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "invalid date" in result["error"].lower()

    def test_error_literal_mismatch(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025/01/15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "expected literal" in result["error"].lower()
            assert result["position"] == 4

    def test_error_trailing_characters(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-01-15 extra", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "trailing characters" in result["error"].lower()

    def test_error_unexpected_end_of_input(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        result = parse_datetime_formatted("2025-01-15", tokens)

        assert not result["success"]
        if not result["success"]:
            error_lower = result["error"].lower()
            assert "unexpected end of input" in error_lower or "expected literal" in error_lower

    def test_error_expected_4_digit_year(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("25-01-15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "4-digit year" in result["error"].lower()
            assert result["position"] == 0

    def test_error_expected_2_digit_month(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-1-15", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "2-digit month" in result["error"].lower()

    def test_complex_format_with_weekday(self):
        tokens = tokenize_datetime_format("ddd, MMM D, YYYY \\a\\t h:mm A")
        result = parse_datetime_formatted("Wed, Jan 15, 2025 at 2:30 PM", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].year == 2025
            assert result["value"].month == 1
            assert result["value"].day == 15
            assert result["value"].hour == 14
            assert result["value"].minute == 30

    def test_round_trip_format_then_parse(self):
        format_str = "YYYY-MM-DD HH:mm:ss.SSS"
        tokens = tokenize_datetime_format(format_str)

        original_date = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        formatted = "2025-01-15 14:30:45.123"
        result = parse_datetime_formatted(formatted, tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].replace(tzinfo=UTC).timestamp() == original_date.timestamp()

    def test_all_months(self):
        short_names = [
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
        tokens = tokenize_datetime_format("MMM YYYY")

        for i in range(12):
            result = parse_datetime_formatted(f"{short_names[i]} 2025", tokens)
            assert result["success"], f"Failed to parse {short_names[i]}"
            if result["success"]:
                assert result["value"].month == i + 1

    def test_all_weekdays(self):
        # Use dates that actually match each weekday (January 2025)
        weekdays_with_dates = [
            {"name": "Sunday", "date": "2025-01-05"},  # Jan 5, 2025 is Sunday
            {"name": "Monday", "date": "2025-01-06"},  # Jan 6, 2025 is Monday
            {"name": "Tuesday", "date": "2025-01-07"},  # Jan 7, 2025 is Tuesday
            {"name": "Wednesday", "date": "2025-01-15"},  # Jan 15, 2025 is Wednesday
            {"name": "Thursday", "date": "2025-01-09"},  # Jan 9, 2025 is Thursday
            {"name": "Friday", "date": "2025-01-10"},  # Jan 10, 2025 is Friday
            {"name": "Saturday", "date": "2025-01-11"},  # Jan 11, 2025 is Saturday
        ]
        tokens = tokenize_datetime_format("dddd, YYYY-MM-DD")

        for item in weekdays_with_dates:
            result = parse_datetime_formatted(f"{item['name']}, {item['date']}", tokens)
            assert result["success"], f"Failed to parse {item['name']}"

    def test_lowercase_ampm(self):
        tokens = tokenize_datetime_format("h:mm a")

        result_am = parse_datetime_formatted("9:30 am", tokens)
        assert result_am["success"]
        if result_am["success"]:
            assert result_am["value"].hour == 9

        result_pm = parse_datetime_formatted("2:30 pm", tokens)
        assert result_pm["success"]
        if result_pm["success"]:
            assert result_pm["value"].hour == 14

    def test_defaults_for_missing_time_components(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        result = parse_datetime_formatted("2025-01-15", tokens)

        assert result["success"]
        if result["success"]:
            # Time components should default to 0
            assert result["value"].hour == 0
            assert result["value"].minute == 0
            assert result["value"].second == 0
            assert result["value"].microsecond == 0

    def test_unpadded_time_components(self):
        tokens = tokenize_datetime_format("H:m:s")
        result = parse_datetime_formatted("9:5:7", tokens)

        assert result["success"]
        if result["success"]:
            assert result["value"].hour == 9
            assert result["value"].minute == 5
            assert result["value"].second == 7

    def test_error_hour_24_hour_out_of_range(self):
        tokens = tokenize_datetime_format("HH:mm")
        result = parse_datetime_formatted("24:00", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()

    def test_error_hour_12_hour_out_of_range(self):
        tokens = tokenize_datetime_format("hh:mm A")
        result = parse_datetime_formatted("13:00 PM", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()

    def test_error_minute_out_of_range(self):
        tokens = tokenize_datetime_format("HH:mm")
        result = parse_datetime_formatted("14:60", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()

    def test_error_second_out_of_range(self):
        tokens = tokenize_datetime_format("HH:mm:ss")
        result = parse_datetime_formatted("14:30:60", tokens)

        assert not result["success"]
        if not result["success"]:
            assert "out of range" in result["error"].lower()
