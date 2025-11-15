"""Tests for datetime formatting.

This module tests the format_datetime function.
"""

from datetime import UTC, datetime

from east.datetime_format import format_datetime, tokenize_datetime_format


def v(tag: str, value=None):
    """Helper to create datetime format token variants for testing."""
    return {"type": tag, "value": value}


class TestFormatDateTime:
    """Tests for format_datetime function."""

    def test_year_tokens(self):
        date = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)  # Jan 15, 2025 14:30:45.123

        assert format_datetime(date, [v("year4")]) == "2025"
        assert format_datetime(date, [v("year2")]) == "25"

    def test_year2_padding(self):
        date = datetime(2001, 1, 1, tzinfo=UTC)  # Year ending in 01

        assert format_datetime(date, [v("year2")]) == "01"

    def test_month_tokens(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)  # January

        assert format_datetime(date, [v("month1")]) == "1"
        assert format_datetime(date, [v("month2")]) == "01"
        assert format_datetime(date, [v("monthNameShort")]) == "Jan"
        assert format_datetime(date, [v("monthNameFull")]) == "January"

    def test_month_names_for_all_months(self):
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
        full_names = [
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

        for month in range(1, 13):  # Python months are 1-indexed
            date = datetime(2025, month, 15, tzinfo=UTC)

            assert format_datetime(date, [v("monthNameShort")]) == short_names[month - 1]
            assert format_datetime(date, [v("monthNameFull")]) == full_names[month - 1]

    def test_day_tokens(self):
        date = datetime(2025, 1, 5, tzinfo=UTC)  # 5th day

        assert format_datetime(date, [v("day1")]) == "5"
        assert format_datetime(date, [v("day2")]) == "05"

    def test_weekday_tokens(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)  # Wednesday

        assert format_datetime(date, [v("weekdayNameMin")]) == "We"
        assert format_datetime(date, [v("weekdayNameShort")]) == "Wed"
        assert format_datetime(date, [v("weekdayNameFull")]) == "Wednesday"

    def test_weekday_names_for_all_days(self):
        min_names = ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
        short_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        full_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

        # January 5, 2025 is a Sunday, so we can iterate from there
        for day_offset in range(7):
            date = datetime(2025, 1, 5 + day_offset, tzinfo=UTC)

            assert (
                format_datetime(date, [v("weekdayNameMin")]) == min_names[day_offset]
            ), f"Day {day_offset} min name"
            assert (
                format_datetime(date, [v("weekdayNameShort")]) == short_names[day_offset]
            ), f"Day {day_offset} short name"
            assert (
                format_datetime(date, [v("weekdayNameFull")]) == full_names[day_offset]
            ), f"Day {day_offset} full name"

    def test_24_hour_format(self):
        date = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)  # 9:30 AM

        assert format_datetime(date, [v("hour24_1")]) == "9"
        assert format_datetime(date, [v("hour24_2")]) == "09"

        date2 = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)  # 2:30 PM

        assert format_datetime(date2, [v("hour24_1")]) == "14"
        assert format_datetime(date2, [v("hour24_2")]) == "14"

    def test_12_hour_format(self):
        date = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)  # 9:30 AM

        assert format_datetime(date, [v("hour12_1")]) == "9"
        assert format_datetime(date, [v("hour12_2")]) == "09"

        date2 = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)  # 2:30 PM

        assert format_datetime(date2, [v("hour12_1")]) == "2"
        assert format_datetime(date2, [v("hour12_2")]) == "02"

    def test_12_hour_format_edge_cases(self):
        # Midnight (12:00 AM)
        midnight = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)
        assert format_datetime(midnight, [v("hour12_1")]) == "12"

        # Noon (12:00 PM)
        noon = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        assert format_datetime(noon, [v("hour12_1")]) == "12"

        # 1 AM
        one_am = datetime(2025, 1, 15, 1, 0, tzinfo=UTC)
        assert format_datetime(one_am, [v("hour12_1")]) == "1"

        # 1 PM
        one_pm = datetime(2025, 1, 15, 13, 0, tzinfo=UTC)
        assert format_datetime(one_pm, [v("hour12_1")]) == "1"

    def test_minute_tokens(self):
        date = datetime(2025, 1, 15, 14, 5, tzinfo=UTC)  # 5 minutes

        assert format_datetime(date, [v("minute1")]) == "5"
        assert format_datetime(date, [v("minute2")]) == "05"

    def test_second_tokens(self):
        date = datetime(2025, 1, 15, 14, 30, 7, tzinfo=UTC)  # 7 seconds

        assert format_datetime(date, [v("second1")]) == "7"
        assert format_datetime(date, [v("second2")]) == "07"

    def test_millisecond_token(self):
        date1 = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)  # 123ms = 123000μs

        assert format_datetime(date1, [v("millisecond3")]) == "123"

        date2 = datetime(2025, 1, 15, 14, 30, 45, 7000, tzinfo=UTC)  # 7ms = 7000μs

        assert format_datetime(date2, [v("millisecond3")]) == "007"

    def test_ampm_tokens(self):
        am = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)

        assert format_datetime(am, [v("ampmUpper")]) == "AM"
        assert format_datetime(am, [v("ampmLower")]) == "am"

        pm = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)

        assert format_datetime(pm, [v("ampmUpper")]) == "PM"
        assert format_datetime(pm, [v("ampmLower")]) == "pm"

        # Midnight is AM
        midnight = datetime(2025, 1, 15, 0, 0, tzinfo=UTC)
        assert format_datetime(midnight, [v("ampmUpper")]) == "AM"

        # Noon is PM
        noon = datetime(2025, 1, 15, 12, 0, tzinfo=UTC)
        assert format_datetime(noon, [v("ampmUpper")]) == "PM"

    def test_literal_token(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)

        assert format_datetime(date, [v("literal", "Hello")]) == "Hello"
        assert format_datetime(date, [v("literal", "Year: ")]) == "Year: "

    def test_iso_8601_date_format(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD")

        assert format_datetime(date, tokens) == "2025-01-15"

    def test_iso_8601_datetime_format(self):
        date = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")

        assert format_datetime(date, tokens) == "2025-01-15 14:30:45"

    def test_iso_8601_with_milliseconds(self):
        date = datetime(2025, 1, 15, 14, 30, 45, 123000, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")

        assert format_datetime(date, tokens) == "2025-01-15 14:30:45.123"

    def test_12_hour_format_with_ampm(self):
        date = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)
        tokens = tokenize_datetime_format("h:mm A")

        assert format_datetime(date, tokens) == "2:30 PM"

    def test_long_date_format(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = tokenize_datetime_format("MMMM D, YYYY")

        assert format_datetime(date, tokens) == "January 15, 2025"

    def test_weekday_with_date(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)  # Wednesday
        tokens = tokenize_datetime_format("dddd, MMMM D, YYYY")

        assert format_datetime(date, tokens) == "Wednesday, January 15, 2025"

    def test_compact_format(self):
        date = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)
        tokens = tokenize_datetime_format("M/D/YY h:mm A")

        assert format_datetime(date, tokens) == "1/15/25 2:30 PM"

    def test_with_escaped_literals(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = tokenize_datetime_format("\\Y\\e\\a\\r: YYYY")

        assert format_datetime(date, tokens) == "Year: 2025"

    def test_complex_real_world_format(self):
        date = datetime(2025, 1, 15, 14, 30, 45, tzinfo=UTC)
        tokens = tokenize_datetime_format("ddd, MMM D, YYYY \\a\\t h:mm:ss A")

        assert format_datetime(date, tokens) == "Wed, Jan 15, 2025 at 2:30:45 PM"

    def test_edge_case_year_2000(self):
        date = datetime(2000, 1, 1, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD")

        assert format_datetime(date, tokens) == "2000-01-01"

    def test_edge_case_leap_year(self):
        date = datetime(2024, 2, 29, tzinfo=UTC)  # Feb 29, 2024
        tokens = tokenize_datetime_format("YYYY-MM-DD")

        assert format_datetime(date, tokens) == "2024-02-29"

    def test_edge_case_end_of_year(self):
        date = datetime(2025, 12, 31, 23, 59, 59, 999000, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")

        assert format_datetime(date, tokens) == "2025-12-31 23:59:59.999"

    def test_empty_token_array(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)

        assert format_datetime(date, []) == ""

    def test_only_literals(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = [
            v("literal", "Hello "),
            v("literal", "World"),
        ]

        assert format_datetime(date, tokens) == "Hello World"

    def test_all_24_hour_hours(self):
        for hour in range(24):
            date = datetime(2025, 1, 15, hour, 0, tzinfo=UTC)
            tokens = [v("hour24_2")]

            expected = str(hour).zfill(2)
            assert format_datetime(date, tokens) == expected, f"Hour {hour}"

    def test_all_12_hour_hours_with_ampm(self):
        expected_12_hour = [
            {"hour12": "12", "ampm": "AM"},  # 0:00 = 12 AM
            {"hour12": "1", "ampm": "AM"},  # 1:00 = 1 AM
            {"hour12": "2", "ampm": "AM"},
            {"hour12": "3", "ampm": "AM"},
            {"hour12": "4", "ampm": "AM"},
            {"hour12": "5", "ampm": "AM"},
            {"hour12": "6", "ampm": "AM"},
            {"hour12": "7", "ampm": "AM"},
            {"hour12": "8", "ampm": "AM"},
            {"hour12": "9", "ampm": "AM"},
            {"hour12": "10", "ampm": "AM"},
            {"hour12": "11", "ampm": "AM"},
            {"hour12": "12", "ampm": "PM"},  # 12:00 = 12 PM
            {"hour12": "1", "ampm": "PM"},  # 13:00 = 1 PM
            {"hour12": "2", "ampm": "PM"},
            {"hour12": "3", "ampm": "PM"},
            {"hour12": "4", "ampm": "PM"},
            {"hour12": "5", "ampm": "PM"},
            {"hour12": "6", "ampm": "PM"},
            {"hour12": "7", "ampm": "PM"},
            {"hour12": "8", "ampm": "PM"},
            {"hour12": "9", "ampm": "PM"},
            {"hour12": "10", "ampm": "PM"},
            {"hour12": "11", "ampm": "PM"},
        ]

        for hour in range(24):
            date = datetime(2025, 1, 15, hour, 0, tzinfo=UTC)
            tokens = tokenize_datetime_format("h A")

            expected = f"{expected_12_hour[hour]['hour12']} {expected_12_hour[hour]['ampm']}"
            assert format_datetime(date, tokens) == expected, f"Hour {hour} ({expected})"

    def test_unicode_in_literals(self):
        date = datetime(2025, 1, 15, tzinfo=UTC)
        tokens = tokenize_datetime_format("YYYY年MM月DD日")

        assert format_datetime(date, tokens) == "2025年01月15日"

    def test_newlines_in_format(self):
        date = datetime(2025, 1, 15, 14, 30, tzinfo=UTC)
        tokens = [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
            v("literal", "\n"),
            v("hour24_2"),
            v("literal", ":"),
            v("minute2"),
        ]

        assert format_datetime(date, tokens) == "2025-01-15\n14:30"
