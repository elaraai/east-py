"""Tests for datetime format tokenization.

This module tests the parseDateTimeFormat and formatTokensToString functions.
"""

from east.datetime_format import format_tokens_to_string, tokenize_datetime_format
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


class TestParseDateTimeFormat:
    """Tests for parseDateTimeFormat function."""

    def test_empty_string(self):
        result = tokenize_datetime_format("")
        assert result == []

    def test_year_tokens(self):
        assert tokenize_datetime_format("YYYY") == [v("year4")]
        assert tokenize_datetime_format("YY") == [v("year2")]

    def test_month_tokens(self):
        assert tokenize_datetime_format("M") == [v("month1")]
        assert tokenize_datetime_format("MM") == [v("month2")]
        assert tokenize_datetime_format("MMM") == [v("monthNameShort")]
        assert tokenize_datetime_format("MMMM") == [v("monthNameFull")]

    def test_day_tokens(self):
        assert tokenize_datetime_format("D") == [v("day1")]
        assert tokenize_datetime_format("DD") == [v("day2")]

    def test_weekday_tokens(self):
        assert tokenize_datetime_format("dd") == [v("weekdayNameMin")]
        assert tokenize_datetime_format("ddd") == [v("weekdayNameShort")]
        assert tokenize_datetime_format("dddd") == [v("weekdayNameFull")]

    def test_hour_tokens_24h(self):
        assert tokenize_datetime_format("H") == [v("hour24_1")]
        assert tokenize_datetime_format("HH") == [v("hour24_2")]

    def test_hour_tokens_12h(self):
        assert tokenize_datetime_format("h") == [v("hour12_1")]
        assert tokenize_datetime_format("hh") == [v("hour12_2")]

    def test_minute_tokens(self):
        assert tokenize_datetime_format("m") == [v("minute1")]
        assert tokenize_datetime_format("mm") == [v("minute2")]

    def test_second_tokens(self):
        assert tokenize_datetime_format("s") == [v("second1")]
        assert tokenize_datetime_format("ss") == [v("second2")]

    def test_millisecond_token(self):
        assert tokenize_datetime_format("SSS") == [v("millisecond3")]

    def test_ampm_tokens(self):
        assert tokenize_datetime_format("A") == [v("ampmUpper")]
        assert tokenize_datetime_format("a") == [v("ampmLower")]

    def test_iso_8601_date_format(self):
        result = tokenize_datetime_format("YYYY-MM-DD")
        assert result == [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
        ]

    def test_iso_8601_datetime_format(self):
        result = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss")
        assert result == [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
            v("literal", " "),
            v("hour24_2"),
            v("literal", ":"),
            v("minute2"),
            v("literal", ":"),
            v("second2"),
        ]

    def test_datetime_with_milliseconds(self):
        result = tokenize_datetime_format("YYYY-MM-DD HH:mm:ss.SSS")
        assert result == [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
            v("literal", " "),
            v("hour24_2"),
            v("literal", ":"),
            v("minute2"),
            v("literal", ":"),
            v("second2"),
            v("literal", "."),
            v("millisecond3"),
        ]

    def test_12_hour_format_with_ampm(self):
        result = tokenize_datetime_format("h:mm A")
        assert result == [
            v("hour12_1"),
            v("literal", ":"),
            v("minute2"),
            v("literal", " "),
            v("ampmUpper"),
        ]

    def test_long_date_format_with_month_name(self):
        result = tokenize_datetime_format("MMMM D, YYYY")
        assert result == [
            v("monthNameFull"),
            v("literal", " "),
            v("day1"),
            v("literal", ", "),
            v("year4"),
        ]

    def test_weekday_format(self):
        result = tokenize_datetime_format("dddd, MMMM D, YYYY")
        assert result == [
            v("weekdayNameFull"),
            v("literal", ", "),
            v("monthNameFull"),
            v("literal", " "),
            v("day1"),
            v("literal", ", "),
            v("year4"),
        ]

    def test_pure_literal_string_must_escape(self):
        # Unescaped format codes are parsed as tokens
        result = tokenize_datetime_format("\\H\\e\\l\\l\\o\\ \\W\\o\\r\\l\\d")
        assert result == [v("literal", "Hello World")]

    def test_literal_grouping_escape_letters_that_are_format_codes(self):
        # 'a' and 's' are format codes, so they must be escaped
        result = tokenize_datetime_format("Tod\\ay i\\s YYYY")
        assert result == [
            v("literal", "Today is "),
            v("year4"),
        ]

    def test_escape_single_character(self):
        result = tokenize_datetime_format("\\Y\\Y\\Y\\Y")
        assert result == [v("literal", "YYYY")]

    def test_escape_within_tokens(self):
        result = tokenize_datetime_format("YYYY\\-MM\\-DD")
        assert result == [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
        ]

    def test_escape_backslash(self):
        result = tokenize_datetime_format("YYYY\\\\MM")
        assert result == [
            v("year4"),
            v("literal", "\\"),
            v("month2"),
        ]

    def test_multiple_escaped_backslashes(self):
        result = tokenize_datetime_format("\\\\\\\\")
        assert result == [v("literal", "\\\\")]

    def test_escape_non_token_characters(self):
        result = tokenize_datetime_format("\\a\\b\\c")
        assert result == [v("literal", "abc")]

    def test_terminating_backslash(self):
        result = tokenize_datetime_format("YYYY\\")
        assert result == [v("year4"), v("literal", "\\")]

    def test_unicode_cjk_characters(self):
        result = tokenize_datetime_format("YYYY年MM月DD日")
        assert result == [
            v("year4"),
            v("literal", "年"),
            v("month2"),
            v("literal", "月"),
            v("day2"),
            v("literal", "日"),
        ]

    def test_unicode_emoji(self):
        result = tokenize_datetime_format("📅 YYYY-MM-DD")
        assert result == [
            v("literal", "📅 "),
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
        ]

    def test_unicode_emoji_in_escape(self):
        result = tokenize_datetime_format("\\📅")
        assert result == [v("literal", "📅")]

    def test_mixed_padding_unpadded_time(self):
        result = tokenize_datetime_format("H:m:s")
        assert result == [
            v("hour24_1"),
            v("literal", ":"),
            v("minute1"),
            v("literal", ":"),
            v("second1"),
        ]

    def test_ambiguous_mm_vs_m_longer_pattern_wins(self):
        result = tokenize_datetime_format("MM")
        assert result == [v("month2")]

    def test_ambiguous_mmmm_vs_mmm_longer_pattern_wins(self):
        result = tokenize_datetime_format("MMMM")
        assert result == [v("monthNameFull")]

    def test_adjacent_tokens_without_separators(self):
        result = tokenize_datetime_format("YYYYMMDD")
        assert result == [
            v("year4"),
            v("month2"),
            v("day2"),
        ]

    def test_adjacent_different_token_types(self):
        result = tokenize_datetime_format("YYYYMMDDHHmmss")
        assert result == [
            v("year4"),
            v("month2"),
            v("day2"),
            v("hour24_2"),
            v("minute2"),
            v("second2"),
        ]

    def test_partial_token_followed_by_literal(self):
        result = tokenize_datetime_format("Mx")
        assert result == [v("month1"), v("literal", "x")]

    def test_token_at_end(self):
        # 'D' is a format code, must escape
        result = tokenize_datetime_format("\\D\\ate: YYYY")
        assert result == [v("literal", "Date: "), v("year4")]

    def test_complex_real_world_format_1(self):
        # Must escape 'a' in "at" since it's a format code
        result = tokenize_datetime_format("dddd, MMMM D, YYYY [\\at] h:mm A")
        assert result == [
            v("weekdayNameFull"),
            v("literal", ", "),
            v("monthNameFull"),
            v("literal", " "),
            v("day1"),
            v("literal", ", "),
            v("year4"),
            v("literal", " [at] "),
            v("hour12_1"),
            v("literal", ":"),
            v("minute2"),
            v("literal", " "),
            v("ampmUpper"),
        ]

    def test_complex_real_world_format_2(self):
        result = tokenize_datetime_format("MM/DD/YYYY hh:mm:ss A")
        assert result == [
            v("month2"),
            v("literal", "/"),
            v("day2"),
            v("literal", "/"),
            v("year4"),
            v("literal", " "),
            v("hour12_2"),
            v("literal", ":"),
            v("minute2"),
            v("literal", ":"),
            v("second2"),
            v("literal", " "),
            v("ampmUpper"),
        ]

    def test_format_with_newlines(self):
        result = tokenize_datetime_format("YYYY-MM-DD\nHH:mm:ss")
        assert result == [
            v("year4"),
            v("literal", "-"),
            v("month2"),
            v("literal", "-"),
            v("day2"),
            v("literal", "\n"),
            v("hour24_2"),
            v("literal", ":"),
            v("minute2"),
            v("literal", ":"),
            v("second2"),
        ]

    def test_format_with_tabs(self):
        result = tokenize_datetime_format("YYYY\tMM\tDD")
        assert result == [
            v("year4"),
            v("literal", "\t"),
            v("month2"),
            v("literal", "\t"),
            v("day2"),
        ]

    def test_escape_n_as_literal_n_not_newline(self):
        result = tokenize_datetime_format("\\n")
        assert result == [v("literal", "n")]

    def test_escape_t_as_literal_t_not_tab(self):
        result = tokenize_datetime_format("\\t")
        assert result == [v("literal", "t")]


class TestFormatTokensToString:
    """Tests for formatTokensToString function."""

    def test_empty_token_array(self):
        assert format_tokens_to_string([]) == ""

    def test_single_format_token(self):
        assert format_tokens_to_string([v("year4")]) == "YYYY"
        assert format_tokens_to_string([v("month2")]) == "MM"

    def test_iso_8601_format(self):
        tokens = tokenize_datetime_format("YYYY-MM-DD")
        assert format_tokens_to_string(tokens) == "YYYY-MM-DD"

    def test_minimal_escaping_only_escapes_when_necessary(self):
        # "Hello" - 'H' and 'h' patterns need escaping ('HH', 'H')
        tokens1 = [v("literal", "Hello")]
        assert format_tokens_to_string(tokens1) == "\\Hello"

        # "World" - 'd' at end could form pattern, but single 'd' alone isn't a pattern
        tokens2 = [v("literal", "World")]
        assert format_tokens_to_string(tokens2) == "World"

        # "Date" - 'D' at position 0 could start 'DD' or 'D' pattern, 'a' is pattern
        tokens3 = [v("literal", "Date")]
        assert format_tokens_to_string(tokens3) == "\\D\\ate"

    def test_minimal_escaping_format_tokens_separate_literals(self):
        # When tokens are properly separated, literals stringify cleanly
        tokens = [
            v("literal", "Tod"),
            v("ampmLower"),
            v("literal", "y i"),
            v("second1"),
            v("literal", " "),
            v("year4"),
        ]
        # "Tod" safe (no D or other patterns at start), "a" token, "y i" safe, "s" token, " " safe, "YYYY" token
        assert format_tokens_to_string(tokens) == "Today is YYYY"

    def test_escapes_backslashes(self):
        tokens = [v("literal", "\\")]
        assert format_tokens_to_string(tokens) == "\\\\"

    def test_escapes_multiple_backslashes(self):
        tokens = [v("literal", "\\\\")]
        assert format_tokens_to_string(tokens) == "\\\\\\\\"

    def test_unicode_cjk_characters_need_no_escaping(self):
        tokens = [
            v("year4"),
            v("literal", "年"),
            v("month2"),
            v("literal", "月"),
        ]
        assert format_tokens_to_string(tokens) == "YYYY年MM月"

    def test_unicode_emoji_need_no_escaping(self):
        tokens = [v("literal", "📅 ")]
        assert format_tokens_to_string(tokens) == "📅 "

    def test_round_trip_property_iso_format(self):
        original = "YYYY-MM-DD HH:mm:ss"
        tokens = tokenize_datetime_format(original)
        canonical = format_tokens_to_string(tokens)
        tokens2 = tokenize_datetime_format(canonical)
        assert tokens == tokens2

    def test_round_trip_property_escaped_format(self):
        original = "\\Y\\Y\\Y\\Y-\\M\\M"
        tokens = tokenize_datetime_format(original)
        canonical = format_tokens_to_string(tokens)
        tokens2 = tokenize_datetime_format(canonical)
        assert tokens == tokens2

    def test_round_trip_property_mixed(self):
        original = "Tod\\ay i\\s YYYY"
        tokens = tokenize_datetime_format(original)
        canonical = format_tokens_to_string(tokens)
        tokens2 = tokenize_datetime_format(canonical)
        assert tokens == tokens2

    def test_round_trip_property_complex_format(self):
        original = "dddd, MMMM D, YYYY [\\at] h:mm A"
        tokens = tokenize_datetime_format(original)
        canonical = format_tokens_to_string(tokens)
        tokens2 = tokenize_datetime_format(canonical)
        assert tokens == tokens2

    def test_yyyy_as_literal_requires_multiple_escapes(self):
        # "YYYY" - Each Y can start "YY" or "YYYY" pattern, need multiple escapes
        tokens = [v("literal", "YYYY")]
        # Y at 0: starts "YYYY", escape. Y at 1: starts "YY", escape. Y at 2: starts "YY", escape. Y at 3: safe.
        assert format_tokens_to_string(tokens) == "\\Y\\Y\\YY"

    def test_pattern_at_end_of_literal_with_format_codes_in_middle(self):
        tokens = [v("literal", "Year: YYYY")]
        # "Year: YYYY" -> Y-e-a-r-:-space-Y-Y-Y-Y
        # Position 0: "Y" starts "YY"? No, "Ye" doesn't match. But "YYYY" at position 6!
        # Actually: Y at 0 doesn't start pattern (would need YY), e-a-r safe, then "a" matches!
        assert format_tokens_to_string(tokens) == "Ye\\ar: \\Y\\Y\\YY"

    def test_adjacent_literals_and_tokens(self):
        tokens = [
            v("literal", "Date is "),
            v("year4"),
            v("literal", ", time is "),
            v("hour24_2"),
        ]
        # "Date is " -> D-a-t-e-space-i-s-space
        # D at 0 starts "D" or "DD", escape. a at 1 starts "a", escape. Others safe.
        # ", time is " -> comma safe, space safe, t safe, i safe, m starts "m" or "mm", escape. Others safe.
        assert format_tokens_to_string(tokens) == "\\D\\ate i\\s YYYY, ti\\me i\\s HH"

    def test_all_token_types_round_trip_with_separators(self):
        # Note: Format tokens need separators to avoid ambiguity
        # "MMMMM" could be MMMM+M or MM+MMM, etc.
        all_tokens = [
            v("year4"),
            v("literal", " "),
            v("year2"),
            v("literal", " "),
            v("month1"),
            v("literal", " "),
            v("month2"),
            v("literal", " "),
            v("monthNameShort"),
            v("literal", " "),
            v("monthNameFull"),
            v("literal", " "),
            v("day1"),
            v("literal", " "),
            v("day2"),
            v("literal", " "),
            v("weekdayNameMin"),
            v("literal", " "),
            v("weekdayNameShort"),
            v("literal", " "),
            v("weekdayNameFull"),
            v("literal", " "),
            v("hour24_1"),
            v("literal", " "),
            v("hour24_2"),
            v("literal", " "),
            v("hour12_1"),
            v("literal", " "),
            v("hour12_2"),
            v("literal", " "),
            v("minute1"),
            v("literal", " "),
            v("minute2"),
            v("literal", " "),
            v("second1"),
            v("literal", " "),
            v("second2"),
            v("literal", " "),
            v("millisecond3"),
            v("literal", " "),
            v("ampmUpper"),
            v("literal", " "),
            v("ampmLower"),
        ]
        str_result = format_tokens_to_string(all_tokens)
        parsed = tokenize_datetime_format(str_result)
        assert parsed == all_tokens

    def test_debugging_use_case_show_interpretation(self):
        # User writes something with accidental format codes
        user_input = "Today is YYYY-MM-DD at HH:MM"
        tokens = tokenize_datetime_format(user_input)
        canonical = format_tokens_to_string(tokens)

        # The parser breaks "Today" at 'a' and "is" at 's', producing tokens that
        # stringify back to the same string (format tokens act as separators)
        assert canonical == "Today is YYYY-MM-DD at HH:MM"

        # And it round-trips correctly
        tokens_again = tokenize_datetime_format(canonical)
        assert tokens == tokens_again

        # User remembers to escape format codes
        user_input2 = "Tod\\ay i\\s YYYY-MM-DD \\at HH:MM"
        tokens2 = tokenize_datetime_format(user_input2)
        canonical2 = format_tokens_to_string(tokens2)

        # The parser breaks "Today" at 'a' and "is" at 's', producing tokens that
        # stringify back to the same string (format tokens act as separators)
        assert canonical2 == "Tod\\ay i\\s YYYY-MM-DD \\at HH:MM"

        # And it round-trips correctly
        tokens_again2 = tokenize_datetime_format(canonical2)
        assert tokens2 == tokens_again2
