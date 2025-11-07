"""Tests for East tokenizer."""

import math

import pytest

from east.serialization.tokenizer import TokenType, tokenize


class TestKeywords:
    """Tests for keyword tokenization."""

    def test_null(self):
        """Tokenize null."""
        tokens = tokenize("null")
        assert len(tokens) == 2  # null + EOF
        assert tokens[0].type == TokenType.NULL
        assert tokens[0].value is None

    def test_true(self):
        """Tokenize true."""
        tokens = tokenize("true")
        assert tokens[0].type == TokenType.TRUE
        assert tokens[0].value is True

    def test_false(self):
        """Tokenize false."""
        tokens = tokenize("false")
        assert tokens[0].type == TokenType.FALSE
        assert tokens[0].value is False


class TestNumbers:
    """Tests for number tokenization."""

    def test_integer(self):
        """Tokenize integers."""
        tokens = tokenize("42")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 42

    def test_negative_integer(self):
        """Tokenize negative integer."""
        tokens = tokenize("-123")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == -123

    def test_zero(self):
        """Tokenize zero."""
        tokens = tokenize("0")
        assert tokens[0].type == TokenType.INTEGER
        assert tokens[0].value == 0

    def test_float(self):
        """Tokenize float."""
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == 3.14

    def test_negative_float(self):
        """Tokenize negative float."""
        tokens = tokenize("-2.5")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == -2.5

    def test_nan(self):
        """Tokenize NaN."""
        tokens = tokenize("NaN")
        assert tokens[0].type == TokenType.FLOAT
        assert math.isnan(tokens[0].value)

    def test_infinity(self):
        """Tokenize Infinity."""
        tokens = tokenize("Infinity")
        assert tokens[0].type == TokenType.FLOAT
        assert tokens[0].value == float("inf")


class TestStrings:
    """Tests for string tokenization."""

    def test_double_quotes(self):
        """Tokenize double-quoted string."""
        tokens = tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "hello"

    def test_single_quotes(self):
        """Tokenize single-quoted string."""
        tokens = tokenize("'world'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "world"

    def test_empty_string(self):
        """Tokenize empty string."""
        tokens = tokenize('""')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == ""

    def test_escaped_quote(self):
        """Tokenize string with escaped quote."""
        tokens = tokenize(r'"say \"hi\""')
        assert tokens[0].value == 'say "hi"'

    def test_escaped_backslash(self):
        """Tokenize string with escaped backslash."""
        tokens = tokenize(r'"path\\to\\file"')
        assert tokens[0].value == r"path\to\file"

    def test_escaped_newline(self):
        """Tokenize string with escaped newline."""
        tokens = tokenize(r'"line1\nline2"')
        assert tokens[0].value == "line1\nline2"


class TestBlob:
    """Tests for blob tokenization."""

    def test_empty_blob(self):
        """Tokenize empty blob."""
        tokens = tokenize("0x")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == ""

    def test_blob_lowercase(self):
        """Tokenize blob with lowercase hex."""
        tokens = tokenize("0xabcd")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "abcd"

    def test_blob_uppercase(self):
        """Tokenize blob with uppercase hex."""
        tokens = tokenize("0xABCD")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "ABCD"

    def test_blob_mixed_case(self):
        """Tokenize blob with mixed case hex."""
        tokens = tokenize("0x12AbCd")
        assert tokens[0].type == TokenType.BLOB
        assert tokens[0].value == "12AbCd"


class TestIdentifiers:
    """Tests for identifier tokenization."""

    def test_simple_identifier(self):
        """Tokenize simple identifier."""
        tokens = tokenize("foo")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "foo"

    def test_identifier_with_underscore(self):
        """Tokenize identifier with underscore."""
        tokens = tokenize("foo_bar")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "foo_bar"

    def test_identifier_with_numbers(self):
        """Tokenize identifier with numbers."""
        tokens = tokenize("var123")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "var123"

    def test_backtick_identifier(self):
        """Tokenize backtick-escaped identifier."""
        tokens = tokenize("`my-special-name`")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "my-special-name"


class TestVariantTags:
    """Tests for variant tag tokenization."""

    def test_simple_tag(self):
        """Tokenize simple variant tag."""
        tokens = tokenize(".Some")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "Some"

    def test_tag_with_underscore(self):
        """Tokenize tag with underscore."""
        tokens = tokenize(".My_Tag")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "My_Tag"


class TestDelimiters:
    """Tests for delimiter tokenization."""

    def test_brackets(self):
        """Tokenize brackets."""
        tokens = tokenize("[]")
        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[1].type == TokenType.RBRACKET

    def test_braces(self):
        """Tokenize braces."""
        tokens = tokenize("{}")
        assert tokens[0].type == TokenType.LBRACE
        assert tokens[1].type == TokenType.RBRACE

    def test_parens(self):
        """Tokenize parentheses."""
        tokens = tokenize("()")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.RPAREN

    def test_comma(self):
        """Tokenize comma."""
        tokens = tokenize(",")
        assert tokens[0].type == TokenType.COMMA

    def test_colon(self):
        """Tokenize colon."""
        tokens = tokenize(":")
        assert tokens[0].type == TokenType.COLON

    def test_equals(self):
        """Tokenize equals."""
        tokens = tokenize("=")
        assert tokens[0].type == TokenType.EQUALS


class TestWhitespace:
    """Tests for whitespace handling."""

    def test_spaces(self):
        """Whitespace is skipped."""
        tokens = tokenize("  42  ")
        assert len(tokens) == 2  # number + EOF
        assert tokens[0].type == TokenType.INTEGER

    def test_newlines(self):
        """Newlines are skipped."""
        tokens = tokenize("42\n\n43")
        assert len(tokens) == 3  # num + num + EOF
        assert tokens[0].value == 42
        assert tokens[1].value == 43

    def test_tabs(self):
        """Tabs are skipped."""
        tokens = tokenize("\t42\t")
        assert len(tokens) == 2
        assert tokens[0].value == 42


class TestComments:
    """Tests for comment handling."""

    def test_comment(self):
        """Comments are skipped."""
        tokens = tokenize("# this is a comment\n42")
        assert len(tokens) == 2  # number + EOF
        assert tokens[0].value == 42

    def test_comment_at_end(self):
        """Comment at end of line."""
        tokens = tokenize("42 # comment")
        assert len(tokens) == 2
        assert tokens[0].value == 42


class TestPositionTracking:
    """Tests for line/column position tracking."""

    def test_line_tracking(self):
        """Track line numbers."""
        tokens = tokenize("42\n\n43")
        assert tokens[0].line == 1
        assert tokens[1].line == 3

    def test_column_tracking(self):
        """Track column numbers."""
        tokens = tokenize("  42")
        assert tokens[0].column == 3  # After two spaces


class TestComplexExamples:
    """Tests for complex token sequences."""

    def test_array(self):
        """Tokenize array."""
        tokens = tokenize("[1, 2, 3]")
        assert tokens[0].type == TokenType.LBRACKET
        assert tokens[1].type == TokenType.INTEGER
        assert tokens[2].type == TokenType.COMMA
        assert tokens[3].type == TokenType.INTEGER
        assert tokens[4].type == TokenType.COMMA
        assert tokens[5].type == TokenType.INTEGER
        assert tokens[6].type == TokenType.RBRACKET

    def test_struct(self):
        """Tokenize struct."""
        tokens = tokenize("(name='Alice', age=30)")
        assert tokens[0].type == TokenType.LPAREN
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "name"
        assert tokens[2].type == TokenType.EQUALS
        assert tokens[3].type == TokenType.STRING
        assert tokens[3].value == "Alice"

    def test_variant(self):
        """Tokenize variant."""
        tokens = tokenize(".Some 42")
        assert tokens[0].type == TokenType.VARIANT_TAG
        assert tokens[0].value == "Some"
        assert tokens[1].type == TokenType.INTEGER
        assert tokens[1].value == 42


class TestErrors:
    """Tests for error handling."""

    def test_unterminated_string(self):
        """Unterminated string raises error."""
        with pytest.raises(ValueError, match="Unterminated string"):
            tokenize('"hello')

    def test_invalid_variant_tag(self):
        """Invalid variant tag raises error."""
        with pytest.raises(ValueError, match="Invalid variant tag"):
            tokenize(". ")

    def test_unexpected_character(self):
        """Unexpected character raises error."""
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("@")
