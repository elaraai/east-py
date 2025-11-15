"""Tokenizer for East text format.

The tokenizer breaks East text into tokens for parsing.
Supports: null, true, false, integers, floats, strings, blobs, datetimes,
identifiers, variant tags, and delimiters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any


class TokenType(Enum):
    """Token types for East text format."""

    # Keywords
    NULL = auto()
    TRUE = auto()
    FALSE = auto()

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    BLOB = auto()
    DATETIME = auto()

    # Identifiers and tags
    IDENTIFIER = auto()
    VARIANT_TAG = auto()  # .Tag

    # Delimiters
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()  # ,
    COLON = auto()  # :
    EQUALS = auto()  # =
    AMPERSAND = auto()  # &

    # Special
    REFERENCE = auto()  # References like 3#.location
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """A token in the East text format.

    Attributes:
        type: The type of token
        value: The token value (may be None for delimiters)
        line: Line number (1-indexed)
        column: Column number (1-indexed)
        text: Original text of the token (for error messages)
    """

    type: TokenType
    value: Any
    line: int
    column: int
    text: str | None = None

    def __repr__(self) -> str:
        """Return readable representation."""
        if self.value is None:
            return f"Token({self.type.name}, line={self.line}, col={self.column})"
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class Tokenizer:
    """Tokenizer for East text format.

    Converts text into a sequence of tokens.
    """

    def __init__(self, text: str):
        """Initialize tokenizer.

        Args:
            text: The East text to tokenize
        """
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def current_char(self) -> str | None:
        """Get current character without advancing.

        Returns:
            Current character, or None if at end
        """
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def peek_char(self, offset: int = 1) -> str | None:
        """Peek ahead at a character.

        Args:
            offset: Number of positions to look ahead

        Returns:
            Character at offset, or None if past end
        """
        pos = self.pos + offset
        if pos >= len(self.text):
            return None
        return self.text[pos]

    def advance(self) -> str | None:
        """Advance to next character.

        Returns:
            The character that was consumed
        """
        if self.pos >= len(self.text):
            return None

        char = self.text[self.pos]
        self.pos += 1

        if char == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1

        return char

    def skip_whitespace(self) -> None:
        """Skip whitespace and comments."""
        while True:
            char = self.current_char()
            if char is None:
                break

            # Skip whitespace
            if char in " \t\n\r":
                self.advance()
                continue

            # Skip # comments (to end of line)
            if char == "#":
                while char is not None and char != "\n":
                    char = self.advance()
                continue

            # No more whitespace
            break

    def read_string(self) -> str:
        """Read a string literal.

        Returns:
            The string value (unescaped)
        """
        # Consume opening quote
        quote = self.advance()
        assert quote in ('"', "'")

        chars = []
        while True:
            char = self.current_char()
            if char is None:
                raise ValueError(
                    f"unterminated string (missing closing quote) at line {self.line}, col {self.column}"
                )

            if char == quote:
                self.advance()
                break

            if char == "\\":
                self.advance()
                next_char = self.current_char()
                if next_char is None:
                    raise ValueError(
                        f"unterminated string (missing closing quote) at line {self.line}, col {self.column}"
                    )

                # Escape sequences - East text format only supports \\ and \"
                if next_char == "\\":
                    chars.append("\\")
                elif next_char == quote:
                    chars.append(quote)
                elif next_char in "ntr":
                    # East text format does not support \n, \t, \r - must use actual newlines/tabs
                    raise ValueError(
                        f"unexpected escape sequence in string at line {self.line}, col {self.column}"
                    )
                else:
                    # Unknown escape, raise error
                    raise ValueError(
                        f"Unknown escape sequence \\{next_char} at line {self.line}, col {self.column}"
                    )
                self.advance()
            else:
                chars.append(char)
                self.advance()

        return "".join(chars)

    def read_number_or_datetime(self) -> Token:
        """Read a number (integer/float) or datetime.

        Also handles special float values like NaN, Infinity, -Infinity.

        Returns:
            Token for integer, float, or datetime
        """
        start_line = self.line
        start_column = self.column

        # Check for -Infinity (special case since it starts with -)
        if self.current_char() == "-":
            # Peek ahead to see if this is -Infinity
            saved_pos = self.pos
            self.advance()  # Skip '-'
            if self.text[self.pos : self.pos + 8] == "Infinity":
                # It's -Infinity
                for _ in range(8):
                    self.advance()
                return Token(TokenType.FLOAT, float("-inf"), start_line, start_column)
            # Not -Infinity, restore position and continue with normal number parsing
            self.pos = saved_pos

        # Read digits and special characters
        chars: list[str] = []
        has_datetime_separator = False  # Track if we've seen 'T' (datetime separator)
        while True:
            char = self.current_char()
            if char is None:
                break

            # Only consume ':' if we've seen a datetime separator or dash (date portion)
            if char == ":":
                if has_datetime_separator or "-" in "".join(chars):
                    chars.append(char)
                    self.advance()
                else:
                    # Not part of datetime, stop here
                    break
            elif char.isdigit() or char in "+-.TZeE":
                if char == "T":
                    has_datetime_separator = True
                chars.append(char)
                self.advance()
            else:
                break

        text = "".join(chars)

        # Check for datetime (ISO 8601 format)
        if "T" in text or (":" in text and "-" in text):
            # This looks like a datetime (has T separator, or has both : and -)
            return Token(TokenType.DATETIME, text, start_line, start_column)

        # Check for float (has decimal point or exponential notation)
        if "." in text or "e" in text or "E" in text:
            # Special float values
            if text == "NaN":
                value = float("nan")
            elif text == "Infinity":
                value = float("inf")
            else:
                # Check for incomplete exponent (e.g., "1.5e" without digits)
                if (
                    text.endswith("e")
                    or text.endswith("E")
                    or (
                        ("e+" in text or "E+" in text or "e-" in text or "E-" in text)
                        and text[-1] in "+-"
                    )
                ):
                    raise ValueError(
                        f"expected digits in float exponent at line {start_line}, col {len(text) + start_column}"
                    )
                try:
                    value = float(text)
                except ValueError:
                    # Check if it's an exponent issue
                    if "e" in text.lower():
                        raise ValueError(
                            f"expected digits in float exponent at line {start_line}, col {len(text) + start_column}"
                        ) from None
                    raise
            return Token(TokenType.FLOAT, value, start_line, start_column)

        # Integer
        value = int(text)

        # Check if this is a reference (INTEGER#path)
        if self.current_char() == "#":
            # Parse as reference
            ref_chars = [text, "#"]
            self.advance()  # Skip #

            # Read the path part, tracking bracket depth
            bracket_depth = 0
            while True:
                char = self.current_char()
                if char is None:
                    break

                # Track bracket depth to avoid consuming ] that closes outer context
                if char == "[":
                    ref_chars.append(char)
                    self.advance()
                    bracket_depth += 1
                elif char == "]":
                    if bracket_depth > 0:
                        ref_chars.append(char)
                        self.advance()
                        bracket_depth -= 1
                    else:
                        # This ] closes something outside the reference
                        break
                elif char.isalnum() or char in "._":
                    ref_chars.append(char)
                    self.advance()
                else:
                    break

            ref_str = "".join(ref_chars)
            return Token(TokenType.REFERENCE, ref_str, start_line, start_column)

        return Token(TokenType.INTEGER, value, start_line, start_column, text)

    def read_identifier_or_keyword(self) -> Token:
        """Read an identifier or keyword.

        Returns:
            Token for identifier or keyword
        """
        start_line = self.line
        start_column = self.column

        # Check for backtick-escaped identifier
        if self.current_char() == "`":
            self.advance()
            chars = []
            while True:
                char = self.current_char()
                if char is None:
                    raise ValueError(
                        f"Unterminated backtick identifier at line {self.line}, col {self.column}"
                    )
                if char == "`":
                    self.advance()
                    break
                chars.append(char)
                self.advance()
            text = "".join(chars)
            return Token(TokenType.IDENTIFIER, text, start_line, start_column, f"`{text}`")

        # Regular identifier
        chars = []
        while True:
            char = self.current_char()
            if char is None:
                break
            if char.isalnum() or char == "_":
                chars.append(char)
                self.advance()
            else:
                break

        text = "".join(chars)

        # Check for keywords
        if text == "null":
            return Token(TokenType.NULL, None, start_line, start_column, text)
        if text == "true":
            return Token(TokenType.TRUE, True, start_line, start_column, text)
        if text == "false":
            return Token(TokenType.FALSE, False, start_line, start_column, text)

        # Check for special float keywords
        if text == "NaN":
            return Token(TokenType.FLOAT, float("nan"), start_line, start_column)
        if text == "Infinity":
            return Token(TokenType.FLOAT, float("inf"), start_line, start_column)

        return Token(TokenType.IDENTIFIER, text, start_line, start_column, text)

    def read_blob(self) -> str:
        """Read a blob literal (0x...).

        Returns:
            Hex string (without 0x prefix)
        """
        # Consume 0x
        self.advance()
        self.advance()

        chars = []
        while True:
            char = self.current_char()
            if char is None:
                break
            if char in "0123456789abcdefABCDEF":
                chars.append(char)
                self.advance()
            else:
                break

        return "".join(chars)

    def tokenize(self) -> list[Token]:
        """Tokenize the entire text.

        Returns:
            List of tokens

        Raises:
            ValueError: If there's a syntax error
        """
        self.tokens = []

        while True:
            self.skip_whitespace()

            char = self.current_char()
            if char is None:
                # End of input
                self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
                break

            start_line = self.line
            start_column = self.column

            # Single-character delimiters
            if char == "[":
                self.advance()
                self.tokens.append(Token(TokenType.LBRACKET, None, start_line, start_column))
            elif char == "]":
                self.advance()
                self.tokens.append(Token(TokenType.RBRACKET, None, start_line, start_column))
            elif char == "{":
                self.advance()
                self.tokens.append(Token(TokenType.LBRACE, None, start_line, start_column))
            elif char == "}":
                self.advance()
                self.tokens.append(Token(TokenType.RBRACE, None, start_line, start_column))
            elif char == "(":
                self.advance()
                self.tokens.append(Token(TokenType.LPAREN, None, start_line, start_column))
            elif char == ")":
                self.advance()
                self.tokens.append(Token(TokenType.RPAREN, None, start_line, start_column))
            elif char == ",":
                self.advance()
                self.tokens.append(Token(TokenType.COMMA, None, start_line, start_column))
            elif char == ":":
                self.advance()
                self.tokens.append(Token(TokenType.COLON, None, start_line, start_column))
            elif char == "=":
                self.advance()
                self.tokens.append(Token(TokenType.EQUALS, None, start_line, start_column))
            elif char == "&":
                self.advance()
                self.tokens.append(Token(TokenType.AMPERSAND, None, start_line, start_column))

            # Variant tag (.Tag)
            elif char == ".":
                self.advance()
                next_char = self.current_char()
                if next_char and (next_char.isalpha() or next_char == "_"):
                    # Read tag name
                    chars = []
                    while True:
                        c = self.current_char()
                        if c is None:
                            break
                        if c.isalnum() or c == "_":
                            chars.append(c)
                            self.advance()
                        else:
                            break
                    tag_name = "".join(chars)
                    self.tokens.append(
                        Token(TokenType.VARIANT_TAG, tag_name, start_line, start_column)
                    )
                else:
                    if next_char and next_char in " \t\n\r":
                        # Report error at position of whitespace, not the dot
                        raise ValueError(
                            f"whitespace not allowed between '.' and case identifier at line {self.line}, col {self.column}"
                        )
                    raise ValueError(
                        f"Invalid variant tag at line {start_line}, col {start_column}"
                    )

            # String literals
            elif char in ('"', "'"):
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, start_line, start_column))

            # Blob literals (0x...)
            elif char == "0" and self.peek_char() in ("x", "X"):
                hex_str = self.read_blob()
                self.tokens.append(Token(TokenType.BLOB, hex_str, start_line, start_column))

            # Numbers (and maybe datetime)
            elif char.isdigit():
                token = self.read_number_or_datetime()
                self.tokens.append(token)

            # Negative number or -Infinity
            elif char == "-":
                next_char = self.peek_char()
                # Handle both -123 and -Infinity
                if next_char and (next_char.isdigit() or next_char == "I"):
                    token = self.read_number_or_datetime()
                    self.tokens.append(token)
                else:
                    raise ValueError(
                        f"Unexpected character '{char}' at line {start_line}, col {start_column}"
                    )

            # Identifiers and keywords
            elif char.isalpha() or char == "_" or char == "`":
                token = self.read_identifier_or_keyword()
                self.tokens.append(token)

            else:
                raise ValueError(
                    f"Unexpected character '{char}' at line {start_line}, col {start_column}"
                )

        return self.tokens


def tokenize(text: str) -> list[Token]:
    """Tokenize East text format.

    Args:
        text: East text to tokenize

    Returns:
        List of tokens

    Raises:
        ValueError: If there's a syntax error
    """
    tokenizer = Tokenizer(text)
    return tokenizer.tokenize()


__all__: list[str] = ["Token", "TokenType", "Tokenizer", "tokenize"]
