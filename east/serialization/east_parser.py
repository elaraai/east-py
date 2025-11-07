"""Parser for East text format (not JSON or BEAST).

The parser is type-directed: it needs to know the target type to parse correctly.
This ensures that parsed values always match their expected types.

This module handles the East text format specifically. Other parsers:
- JSON format: east/serialization/json_parser.py (TODO)
- BEAST binary format: east/serialization/beast_parser.py (TODO)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from east.serialization.tokenizer import Token, TokenType, tokenize
from east.types.primitives import Blob, null

if TYPE_CHECKING:
    from east.types.type_system import EastType


class ParseError(Exception):
    """Error during parsing."""

    pass


class TokenStream:
    """Stream of tokens for parsing.

    Provides lookahead and position tracking.
    """

    def __init__(self, tokens: list[Token]):
        """Initialize token stream.

        Args:
            tokens: List of tokens to parse
        """
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        """Get current token without advancing.

        Returns:
            Current token
        """
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        """Peek ahead at a token.

        Args:
            offset: Number of positions to look ahead

        Returns:
            Token at offset
        """
        pos = self.pos + offset
        if pos >= len(self.tokens):
            return self.tokens[-1]  # Return EOF
        return self.tokens[pos]

    def advance(self) -> Token:
        """Advance to next token.

        Returns:
            The token that was consumed
        """
        token = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def expect(self, token_type: TokenType) -> Token:
        """Expect a specific token type and advance.

        Args:
            token_type: Expected token type

        Returns:
            The consumed token

        Raises:
            ParseError: If token type doesn't match
        """
        token = self.current()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"at line {token.line}, col {token.column}"
            )
        return self.advance()


def parse_east(target_type: EastType, text: str) -> Any:
    """Parse East text format into a value.

    Args:
        target_type: The expected type of the value
        text: East text to parse

    Returns:
        Parsed value

    Raises:
        ParseError: If parsing fails
    """
    tokens = tokenize(text)
    stream = TokenStream(tokens)
    value = parse_value(stream, target_type)

    # Should be at EOF
    if stream.current().type != TokenType.EOF:
        token = stream.current()
        raise ParseError(
            f"Unexpected token {token.type.name} at line {token.line}, col {token.column}"
        )

    return value


def parse_value(stream: TokenStream, target_type: EastType) -> Any:
    """Parse a value based on target type.

    Args:
        stream: Token stream
        target_type: Expected type

    Returns:
        Parsed value

    Raises:
        ParseError: If parsing fails
    """
    tag = target_type.tag

    if tag == "Null":
        return parse_null(stream)
    if tag == "Boolean":
        return parse_boolean(stream)
    if tag == "Integer":
        return parse_integer(stream)
    if tag == "Float":
        return parse_float(stream)
    if tag == "String":
        return parse_string(stream)
    if tag == "Blob":
        return parse_blob(stream)
    if tag == "DateTime":
        return parse_datetime(stream)
    if tag == "Array":
        return parse_array(stream, target_type)
    if tag == "Set":
        return parse_set(stream, target_type)
    if tag == "Dict":
        return parse_dict(stream, target_type)
    if tag == "Struct":
        return parse_struct(stream, target_type)
    if tag == "Variant":
        return parse_variant(stream, target_type)

    raise ParseError(f"Cannot parse type {tag}")


def parse_null(stream: TokenStream) -> Any:
    """Parse null value.

    Args:
        stream: Token stream

    Returns:
        null singleton
    """
    stream.expect(TokenType.NULL)
    return null


def parse_boolean(stream: TokenStream) -> bool:
    """Parse boolean value.

    Args:
        stream: Token stream

    Returns:
        Boolean value
    """
    token = stream.current()
    if token.type == TokenType.TRUE:
        stream.advance()
        return True
    if token.type == TokenType.FALSE:
        stream.advance()
        return False
    raise ParseError(f"Expected boolean at line {token.line}, col {token.column}")


def parse_integer(stream: TokenStream) -> int:
    """Parse integer value.

    Args:
        stream: Token stream

    Returns:
        Integer value
    """
    token = stream.expect(TokenType.INTEGER)
    return token.value


def parse_float(stream: TokenStream) -> float:
    """Parse float value.

    Args:
        stream: Token stream

    Returns:
        Float value
    """
    token = stream.expect(TokenType.FLOAT)
    return token.value


def parse_string(stream: TokenStream) -> str:
    """Parse string value.

    Args:
        stream: Token stream

    Returns:
        String value
    """
    token = stream.expect(TokenType.STRING)
    return token.value


def parse_blob(stream: TokenStream) -> Blob:
    """Parse blob value.

    Args:
        stream: Token stream

    Returns:
        Blob value
    """
    token = stream.expect(TokenType.BLOB)
    hex_str = token.value
    # Convert hex string to bytes
    if len(hex_str) == 0:
        return Blob(b"")
    return Blob(bytes.fromhex(hex_str))


def parse_datetime(stream: TokenStream) -> datetime:
    """Parse datetime value.

    Args:
        stream: Token stream

    Returns:
        DateTime value (UTC-aware)
    """
    token = stream.expect(TokenType.DATETIME)
    # Parse ISO 8601 format
    dt = datetime.fromisoformat(token.value)
    # Ensure UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    elif dt.tzinfo != UTC:
        dt = dt.astimezone(UTC)
    return dt


def parse_array(stream: TokenStream, array_type: EastType) -> Any:
    """Parse array value.

    Args:
        stream: Token stream
        array_type: Array type (with element type)

    Returns:
        EastArray instance
    """
    from east.types.containers import EastArray

    element_type = array_type.value

    stream.expect(TokenType.LBRACKET)

    items = []
    while stream.current().type != TokenType.RBRACKET:
        items.append(parse_value(stream, element_type))

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
        elif stream.current().type != TokenType.RBRACKET:
            token = stream.current()
            raise ParseError(f"Expected comma or ] at line {token.line}, col {token.column}")

    stream.expect(TokenType.RBRACKET)

    return EastArray(element_type, items)


def parse_set(stream: TokenStream, set_type: EastType) -> Any:
    """Parse set value.

    Args:
        stream: Token stream
        set_type: Set type (with element type)

    Returns:
        EastSet instance
    """
    from east.types.containers import EastSet

    element_type = set_type.value

    stream.expect(TokenType.LBRACE)

    items = []
    while stream.current().type != TokenType.RBRACE:
        items.append(parse_value(stream, element_type))

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
        elif stream.current().type != TokenType.RBRACE:
            token = stream.current()
            raise ParseError(f"Expected comma or }} at line {token.line}, col {token.column}")

    stream.expect(TokenType.RBRACE)

    return EastSet(element_type, items)


def parse_dict(stream: TokenStream, dict_type: EastType) -> Any:
    """Parse dict value.

    Args:
        stream: Token stream
        dict_type: Dict type (with key and value types)

    Returns:
        EastDict instance
    """
    from east.types.containers import EastDict

    dict_struct = dict_type.value
    key_type = dict_struct.key
    value_type = dict_struct.value

    stream.expect(TokenType.LBRACE)

    # Check for empty dict {:}
    if stream.current().type == TokenType.COLON:
        stream.advance()
        stream.expect(TokenType.RBRACE)
        return EastDict(key_type, value_type, None)

    items = {}
    while stream.current().type != TokenType.RBRACE:
        # Parse key
        key = parse_value(stream, key_type)

        # Expect colon
        stream.expect(TokenType.COLON)

        # Parse value
        val = parse_value(stream, value_type)

        items[key] = val

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
        elif stream.current().type != TokenType.RBRACE:
            token = stream.current()
            raise ParseError(f"Expected comma or }} at line {token.line}, col {token.column}")

    stream.expect(TokenType.RBRACE)

    return EastDict(key_type, value_type, items)


def parse_struct(stream: TokenStream, struct_type: EastType) -> Any:
    """Parse struct value.

    Args:
        stream: Token stream
        struct_type: Struct type (with field specs)

    Returns:
        EastStruct instance
    """
    from east.types.type_system import StructType

    field_specs = struct_type.value

    # Build runtime StructType
    fields = [(field.name, field.type) for field in field_specs]
    runtime_type = StructType(tuple(fields))

    stream.expect(TokenType.LPAREN)

    # Parse field values
    field_values = {}
    while stream.current().type != TokenType.RPAREN:
        # Parse field name
        name_token = stream.expect(TokenType.IDENTIFIER)
        field_name = name_token.value

        # Expect equals
        stream.expect(TokenType.EQUALS)

        # Find field type
        field_type = None
        for name, typ in fields:
            if name == field_name:
                field_type = typ
                break

        if field_type is None:
            raise ParseError(
                f"Unknown field '{field_name}' at line {name_token.line}, col {name_token.column}"
            )

        # Parse field value
        field_values[field_name] = parse_value(stream, field_type)

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
        elif stream.current().type != TokenType.RPAREN:
            token = stream.current()
            raise ParseError(f"Expected comma or ) at line {token.line}, col {token.column}")

    stream.expect(TokenType.RPAREN)

    return runtime_type.create(**field_values)


def parse_variant(stream: TokenStream, variant_type: EastType) -> Any:
    """Parse variant value.

    Args:
        stream: Token stream
        variant_type: Variant type (with case specs)

    Returns:
        EastVariant instance
    """
    from east.types.type_system import VariantType

    case_specs = variant_type.value

    # Parse tag
    tag_token = stream.expect(TokenType.VARIANT_TAG)
    tag = tag_token.value

    # Find case type
    case_type = None
    for case in case_specs:
        if case.name == tag:
            case_type = case.type
            break

    if case_type is None:
        raise ParseError(
            f"Unknown variant case '{tag}' at line {tag_token.line}, col {tag_token.column}"
        )

    # Parse value (if not null)
    value = null if case_type.tag == "Null" else parse_value(stream, case_type)

    # Build runtime VariantType and create instance
    cases = [(case.name, case.type) for case in case_specs]
    runtime_type = VariantType(tuple(cases))

    return runtime_type.create(tag, value)


__all__: list[str] = ["parse_east", "ParseError", "TokenStream", "parse_value"]
