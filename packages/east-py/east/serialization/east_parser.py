#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Parser for East text format (not JSON or BEAST).

The parser is type-directed: it needs to know the target type to parse correctly.
This ensures that parsed values always match their expected types.

This module handles the East text format specifically. Other parsers:
- JSON format: east/serialization/json_parser.py (TODO)
- BEAST binary format: east/serialization/beast_parser.py (TODO)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np_

from east.serialization.east_printer import print_type
from east.serialization.east_tokenizer import Token, TokenType, tokenize
from east.types.types import (
    ArrayTypeAlias,
    DictTypeAlias,
    EastType,
    RefTypeAlias,
    SetTypeAlias,
    StructTypeAlias,
    VariantTypeAlias,
    is_array_type,
    is_blob_type,
    is_boolean_type,
    is_datetime_type,
    is_dict_type,
    is_float_type,
    is_integer_type,
    is_matrix_type,
    is_null_type,
    is_recursive_type,
    is_ref_type,
    is_set_type,
    is_string_type,
    is_struct_type,
    is_variant_type,
    is_vector_type,
)
from east.types.values import (
    EAST_ELEMENT_TO_DTYPE,
    EastArray,
    EastBlob,
    EastDict,
    EastMatrix,
    EastSet,
    EastStruct,
    EastVariant,
    EastVector,
    east_null,
    east_ref,
)


class ParseError(Exception):
    """Error during parsing East text format."""

    def __init__(
        self, message: str, path: str = "", type_str: str = "", line: int = 1, col: int = 1
    ):
        """Initialize parse error.

        Args:
            message: Error message (the "because" part)
            path: Path where error occurred (e.g., "[0].field")
            type_str: String representation of the type being parsed
            line: Line number where error occurred
            col: Column number where error occurred
        """
        super().__init__(message)
        self.message = message
        self.path = path
        self.type_str = type_str
        self.line = line
        self.col = col

    def __str__(self) -> str:
        """Return error message in TypeScript-compatible format."""
        # Format: "Error occurred because <message> at <path> (line X, col Y) while parsing value of type "<type>""
        result = f"Error occurred because {self.message}"
        if self.path:
            result += f" at {self.path}"
        result += f" (line {self.line}, col {self.col})"
        if self.type_str:
            result += f' while parsing value of type "{self.type_str}"'
        return result


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


# =============================================================================
# Aliasing support for circular references
# =============================================================================


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    """Find the length of the common prefix between two path arrays.

    Args:
        a: First path array
        b: Second path array

    Returns:
        Length of common prefix
    """
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return i


def _encode_relative_ref(current_path: list[str], target_path: list[str]) -> str:
    """Compute a relative reference string from currentPath to targetPath.

    Returns a string like "2#.foo[0]" or "1#"

    The format is: "upLevels#remaining_path_components"

    Args:
        current_path: Current location in the value tree
        target_path: Target location we're referencing

    Returns:
        Relative reference string
    """
    common_len = _common_prefix_length(current_path, target_path)
    up_levels = len(current_path) - common_len
    remaining = target_path[common_len:]

    if not remaining:
        return f"{up_levels}#"

    remaining_str = "".join(remaining)
    return f"{up_levels}#{remaining_str}"


def _decode_relative_ref(ref_str: str, current_path: list[str]) -> list[str]:
    """Decode a relative reference string and return the target path array.

    Input like "2#.foo[0]" returns the target path array.
    Input like "1#" returns the target path array.

    Args:
        ref_str: Relative reference string (e.g., "2#.foo[0]")
        current_path: Current location in the value tree

    Returns:
        Target path array

    Raises:
        ValueError: If reference is invalid
    """
    hash_idx = ref_str.find("#")
    if hash_idx == -1:
        raise ValueError(f"Invalid relative reference: {ref_str}")

    up_level_str = ref_str[:hash_idx]
    remaining_str = ref_str[hash_idx + 1 :]

    try:
        up_levels = int(up_level_str)
    except ValueError as e:
        raise ValueError(f"Invalid relative reference: {ref_str}") from e

    if up_levels < 0 or up_levels > len(current_path):
        raise ValueError(
            f"Invalid relative reference: going up {up_levels} levels "
            f"from depth {len(current_path)}"
        )

    # Build target path
    target_path = current_path[: len(current_path) - up_levels]

    # Add remaining components if any
    if remaining_str:
        # Parse the remaining punctuated path
        # Format: .field[0][key] etc.
        pos = 0
        while pos < len(remaining_str):
            if remaining_str[pos] == ".":
                # Identifier follows
                pos += 1
                end = pos
                while end < len(remaining_str) and (
                    remaining_str[end].isalnum() or remaining_str[end] == "_"
                ):
                    end += 1
                target_path.append(f".{remaining_str[pos:end]}")
                pos = end
            elif remaining_str[pos] == "[":
                # Bracket expression
                end = pos + 1
                depth = 1
                while end < len(remaining_str) and depth > 0:
                    if remaining_str[end] == "[":
                        depth += 1
                    elif remaining_str[end] == "]":
                        depth -= 1
                    end += 1
                target_path.append(remaining_str[pos:end])
                pos = end
            else:
                pos += 1

    return target_path


# =============================================================================
# Main parsing functions
# =============================================================================


def _find_recursive_marker(typ: EastType) -> Any | None:
    """Find the RecursiveTypeMarker that this type owns (if any).

    For a Struct/Variant type created with recursive_type(), this returns the marker
    by checking if any Recursive refs in the type point back to this type as their node.

    Args:
        typ: The type to search

    Returns:
        The RecursiveTypeMarker if found, None otherwise
    """
    from east.types.types import RecursiveTypeMarker

    # Helper to find all markers in a type
    def find_all_markers(t: EastType, markers: set[Any]) -> None:
        if not hasattr(t, "type"):
            return

        if is_recursive_type(t):
            marker = t.value
            if isinstance(marker, RecursiveTypeMarker):
                markers.add(marker)
            return

        if is_array_type(t):
            find_all_markers(t.value, markers)
            return
        if is_set_type(t):
            find_all_markers(t.value, markers)
            return

        if is_dict_type(t):
            find_all_markers(t.value["key"], markers)
            find_all_markers(t.value["value"], markers)
            return

        if is_struct_type(t):
            for field in t.value:
                find_all_markers(field["type"], markers)
            return

        if is_variant_type(t):
            for case in t.value:
                find_all_markers(case["type"], markers)
            return

    # Find all markers referenced in this type
    markers: set[Any] = set()
    find_all_markers(typ, markers)

    # Check if any marker's node points to this type (object identity)
    for marker in markers:
        if hasattr(marker, "node") and marker.node is typ:
            return marker

    return None


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
    # Generate type string for error messages
    type_str = print_type(target_type)

    try:
        tokens = tokenize(text)
    except ValueError as e:
        # Convert tokenizer errors to ParseError
        # For DateTime types, any tokenization error means invalid format
        # (TypeScript has no tokenizer - parseDateTime works directly on string)
        if is_datetime_type(target_type):
            raise ParseError(
                "expected DateTime in format YYYY-MM-DDTHH:MM:SS.sss",
                type_str=type_str,
                line=1,
                col=1,
            ) from None

        # For other types, preserve the specific error with line/col
        error_msg = str(e)
        # Try to parse "... at line X, col Y"
        import re

        match = re.search(r"at line (\d+), col (\d+)", error_msg)
        if match:
            line = int(match.group(1))
            col = int(match.group(2))
            # Extract the main message before " at line"
            message = error_msg.split(" at line")[0]
            raise ParseError(message, type_str=type_str, line=line, col=col) from None
        # Fallback if we can't parse line/col
        raise ParseError(error_msg, type_str=type_str, line=1, col=1) from None

    stream = TokenStream(tokens)

    # Track parsed values for reference resolution
    value_tree: dict[str, Any] = {}
    current_path: list[str] = []

    # Track type context for recursive types
    type_ctx: list[EastType] = []
    marker_map: dict[Any, int] = {}

    # Find and register the marker for the top-level type
    marker = _find_recursive_marker(target_type)
    if marker is not None:
        type_ctx.append(target_type)
        marker_map[id(marker)] = 0

    value = parse_value_with_tracking(
        stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
    )

    # Should be at EOF
    if stream.current().type != TokenType.EOF:
        token = stream.current()
        raise ParseError(
            "unexpected input after parsed value",
            type_str=type_str,
            line=token.line,
            col=token.column,
        )

    return value


def parse_value_with_tracking(
    stream: TokenStream,
    target_type: EastType,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse a value with reference tracking.

    Args:
        stream: Token stream
        target_type: Expected type
        type_str: String representation of type for error messages
        value_tree: Dictionary tracking all parsed values by path
        current_path: Current path in the value tree
        type_ctx: Stack of recursive types
        marker_map: Maps marker IDs to indices in type_ctx

    Returns:
        Parsed value

    Raises:
        ParseError: If parsing fails
    """
    token = stream.current()

    # Check if this is a reference
    if token.type == TokenType.REFERENCE:
        # Decode and resolve reference
        ref_str = token.value
        target_path = _decode_relative_ref(ref_str, current_path)
        path_key = "".join(target_path)

        if path_key not in value_tree:
            raise ParseError(
                f"unresolved reference '{ref_str}'",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )

        stream.advance()
        return value_tree[path_key]

    # Parse the value normally
    value = parse_value(
        stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
    )

    # Store in value tree if it's a mutable type (or Ref which can be aliased)
    if (
        is_array_type(target_type)
        or is_set_type(target_type)
        or is_dict_type(target_type)
        or is_struct_type(target_type)
        or is_ref_type(target_type)
    ):
        path_key = "".join(current_path)
        value_tree[path_key] = value

    return value


def parse_value(
    stream: TokenStream,
    target_type: EastType,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse a value based on target type.

    Args:
        stream: Token stream
        target_type: Expected type
        type_str: String representation of type for error messages
        value_tree: Dictionary tracking all parsed values by path
        current_path: Current path in the value tree
        type_ctx: Stack of recursive types
        marker_map: Maps marker IDs to indices in type_ctx

    Returns:
        Parsed value

    Raises:
        ParseError: If parsing fails
    """
    token = stream.current()

    # Handle recursive types
    if is_recursive_type(target_type):
        from east.types.types import RecursiveTypeMarker

        marker = target_type.value
        if isinstance(marker, RecursiveTypeMarker):
            marker_id = id(marker)
            if marker_id not in marker_map:
                raise ValueError(f"Unresolved recursive type marker: marker_id={marker_id}")
            ctx_index = marker_map[marker_id]
            resolved_type = type_ctx[ctx_index]
        elif isinstance(marker, int):
            # Integer scope_id from TypeScript exports
            ctx_index = len(type_ctx) - marker
            if ctx_index < 0 or ctx_index >= len(type_ctx):
                raise ValueError(
                    f"Invalid recursive scope_id {marker} (ctx len={len(type_ctx)}, calculated index={ctx_index})"
                )
            resolved_type = type_ctx[ctx_index]
        else:
            raise ValueError(f"Expected RecursiveTypeMarker or int, got {type(marker)}")

        return parse_value(
            stream, resolved_type, type_str, value_tree, current_path, type_ctx, marker_map
        )

    if is_null_type(target_type):
        return parse_null(stream, type_str)
    if is_boolean_type(target_type):
        return parse_boolean(stream, type_str)
    if is_integer_type(target_type):
        return parse_integer(stream, type_str)
    if is_float_type(target_type):
        return parse_float(stream, type_str)
    if is_string_type(target_type):
        return parse_string(stream, type_str)
    if is_blob_type(target_type):
        return parse_blob(stream, type_str)
    if is_datetime_type(target_type):
        return parse_datetime(stream, type_str)
    if is_vector_type(target_type):
        return parse_vector(stream, target_type, type_str)
    if is_matrix_type(target_type):
        return parse_matrix(stream, target_type, type_str)
    if is_array_type(target_type):
        # Push array type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_array(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if is_set_type(target_type):
        # Push set type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_set(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if is_dict_type(target_type):
        # Push dict type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_dict(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if is_ref_type(target_type):
        # Push ref type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_ref(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if is_struct_type(target_type):
        # Push struct type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_struct(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()
    if is_variant_type(target_type):
        # Push variant type onto context stack
        from east.serialization.east_printer import _find_recursive_marker

        type_ctx.append(target_type)
        marker = _find_recursive_marker(target_type)
        if marker is not None and id(marker) not in marker_map:
            marker_map[id(marker)] = len(type_ctx) - 1
        try:
            return parse_variant(
                stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
            )
        finally:
            type_ctx.pop()

    raise ParseError(
        f"cannot parse type {target_type}", type_str=type_str, line=token.line, col=token.column
    )


def parse_null(stream: TokenStream, type_str: str) -> Any:
    """Parse east_null value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        east_null singleton
    """
    token = stream.current()
    try:
        stream.expect(TokenType.NULL)
        return east_null
    except ParseError:
        # Show first character of the token text for error message (matches TypeScript)
        if token.type == TokenType.EOF:
            got = "end of input"
        elif token.type == TokenType.STRING:
            got = '"'
        elif token.text:
            got = token.text[0]
        elif hasattr(token, "value") and token.value is not None:
            got = str(token.value)
        else:
            got = token.type.name
        raise ParseError(
            f"expected null, got {got}" if got == "end of input" else f"expected null, got '{got}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None


def parse_boolean(stream: TokenStream, type_str: str) -> bool:
    """Parse boolean value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

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
    # Show first character of the token text for error message (matches TypeScript)
    if token.type == TokenType.EOF:
        got = "end of input"
    elif token.type == TokenType.STRING:
        got = '"'
    elif token.text:
        got = token.text[0]
    elif hasattr(token, "value") and token.value is not None:
        got = str(token.value)
    else:
        got = token.type.name
    raise ParseError(
        f"expected boolean, got {got}"
        if got == "end of input"
        else f"expected boolean, got '{got}'",
        type_str=type_str,
        line=token.line,
        col=token.column,
    )


def parse_integer(stream: TokenStream, type_str: str) -> int:
    """Parse integer value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        Integer value
    """
    token = stream.current()
    if token.type == TokenType.INTEGER:
        stream.advance()
        value = token.value
        # Check for 64-bit signed integer range: -2^63 to 2^63-1
        if value < -(2**63) or value > 2**63 - 1:
            raise ParseError(
                f"integer out of range (must be 64-bit signed), got {token.text or value}",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )
        return value
    # Show first character of the token for error message (matches TypeScript)
    if token.type == TokenType.EOF:
        got = "end of input"
    elif token.type == TokenType.STRING:
        got = '"'  # Strings start with quote
    elif token.text:
        got = token.text[0]
    elif hasattr(token, "value") and token.value is not None:
        got = str(token.value)
    else:
        got = token.type.name
    raise ParseError(
        f"expected integer, got {got}"
        if got == "end of input"
        else f"expected integer, got '{got}'",
        type_str=type_str,
        line=token.line,
        col=token.column,
    )


def parse_float(stream: TokenStream, type_str: str) -> float:
    """Parse float value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        Float value
    """
    token = stream.current()

    # Accept both FLOAT and INTEGER tokens (convert int to float)
    if token.type == TokenType.FLOAT:
        stream.advance()
        return token.value
    if token.type == TokenType.INTEGER:
        stream.advance()
        return float(token.value)
    # Show first character of the token for error message (matches TypeScript)
    if token.type == TokenType.EOF:
        got = "end of input"
    elif token.type == TokenType.STRING:
        got = '"'
    elif token.text:
        got = token.text[0]
    elif hasattr(token, "value") and token.value is not None:
        got = str(token.value)
    else:
        got = token.type.name
    raise ParseError(
        f"expected float, got {got}" if got == "end of input" else f"expected float, got '{got}'",
        type_str=type_str,
        line=token.line,
        col=token.column,
    )


def parse_string(stream: TokenStream, type_str: str) -> str:
    """Parse string value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        String value
    """
    token = stream.current()
    try:
        token = stream.expect(TokenType.STRING)
        return token.value
    except ParseError:
        # Show first character of the token text for error message (matches TypeScript)
        if token.type == TokenType.EOF:
            got = "end of input"
        elif token.type == TokenType.RBRACKET:
            got = "]"
        elif token.type == TokenType.RBRACE:
            got = "}"
        elif token.type == TokenType.RPAREN:
            got = ")"
        elif token.type == TokenType.STRING:
            got = '"'
        elif token.text:
            got = token.text[0]
        elif hasattr(token, "value") and token.value is not None:
            got = str(token.value)
        else:
            got = token.type.name

        # Format error message - don't quote "end of input"
        if got == "end of input":
            error_msg = f"expected '\"', got {got}"
        else:
            error_msg = f"expected '\"', got '{got}'"
        raise ParseError(
            error_msg,
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None


def parse_blob(stream: TokenStream, type_str: str) -> EastBlob:
    """Parse blob value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        EastBlob value
    """
    token = stream.current()

    # Check for blob token (starts with 0x)
    if token.type != TokenType.BLOB:
        raise ParseError(
            "expected Blob starting with 0x",
            type_str=type_str,
            line=1,
            col=1,
        )

    stream.advance()
    hex_str = token.value

    # Convert hex string to bytes
    if len(hex_str) == 0:
        return EastBlob(b"")

    # Check for odd length before calling fromhex()
    if len(hex_str) % 2 != 0:
        raise ParseError(
            f'invalid hex string (odd length), got "0x{hex_str}"',
            type_str=type_str,
            line=1,
            col=1,
        )

    try:
        return EastBlob(bytes.fromhex(hex_str))
    except ValueError:
        # Invalid hex characters
        raise ParseError(
            f'invalid hex string, got "0x{hex_str}"',
            type_str=type_str,
            line=1,
            col=1,
        ) from None


def parse_datetime(stream: TokenStream, type_str: str) -> datetime:
    """Parse datetime value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        DateTime value (UTC-aware)
    """
    token = stream.current()
    try:
        token = stream.expect(TokenType.DATETIME)
        # Parse ISO 8601 format
        dt = datetime.fromisoformat(token.value)
        # Ensure UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        elif dt.tzinfo != UTC:
            dt = dt.astimezone(UTC)
        return dt
    except ParseError:
        raise ParseError(
            "expected DateTime in format YYYY-MM-DDTHH:MM:SS.sss",
            type_str=type_str,
            line=1,
            col=1,
        ) from None
    except (ValueError, AttributeError):
        # Invalid datetime values (e.g., month 13, hour 25)
        # Match TypeScript's simple error message
        raise ParseError(
            f'invalid DateTime value, got "{token.value}"', type_str=type_str, line=1, col=1
        ) from None


def parse_vector(
    stream: TokenStream,
    vector_type: EastType,
    type_str: str,
) -> EastVector:
    """Parse vector value (vec[...]).

    Args:
        stream: Token stream
        vector_type: Vector type
        type_str: Type string for error messages

    Returns:
        EastVector instance
    """
    element_type = vector_type.value
    dtype = EAST_ELEMENT_TO_DTYPE[element_type.type]

    token = stream.current()
    if token.type != TokenType.IDENTIFIER or token.value != "vec":
        raise ParseError(
            "expected 'vec' to start vector",
            type_str=type_str,
            line=token.line,
            col=token.column,
        )
    stream.advance()

    token = stream.current()
    try:
        stream.expect(TokenType.LBRACKET)
    except ParseError:
        raise ParseError(
            "expected '[' after 'vec'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    items = []

    if stream.current().type != TokenType.RBRACKET:
        while True:
            token = stream.current()
            if element_type.type == "Boolean":
                if token.type == TokenType.TRUE:
                    items.append(True)
                    stream.advance()
                elif token.type == TokenType.FALSE:
                    items.append(False)
                    stream.advance()
                else:
                    raise ParseError(
                        "expected boolean element in vector",
                        type_str=type_str,
                        line=token.line,
                        col=token.column,
                    )
            elif element_type.type == "Float":
                if token.type in (TokenType.FLOAT, TokenType.INTEGER):
                    items.append(float(token.value))
                    stream.advance()
                else:
                    raise ParseError(
                        "expected float element in vector",
                        type_str=type_str,
                        line=token.line,
                        col=token.column,
                    )
            elif element_type.type == "Integer":
                if token.type == TokenType.INTEGER:
                    items.append(token.value)
                    stream.advance()
                else:
                    raise ParseError(
                        "expected integer element in vector",
                        type_str=type_str,
                        line=token.line,
                        col=token.column,
                    )

            if stream.current().type == TokenType.COMMA:
                stream.advance()
            elif stream.current().type == TokenType.RBRACKET:
                break
            else:
                token = stream.current()
                raise ParseError(
                    "expected ',' or ']' in vector",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )

    try:
        stream.expect(TokenType.RBRACKET)
    except ParseError:
        token = stream.current()
        raise ParseError(
            "expected ']' to close vector",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    return EastVector(element_type, np_.array(items, dtype=dtype))


def parse_matrix(
    stream: TokenStream,
    matrix_type: EastType,
    type_str: str,
) -> EastMatrix:
    """Parse matrix value (mat[[...], [...]]).

    Args:
        stream: Token stream
        matrix_type: Matrix type
        type_str: Type string for error messages

    Returns:
        EastMatrix instance
    """
    element_type = matrix_type.value
    dtype = EAST_ELEMENT_TO_DTYPE[element_type.type]

    token = stream.current()
    if token.type != TokenType.IDENTIFIER or token.value != "mat":
        raise ParseError(
            "expected 'mat' to start matrix",
            type_str=type_str,
            line=token.line,
            col=token.column,
        )
    stream.advance()

    token = stream.current()
    try:
        stream.expect(TokenType.LBRACKET)
    except ParseError:
        raise ParseError(
            "expected '[' after 'mat'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    rows_data = []
    expected_cols = None

    if stream.current().type != TokenType.RBRACKET:
        while True:
            # Parse a row: [elem, elem, ...]
            token = stream.current()
            try:
                stream.expect(TokenType.LBRACKET)
            except ParseError:
                raise ParseError(
                    "expected '[' to start matrix row",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                ) from None

            row_items = []
            if stream.current().type != TokenType.RBRACKET:
                while True:
                    token = stream.current()
                    if element_type.type == "Boolean":
                        if token.type == TokenType.TRUE:
                            row_items.append(True)
                            stream.advance()
                        elif token.type == TokenType.FALSE:
                            row_items.append(False)
                            stream.advance()
                        else:
                            raise ParseError(
                                "expected boolean element in matrix",
                                type_str=type_str,
                                line=token.line,
                                col=token.column,
                            )
                    elif element_type.type == "Float":
                        if token.type in (TokenType.FLOAT, TokenType.INTEGER):
                            row_items.append(float(token.value))
                            stream.advance()
                        else:
                            raise ParseError(
                                "expected float element in matrix",
                                type_str=type_str,
                                line=token.line,
                                col=token.column,
                            )
                    elif element_type.type == "Integer":
                        if token.type == TokenType.INTEGER:
                            row_items.append(token.value)
                            stream.advance()
                        else:
                            raise ParseError(
                                "expected integer element in matrix",
                                type_str=type_str,
                                line=token.line,
                                col=token.column,
                            )

                    if stream.current().type == TokenType.COMMA:
                        stream.advance()
                    elif stream.current().type == TokenType.RBRACKET:
                        break
                    else:
                        token = stream.current()
                        raise ParseError(
                            "expected ',' or ']' in matrix row",
                            type_str=type_str,
                            line=token.line,
                            col=token.column,
                        )

            try:
                stream.expect(TokenType.RBRACKET)
            except ParseError:
                token = stream.current()
                raise ParseError(
                    "expected ']' to close matrix row",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                ) from None

            if expected_cols is None:
                expected_cols = len(row_items)
            elif len(row_items) != expected_cols:
                raise ParseError(
                    f"jagged matrix: row 0 has {expected_cols} columns but row {len(rows_data)} has {len(row_items)}",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )

            rows_data.append(row_items)

            if stream.current().type == TokenType.COMMA:
                stream.advance()
            elif stream.current().type == TokenType.RBRACKET:
                break
            else:
                token = stream.current()
                raise ParseError(
                    "expected ',' or ']' after matrix row",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )

    try:
        stream.expect(TokenType.RBRACKET)
    except ParseError:
        token = stream.current()
        raise ParseError(
            "expected ']' to close matrix",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    if not rows_data:
        return EastMatrix(element_type, np_.empty((0, 0), dtype=dtype))

    data = np_.array(rows_data, dtype=dtype)
    return EastMatrix(element_type, data, len(rows_data), expected_cols)


def parse_array(
    stream: TokenStream,
    array_type: ArrayTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse array value.

    Args:
        stream: Token stream
        array_type: Array type (with element type)
        type_str: Type string for error messages
        value_tree: Dictionary tracking all parsed values by path
        current_path: Current path in the value tree
        type_ctx: Stack of recursive types
        marker_map: Maps marker IDs to indices in type_ctx

    Returns:
        EastArray instance
    """
    element_type = array_type.value
    element_type_str = print_type(element_type)

    # Register marker for element type if it's recursive
    marker = _find_recursive_marker(element_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(element_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.LBRACKET)
    except ParseError:
        raise ParseError(
            "expected '[' to start array",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    items = []
    index = 0

    # Check for empty array
    if stream.current().type == TokenType.RBRACKET:
        pass  # Empty array, will be handled below
    else:
        # Parse first element
        while True:
            element_path = current_path + [f"[{index}]"]
            try:
                items.append(
                    parse_value_with_tracking(
                        stream,
                        element_type,
                        element_type_str,
                        value_tree,
                        element_path,
                        type_ctx,
                        marker_map,
                    )
                )
            except ParseError as e:
                # Re-raise with path and parent type context
                new_path = f"[{index}]" if not e.path else f"[{index}]{e.path}"
                raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

            index += 1

            # Check for comma or end
            if stream.current().type == TokenType.COMMA:
                stream.advance()
                # After comma, loop continues to parse next element (even if it's ']', which will error)
            elif stream.current().type == TokenType.RBRACKET:
                break  # End of array
            else:
                token = stream.current()
                raise ParseError(
                    "expected ',' or ']' after array element",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )

    try:
        stream.expect(TokenType.RBRACKET)
    except ParseError:
        token = stream.current()
        raise ParseError(
            f"expected ']', got '{token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    return EastArray(element_type, items)


def parse_set(
    stream: TokenStream,
    set_type: SetTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse set value.

    Args:
        stream: Token stream
        set_type: Set type (with element type)
        type_str: Type string for error messages

    Returns:
        EastSet instance
    """
    element_type = set_type.value
    element_type_str = print_type(element_type)

    # Register marker for element type if it\'s recursive
    marker = _find_recursive_marker(element_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(element_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.LBRACE)
    except ParseError:
        raise ParseError(
            "expected '{' to start set",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    items = []
    index = 0

    # Check for empty set
    if stream.current().type == TokenType.RBRACE:
        pass  # Empty set, will be handled below
    else:
        # Parse first element
        while True:
            try:
                items.append(
                    parse_value_with_tracking(
                        stream,
                        element_type,
                        element_type_str,
                        value_tree,
                        current_path + [f"[{index}]"],
                        type_ctx,
                        marker_map,
                    )
                )
            except ParseError as e:
                # Re-raise with path and parent type context
                new_path = f"[{index}]" if not e.path else f"[{index}]{e.path}"
                raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

            index += 1

            # Check for comma or end
            if stream.current().type == TokenType.COMMA:
                stream.advance()
                # After comma, loop continues to parse next element (even if it's '}', which will error)
            elif stream.current().type == TokenType.RBRACE:
                break  # End of set
            else:
                token = stream.current()
                raise ParseError(
                    "expected ',' or '}' after set element",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )

    try:
        stream.expect(TokenType.RBRACE)
    except ParseError:
        token = stream.current()
        raise ParseError(
            f"expected '}}', got '{token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    return EastSet(element_type, items)


def parse_dict(
    stream: TokenStream,
    dict_type: DictTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse dict value.

    Args:
        stream: Token stream
        dict_type: Dict type (with key and value types)
        type_str: Type string for error messages

    Returns:
        EastDict instance
    """
    dict_struct = dict_type.value
    key_type = dict_struct["key"]
    value_type = dict_struct["value"]
    key_type_str = print_type(key_type)
    value_type_str = print_type(value_type)

    # Register markers for key and value types if recursive
    key_marker = _find_recursive_marker(key_type)
    if key_marker is not None and id(key_marker) not in marker_map:
        type_ctx.append(key_type)
        marker_map[id(key_marker)] = len(type_ctx) - 1
    value_marker = _find_recursive_marker(value_type)
    if value_marker is not None and id(value_marker) not in marker_map:
        type_ctx.append(value_type)
        marker_map[id(value_marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.LBRACE)
    except ParseError:
        raise ParseError(
            "expected '{' to start dict",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Check for empty dict {} or {:}
    if stream.current().type == TokenType.RBRACE:
        stream.advance()
        return EastDict(key_type, value_type, None)

    if stream.current().type == TokenType.COLON:
        stream.advance()
        try:
            stream.expect(TokenType.RBRACE)
        except ParseError:
            token = stream.current()
            raise ParseError(
                "expected '}' after ':' in empty dict",
                type_str=type_str,
                line=token.line,
                col=token.column,
            ) from None
        return EastDict(key_type, value_type, None)

    items = {}
    index = 0

    # Parse entries
    while True:
        # Parse key
        try:
            key = parse_value_with_tracking(
                stream, key_type, key_type_str, value_tree, current_path, type_ctx, marker_map
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f"[{index}](key)" if not e.path else f"[{index}](key){e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

        # Expect colon
        token = stream.current()
        try:
            stream.expect(TokenType.COLON)
        except ParseError:
            raise ParseError(
                f"expected ':' after dict key at entry {index}",
                type_str=type_str,
                line=token.line,
                col=token.column,
            ) from None

        # Parse value
        # Use the printed key string for the path (matches TypeScript)
        from east.serialization.east_printer import print_east

        key_str = print_east(key, key_type)
        try:
            val = parse_value_with_tracking(
                stream,
                value_type,
                value_type_str,
                value_tree,
                current_path + [f"[{key_str}]"],
                type_ctx,
                marker_map,
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f"[{key_str}]" if not e.path else f"[{key_str}]{e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

        items[key] = val
        index += 1

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
            # After comma, loop continues to parse next entry (even if it's '}', which will error)
        elif stream.current().type == TokenType.RBRACE:
            break  # End of dict
        else:
            token = stream.current()
            raise ParseError(
                "expected ',' or '}' after dict entry",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )

    try:
        stream.expect(TokenType.RBRACE)
    except ParseError:
        token = stream.current()
        raise ParseError(
            f"expected '}}', got '{token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    return EastDict(key_type, value_type, items)


def parse_ref(
    stream: TokenStream,
    ref_type: RefTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse east_ref value.

    Args:
        stream: Token stream
        ref_type: EastRef type (with inner value type)
        type_str: Type string for error messages
        value_tree: Value tree for aliasing
        current_path: Current path
        type_ctx: Type context for recursive types
        marker_map: Marker map for recursive types

    Returns:
        EastRef instance
    """
    inner_type = ref_type.value
    inner_type_str = print_type(inner_type)

    # Register marker for inner type if recursive
    inner_marker = _find_recursive_marker(inner_type)
    if inner_marker is not None and id(inner_marker) not in marker_map:
        type_ctx.append(inner_type)
        marker_map[id(inner_marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.AMPERSAND)
    except ParseError:
        raise ParseError(
            f"expected '&', got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Parse inner value
    try:
        inner_value = parse_value_with_tracking(
            stream,
            inner_type,
            inner_type_str,
            value_tree,
            current_path + ["&"],
            type_ctx,
            marker_map,
        )
    except ParseError as e:
        # Re-raise with path and parent type context
        new_path = f"&{e.path}" if e.path else "&"
        raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

    return east_ref(inner_value)


def parse_struct(
    stream: TokenStream,
    struct_type: StructTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse struct value.

    Args:
        stream: Token stream
        struct_type: Struct type (with field specs)
        type_str: Type string for error messages

    Returns:
        EastStruct instance
    """
    field_specs = struct_type.value

    fields = [(field["name"], field["type"]) for field in field_specs]

    # Register markers for field types if recursive
    for field in field_specs:
        marker = _find_recursive_marker(field["type"])
        if marker is not None and id(marker) not in marker_map:
            type_ctx.append(field["type"])
            marker_map[id(marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.LPAREN)
    except ParseError:
        raise ParseError(
            "expected '(' to start struct",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Parse field values
    field_values: dict[str, Any] = {}
    while stream.current().type != TokenType.RPAREN:
        # Parse field name
        name_token = stream.current()
        try:
            name_token = stream.expect(TokenType.IDENTIFIER)
            field_name = name_token.value
        except ParseError:
            raise ParseError(
                f"expected field name, got '{name_token.type.name}'",
                type_str=type_str,
                line=name_token.line,
                col=name_token.column,
            ) from None

        # Expect equals
        token = stream.current()
        try:
            stream.expect(TokenType.EQUALS)
        except ParseError:
            raise ParseError(
                f"expected '=' after field name '{field_name}'",
                type_str=type_str,
                line=token.line,
                col=token.column,
            ) from None

        # Find field type
        field_type = None
        for name, typ in fields:
            if name == field_name:
                field_type = typ
                break

        if field_type is None:
            # Unknown field
            if field_values:
                # Already parsed some fields - expect struct to close
                raise ParseError(
                    "expected ')' to close struct",
                    type_str=type_str,
                    line=name_token.line,
                    col=name_token.column,
                )
            # First field is unknown - show list of expected fields
            expected_fields = ", ".join(name for name, _ in fields)
            raise ParseError(
                f"unknown field '{field_name}', expected one of: {expected_fields}",
                type_str=type_str,
                line=name_token.line,
                col=name_token.column,
            )

        # Parse field value
        field_type_str = print_type(field_type)
        try:
            field_values[field_name] = parse_value_with_tracking(
                stream,
                field_type,
                field_type_str,
                value_tree,
                current_path + [f".{field_name}"],
                type_ctx,
                marker_map,
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f".{field_name}" if not e.path else f".{field_name}{e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
        elif stream.current().type != TokenType.RPAREN:
            token = stream.current()
            if token.type == TokenType.EOF:
                raise ParseError(
                    "unexpected end of input in struct",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )
            raise ParseError(
                "expected ',' or ')' after struct field",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )

    # Capture token position before consuming RPAREN (for error reporting)
    rparen_token = stream.current()

    try:
        stream.expect(TokenType.RPAREN)
    except ParseError:
        token = stream.current()
        raise ParseError(
            f"expected ')', got '{token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Check for missing required fields
    required_fields = {name for name, _ in fields}
    provided_fields = set(field_values.keys())
    missing_fields = required_fields - provided_fields

    if missing_fields:
        # Report the first missing field (matches TypeScript behavior)
        missing_field = sorted(missing_fields)[0]  # Sort for determinism
        # Use rparen_token position (before advancing past RPAREN)
        raise ParseError(
            f"missing required field '{missing_field}'",
            type_str=type_str,
            line=rparen_token.line,
            col=rparen_token.column,
        )

    try:
        return EastStruct(field_values)
    except ValueError as e:
        token = stream.current()
        raise ParseError(str(e), type_str=type_str, line=token.line, col=token.column) from e


def parse_variant(
    stream: TokenStream,
    variant_type: VariantTypeAlias,
    type_str: str,
    value_tree: dict[str, Any],
    current_path: list[str],
    type_ctx: list[EastType],
    marker_map: dict[Any, int],
) -> Any:
    """Parse variant value.

    Args:
        stream: Token stream
        variant_type: Variant type (with case specs)
        type_str: Type string for error messages

    Returns:
        EastVariant instance
    """
    case_specs = variant_type.value

    # Parse tag
    tag_token = stream.current()
    try:
        tag_token = stream.expect(TokenType.VARIANT_TAG)
        tag = tag_token.value
    except ParseError:
        raise ParseError(
            "expected '.' to start variant case",
            type_str=type_str,
            line=tag_token.line,
            col=tag_token.column,
        ) from None

    # Find case type
    case_type = None
    for case in case_specs:
        if case["name"] == tag:
            case_type = case["type"]
            break

    if case_type is None:
        # Build list of expected cases
        expected_cases = ", ".join(f".{case['name']}" for case in case_specs)
        raise ParseError(
            f"unknown variant case .{tag}, expected one of: {expected_cases}",
            type_str=type_str,
            line=tag_token.line,
            col=tag_token.column + 1,  # Point to tag name, not the dot
        )

    # Register marker for case type if recursive
    marker = _find_recursive_marker(case_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(case_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    # Parse value
    if is_null_type(case_type):
        # For nullary variants, optionally accept explicit "null" token
        token = stream.current()
        if token.type == TokenType.NULL:
            stream.advance()
        elif token.type not in (
            TokenType.EOF,
            TokenType.COMMA,
            TokenType.RBRACKET,
            TokenType.RBRACE,
            TokenType.RPAREN,
        ):
            # Unexpected token after nullary variant - should be end of value or delimiter
            # Show first character of the token for error message
            if token.type == TokenType.STRING:
                got = '"'
            elif token.text:
                got = token.text[0]
            elif hasattr(token, "value") and token.value is not None:
                got = str(token.value)
            else:
                got = token.type.name
            raise ParseError(
                f"expected null, got '{got}'",
                path=f".{tag}",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )
        value = east_null
    else:
        case_type_str = print_type(case_type)
        try:
            value = parse_value_with_tracking(
                stream, case_type, case_type_str, value_tree, current_path, type_ctx, marker_map
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f".{tag}" if not e.path else f".{tag}{e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

    return EastVariant(tag, value)


__all__: list[str] = ["parse_east", "ParseError", "TokenStream", "parse_value"]
