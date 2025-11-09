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

from east.serialization.east_printer import print_type
from east.serialization.east_tokenizer import Token, TokenType, tokenize
from east.types.primitives import Blob, null

if TYPE_CHECKING:
    from east.types.type_system import EastType


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
    from east.types.type_system import RecursiveTypeMarker

    # Helper to find all markers in a type
    def find_all_markers(t: EastType, markers: set[Any]) -> None:
        if not hasattr(t, "tag"):
            return

        tag = t.tag

        if tag == "Recursive":
            marker = t.value
            if isinstance(marker, RecursiveTypeMarker):
                markers.add(marker)
            return

        if tag in ("Array", "Set"):
            find_all_markers(t.value, markers)
            return

        if tag == "Dict":
            find_all_markers(t.value.key, markers)
            find_all_markers(t.value.value, markers)
            return

        if tag == "Struct":
            for field in t.value:
                find_all_markers(field.type, markers)
            return

        if tag == "Variant":
            for case in t.value:
                find_all_markers(case.type, markers)
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

    tokens = tokenize(text)
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
            f"unexpected token {token.type.name}",
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

    # Store in value tree if it's a mutable type
    tag = target_type.tag
    if tag in ("Array", "Set", "Dict", "Struct"):
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
    tag = target_type.tag
    token = stream.current()

    # Handle recursive types
    if tag == "Recursive":
        from east.types.type_system import RecursiveTypeMarker

        marker = target_type.value
        if isinstance(marker, RecursiveTypeMarker):
            marker_id = id(marker)
            if marker_id not in marker_map:
                raise ValueError(f"Unresolved recursive type marker: marker_id={marker_id}")
            ctx_index = marker_map[marker_id]
            resolved_type = type_ctx[ctx_index]
        else:
            raise ValueError(f"Expected RecursiveTypeMarker, got {type(marker)}")

        return parse_value(
            stream, resolved_type, type_str, value_tree, current_path, type_ctx, marker_map
        )

    if tag == "Null":
        return parse_null(stream, type_str)
    if tag == "Boolean":
        return parse_boolean(stream, type_str)
    if tag == "Integer":
        return parse_integer(stream, type_str)
    if tag == "Float":
        return parse_float(stream, type_str)
    if tag == "String":
        return parse_string(stream, type_str)
    if tag == "Blob":
        return parse_blob(stream, type_str)
    if tag == "DateTime":
        return parse_datetime(stream, type_str)
    if tag == "Array":
        return parse_array(
            stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
        )
    if tag == "Set":
        return parse_set(
            stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
        )
    if tag == "Dict":
        return parse_dict(
            stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
        )
    if tag == "Struct":
        return parse_struct(
            stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
        )
    if tag == "Variant":
        return parse_variant(
            stream, target_type, type_str, value_tree, current_path, type_ctx, marker_map
        )

    raise ParseError(
        f"cannot parse type {tag}", type_str=type_str, line=token.line, col=token.column
    )


def parse_null(stream: TokenStream, type_str: str) -> Any:
    """Parse null value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        null singleton
    """
    token = stream.current()
    try:
        stream.expect(TokenType.NULL)
        return null
    except ParseError:
        raise ParseError(
            f"expected null, got '{token.value if hasattr(token, 'value') else token.type.name}'",
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
    raise ParseError(
        f"expected boolean, got '{token.value if hasattr(token, 'value') else token.type.name}'",
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
    try:
        token = stream.expect(TokenType.INTEGER)
        return token.value
    except ParseError:
        raise ParseError(
            f"expected integer, got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None


def parse_float(stream: TokenStream, type_str: str) -> float:
    """Parse float value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        Float value
    """
    token = stream.current()
    try:
        token = stream.expect(TokenType.FLOAT)
        return token.value
    except ParseError:
        raise ParseError(
            f"expected float, got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None


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
        raise ParseError(
            f"expected string, got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None


def parse_blob(stream: TokenStream, type_str: str) -> Blob:
    """Parse blob value.

    Args:
        stream: Token stream
        type_str: Type string for error messages

    Returns:
        Blob value
    """
    token = stream.current()
    try:
        token = stream.expect(TokenType.BLOB)
        hex_str = token.value
        # Convert hex string to bytes
        if len(hex_str) == 0:
            return Blob(b"")
        return Blob(bytes.fromhex(hex_str))
    except ParseError:
        raise ParseError(
            f"expected blob, got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
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
            f"expected datetime, got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None
    except (ValueError, AttributeError) as e:
        raise ParseError(
            f"invalid datetime format: {e}", type_str=type_str, line=token.line, col=token.column
        ) from None


def parse_array(
    stream: TokenStream,
    array_type: EastType,
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
    from east.types.containers import EastArray

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
            f"expected '[', got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    items = []
    index = 0
    while stream.current().type != TokenType.RBRACKET:
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
            # Check for trailing comma
            if stream.current().type == TokenType.RBRACKET:
                token = stream.current()
                raise ParseError(
                    "trailing comma not allowed",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )
        elif stream.current().type != TokenType.RBRACKET:
            token = stream.current()
            raise ParseError(
                f"expected comma or ']', got '{token.type.name}'",
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
    set_type: EastType,
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
    from east.types.containers import EastSet

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
            f"expected '{{', got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    items = []
    index = 0
    while stream.current().type != TokenType.RBRACE:
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
            # Check for trailing comma
            if stream.current().type == TokenType.RBRACE:
                token = stream.current()
                raise ParseError(
                    "trailing comma not allowed",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )
        elif stream.current().type != TokenType.RBRACE:
            token = stream.current()
            raise ParseError(
                f"expected comma or '}}', got '{token.type.name}'",
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
    dict_type: EastType,
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
    from east.types.containers import EastDict

    dict_struct = dict_type.value
    key_type = dict_struct.key
    value_type = dict_struct.value
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
            f"expected '{{', got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Check for empty dict {:}
    if stream.current().type == TokenType.COLON:
        stream.advance()
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
        return EastDict(key_type, value_type, None)

    items = {}
    index = 0
    while stream.current().type != TokenType.RBRACE:
        # Parse key
        try:
            key = parse_value_with_tracking(
                stream, key_type, key_type_str, value_tree, current_path, type_ctx, marker_map
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f"[{index}].key" if not e.path else f"[{index}].key{e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

        # Expect colon
        token = stream.current()
        try:
            stream.expect(TokenType.COLON)
        except ParseError:
            raise ParseError(
                f"expected ':', got '{token.type.name}'",
                type_str=type_str,
                line=token.line,
                col=token.column,
            ) from None

        # Parse value
        try:
            val = parse_value_with_tracking(
                stream,
                value_type,
                value_type_str,
                value_tree,
                current_path + [f"[{index}].value"],
                type_ctx,
                marker_map,
            )
        except ParseError as e:
            # Re-raise with path and parent type context
            new_path = f"[{index}].value" if not e.path else f"[{index}].value{e.path}"
            raise ParseError(e.message, new_path, type_str, e.line, e.col) from None

        items[key] = val
        index += 1

        # Check for comma or end
        if stream.current().type == TokenType.COMMA:
            stream.advance()
            # Check for trailing comma
            if stream.current().type == TokenType.RBRACE:
                token = stream.current()
                raise ParseError(
                    "trailing comma not allowed",
                    type_str=type_str,
                    line=token.line,
                    col=token.column,
                )
        elif stream.current().type != TokenType.RBRACE:
            token = stream.current()
            raise ParseError(
                f"expected comma or '}}', got '{token.type.name}'",
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


def parse_struct(
    stream: TokenStream,
    struct_type: EastType,
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
    from east.types.type_system import _StructTypeClass

    field_specs = struct_type.value

    # Build runtime _StructTypeClass
    fields = [(field.name, field.type) for field in field_specs]
    runtime_type = _StructTypeClass(tuple(fields))

    # Register markers for field types if recursive
    for field in field_specs:
        marker = _find_recursive_marker(field.type)
        if marker is not None and id(marker) not in marker_map:
            type_ctx.append(field.type)
            marker_map[id(marker)] = len(type_ctx) - 1

    token = stream.current()
    try:
        stream.expect(TokenType.LPAREN)
    except ParseError:
        raise ParseError(
            f"expected '(', got '{token.value if hasattr(token, 'value') else token.type.name}'",
            type_str=type_str,
            line=token.line,
            col=token.column,
        ) from None

    # Parse field values
    field_values = {}
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
                f"expected '=', got '{token.type.name}'",
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
            raise ParseError(
                f"unknown field '{field_name}'",
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
            raise ParseError(
                f"expected comma or ')', got '{token.type.name}'",
                type_str=type_str,
                line=token.line,
                col=token.column,
            )

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

    try:
        return runtime_type.create(**field_values)
    except ValueError as e:
        token = stream.current()
        raise ParseError(str(e), type_str=type_str, line=token.line, col=token.column) from e


def parse_variant(
    stream: TokenStream,
    variant_type: EastType,
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
    from east.types.type_system import _VariantTypeClass

    case_specs = variant_type.value

    # Parse tag
    tag_token = stream.current()
    try:
        tag_token = stream.expect(TokenType.VARIANT_TAG)
        tag = tag_token.value
    except ParseError:
        raise ParseError(
            f"expected variant tag, got '{tag_token.type.name}'",
            type_str=type_str,
            line=tag_token.line,
            col=tag_token.column,
        ) from None

    # Find case type
    case_type = None
    for case in case_specs:
        if case.name == tag:
            case_type = case.type
            break

    if case_type is None:
        raise ParseError(
            f"unknown variant case '{tag}'",
            type_str=type_str,
            line=tag_token.line,
            col=tag_token.column,
        )

    # Register marker for case type if recursive
    marker = _find_recursive_marker(case_type)
    if marker is not None and id(marker) not in marker_map:
        type_ctx.append(case_type)
        marker_map[id(marker)] = len(type_ctx) - 1

    # Parse value
    if case_type.tag == "Null":
        # For nullary variants, optionally accept explicit "null" token
        if stream.current().type == TokenType.NULL:
            stream.advance()
        value = null
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

    # Build runtime _VariantTypeClass and create instance
    cases = [(case.name, case.type) for case in case_specs]
    runtime_type = _VariantTypeClass(tuple(cases))

    return runtime_type.create(tag, value)


__all__: list[str] = ["parse_east", "ParseError", "TokenStream", "parse_value"]
