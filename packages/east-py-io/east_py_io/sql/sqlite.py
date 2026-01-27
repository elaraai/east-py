#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""SQLite platform functions for East.

Provides SQLite database operations for East programs, including
connection management and parameterized query execution.
"""

import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from east.runtime.platform import GenericPlatformFunction, PlatformFunction
from east.types.types import NullType, StringType
from east.types.values import EastArray, EastDict, EastStruct, EastVariant, east_null

from .types import (
    ConnectionHandleType,
    SqliteConfigType,
    SqlParametersType,
    SqlParameterType,
    SqlResultType,
    SqlRowType,
)

# Register type converters for BOOLEAN and DATETIME
sqlite3.register_adapter(bool, int)
sqlite3.register_converter("BOOLEAN", lambda v: bool(int(v)))
sqlite3.register_converter(
    "DATETIME",
    lambda v: datetime.fromisoformat(v.decode().replace("Z", "+00:00")).replace(tzinfo=None),
)

# Connection storage
_connections: dict[str, sqlite3.Connection] = {}


def convert_param_to_native(param: EastVariant) -> Any:
    """Convert East SQL parameter to native Python value.

    Args:
        param: East SQL parameter variant

    Returns:
        Native Python value for SQLite binding
    """
    tag = param.type
    value = param.value

    if tag == "String":
        return value
    elif tag == "Integer":
        return int(value) if value is not None else 0  # Convert from BigInt to int
    elif tag == "Float":
        return value
    elif tag == "Boolean":
        return 1 if value else 0  # SQLite uses 0/1 for booleans
    elif tag == "Null":
        return None
    elif tag == "Blob":
        return bytes(value) if value else b""  # bytes
    elif tag == "DateTime":
        return value.isoformat() if value else ""  # Store as ISO string
    else:
        return None


def convert_native_to_param(value: Any, column_type: str | None = None) -> EastVariant:
    """Convert native Python value to East SQL parameter variant.

    SQLite preserves integer/float distinction based on stored value type.
    Python sqlite3 returns int for INTEGER values and float for REAL values.

    Args:
        value: Native Python value from SQLite
        column_type: SQLite declared column type from cursor.description

    Returns:
        East SQL parameter variant
    """
    from east.types.values import EastBlob

    if value is None:
        return EastVariant("Null", east_null)

    # Boolean - comes from BOOLEAN columns via converter
    if isinstance(value, bool):
        return EastVariant("Boolean", value)

    # Integer handling - only return Integer if column is declared as INTEGER
    # For literals like SELECT 1, there's no declared type, so return Float to match TypeScript
    if isinstance(value, int):
        if column_type and column_type.upper() == "INTEGER":
            return EastVariant("Integer", value)
        return EastVariant("Float", float(value))

    # Float
    if isinstance(value, float):
        return EastVariant("Float", value)

    # String
    if isinstance(value, str):
        return EastVariant("String", value)

    # Bytes
    if isinstance(value, bytes):
        return EastVariant("Blob", EastBlob(value))

    # Datetime - comes from DATETIME columns via converter
    if isinstance(value, datetime):
        # Ensure UTC timezone and truncate to milliseconds to match TypeScript behavior
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        # Truncate microseconds to milliseconds (keep first 3 digits of microseconds)
        ms = (value.microsecond // 1000) * 1000
        value = value.replace(microsecond=ms)
        return EastVariant("DateTime", value)

    return EastVariant("Null", east_null)


async def sqlite_connect_impl(config: EastStruct) -> str:
    """Connect to a SQLite database.

    Args:
        config: SQLite connection configuration

    Returns:
        Connection handle (opaque string)

    Raises:
        Exception: If connection fails
    """
    try:
        path = config["path"]
        read_only = False
        memory = False

        read_only_opt = config["readOnly"]
        if read_only_opt.type == "some":
            read_only = read_only_opt.value

        memory_opt = config["memory"]
        if memory_opt.type == "some":
            memory = memory_opt.value

        actual_path = ":memory:" if memory else path

        # Create connection with type detection enabled
        if read_only:
            conn = sqlite3.connect(
                f"file:{actual_path}?mode=ro",
                uri=True,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        else:
            conn = sqlite3.connect(actual_path, detect_types=sqlite3.PARSE_DECLTYPES)

        # Enable foreign keys by default
        conn.execute("PRAGMA foreign_keys = ON")

        # Generate handle
        handle = str(uuid.uuid4())
        _connections[handle] = conn

        return handle
    except Exception as e:
        raise Exception(f"SQLite connection failed: {e}") from e


async def sqlite_query_impl(handle: str, sql: str, params: EastArray) -> EastVariant:
    """Execute a SQL query with parameterized values.

    Args:
        handle: Connection handle
        sql: SQL query string
        params: Query parameters

    Returns:
        Query result variant

    Raises:
        Exception: If query fails or handle is invalid
    """
    try:
        if handle not in _connections:
            raise Exception(f"Invalid connection handle: {handle}")

        conn = _connections[handle]

        # Convert East parameters to native values
        native_params = [convert_param_to_native(p) for p in params]

        # Execute query
        cursor = conn.cursor()
        cursor.execute(sql, native_params)

        # Determine query type
        trimmed_sql = sql.strip().upper()

        if trimmed_sql.startswith("SELECT") or cursor.description:
            # SELECT query - return rows
            rows = cursor.fetchall()
            # cursor.description: (name, type_code, display_size, internal_size, precision, scale, null_ok)
            # For SQLite with PARSE_DECLTYPES, type_code is the declared type as string (or None)
            column_info = (
                [(desc[0], desc[1]) for desc in cursor.description] if cursor.description else []
            )

            # Convert rows to East format
            east_rows: EastArray = EastArray(SqlRowType, [])
            for row in rows:
                row_dict: EastDict = EastDict(StringType, SqlParameterType)
                for (col_name, col_type), value in zip(column_info, row, strict=True):
                    row_dict[col_name] = convert_native_to_param(value, col_type)
                east_rows.append(row_dict)

            return EastVariant("select", EastStruct({"rows": east_rows}))
        elif trimmed_sql.startswith("INSERT"):
            # INSERT query
            conn.commit()
            rows_affected = cursor.rowcount
            last_insert_id = cursor.lastrowid

            last_id_opt: EastVariant = (
                EastVariant("some", last_insert_id)
                if last_insert_id and last_insert_id != 0
                else EastVariant("none", None)
            )

            return EastVariant(
                "insert",
                EastStruct({"rowsAffected": rows_affected, "lastInsertId": last_id_opt}),
            )
        elif trimmed_sql.startswith("UPDATE"):
            # UPDATE query
            conn.commit()
            rows_affected = cursor.rowcount

            return EastVariant("update", EastStruct({"rowsAffected": rows_affected}))
        elif trimmed_sql.startswith("DELETE"):
            # DELETE query
            conn.commit()
            rows_affected = cursor.rowcount

            return EastVariant("delete", EastStruct({"rowsAffected": rows_affected}))
        else:
            # Other queries (CREATE, DROP, etc.) - treat as update
            conn.commit()
            rows_affected = cursor.rowcount

            return EastVariant("update", EastStruct({"rowsAffected": rows_affected}))
    except Exception as e:
        raise Exception(f"SQLite query failed: {e}") from e


async def sqlite_close_impl(handle: str) -> None:
    """Close a SQLite database connection.

    Args:
        handle: Connection handle

    Raises:
        Exception: If handle is invalid
    """
    try:
        if handle not in _connections:
            raise Exception(f"Invalid connection handle: {handle}")

        conn = _connections[handle]
        conn.close()
        del _connections[handle]
    except Exception as e:
        raise Exception(f"SQLite close failed: {e}") from e


async def sqlite_close_all_impl() -> None:
    """Close all SQLite connections.

    Useful for test cleanup.
    """
    for conn in _connections.values():
        conn.close()
    _connections.clear()


# SQLite type categories for type validation
_SQLITE_INTEGER_TYPES = {
    "INTEGER",
    "INT",
    "TINYINT",
    "SMALLINT",
    "MEDIUMINT",
    "BIGINT",
    "UNSIGNED BIG INT",
    "INT2",
    "INT8",
}
_SQLITE_FLOAT_TYPES = {
    "REAL",
    "DOUBLE",
    "DOUBLE PRECISION",
    "FLOAT",
    "NUMERIC",
    "DECIMAL",
}
_SQLITE_TEXT_TYPES = {
    "TEXT",
    "CHARACTER",
    "VARCHAR",
    "VARYING CHARACTER",
    "NCHAR",
    "NATIVE CHARACTER",
    "NVARCHAR",
    "CLOB",
    "DATE",
}


def _get_sqlite_east_type(col_type: str | None) -> str:
    """Get expected East type from SQLite column type."""
    if col_type is None:
        return "Float"  # Default for literals

    col_upper = col_type.upper()

    if col_upper in _SQLITE_INTEGER_TYPES:
        return "Integer"
    elif col_upper in _SQLITE_FLOAT_TYPES:
        return "Float"
    elif col_upper in _SQLITE_TEXT_TYPES:
        return "String"
    elif col_upper == "DATETIME":
        return "DateTime"
    elif col_upper == "BLOB":
        return "Blob"
    elif col_upper == "BOOLEAN":
        return "Boolean"
    else:
        return "Float"  # Unknown - default to Float


def _is_option_type(field_type: dict, base_type: str) -> bool:
    """Check if field_type is OptionType(base_type)."""
    if field_type.get("type") != "Option":
        return False
    inner = field_type.get("value")
    return inner is not None and inner.get("type") == base_type


def _is_matching_type(field_type: dict, base_type: str) -> bool:
    """Check if field_type matches base_type."""
    return field_type.get("type") == base_type


def _convert_sqlite_select_value(value: Any, col_type: str | None) -> Any:
    """Convert SQLite value to East value for select results."""
    from east.types.values import EastBlob

    if col_type is None:
        return value

    col_upper = col_type.upper()

    if col_upper in _SQLITE_INTEGER_TYPES:
        return int(value)
    elif col_upper == "DATETIME":
        if isinstance(value, str):
            from datetime import UTC

            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
        return value
    elif col_upper == "BLOB" and isinstance(value, bytes):
        return EastBlob(value)
    else:
        return value


def sqlite_select_factory(row_type: Any) -> Any:
    """Factory for sqlite_select that captures the type parameter.

    Args:
        row_type: Row type parameter (East IR type format)

    Returns:
        Async implementation function for sqlite_select
    """

    async def sqlite_select_impl(handle: str, sql: str, params: EastArray) -> EastArray:
        """Execute a SELECT query with typed results.

        Args:
            handle: Connection handle
            sql: SQL SELECT query string
            params: Query parameters

        Returns:
            Array of rows matching the type parameter T

        Raises:
            Exception: If query fails or types don't match
        """
        try:
            if handle not in _connections:
                raise Exception(f"Invalid connection handle: {handle}")

            conn = _connections[handle]

            # Convert East parameters to native values
            native_params = [convert_param_to_native(p) for p in params]

            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql, native_params)

            # Verify this is a SELECT query
            if not cursor.description:
                raise Exception(
                    "sqlite_select only supports SELECT queries. "
                    "Use sqlite_query for INSERT/UPDATE/DELETE."
                )

            # Get column metadata
            column_info = [(desc[0], desc[1]) for desc in cursor.description]

            # Validate row type T is a Struct
            if row_type.get("type") != "Struct":
                raise Exception(f"Expected row type must be a Struct, got {row_type.get('type')}")

            # Build column type map
            column_types = dict(column_info)

            # Validate field types match columns
            fields = row_type.get("value", [])
            field_info: dict[str, dict[str, Any]] = {}

            for field in fields:
                field_name = field["name"]
                field_type = field["type"]

                if field_name not in column_types:
                    raise Exception(f"Column '{field_name}' not found in query result")

                col_type = column_types[field_name]
                expected_east = _get_sqlite_east_type(col_type)

                # Check if field type matches expected type or OptionType(expected)
                is_option = _is_option_type(field_type, expected_east)
                is_base = _is_matching_type(field_type, expected_east)

                if not is_base and not is_option:
                    raise Exception(
                        f"Type mismatch for column '{field_name}': SQLite column is {col_type}, "
                        f"expected {expected_east} or OptionType({expected_east})"
                    )

                field_info[field_name] = {
                    "is_option": is_option,
                    "col_type": col_type,
                }

            # Fetch and convert rows
            raw_rows = cursor.fetchall()
            rows: list[EastStruct] = []

            for row_idx, raw_row in enumerate(raw_rows):
                converted: dict[str, Any] = {}
                for (col_name, _col_type), value in zip(column_info, raw_row, strict=True):
                    info = field_info.get(col_name)
                    if info is None:
                        continue  # Column not in expected type

                    if value is None:
                        if info["is_option"]:
                            converted[col_name] = EastVariant("none", None)
                        else:
                            raise Exception(
                                f"null value at row[{row_idx}] for required field '{col_name}' - "
                                f"use OptionType for nullable columns"
                            )
                    else:
                        # Convert based on column type
                        converted_value = _convert_sqlite_select_value(value, info["col_type"])

                        if info["is_option"]:
                            converted[col_name] = EastVariant("some", converted_value)
                        else:
                            converted[col_name] = converted_value

                rows.append(EastStruct(converted))

            return EastArray(row_type, rows)
        except Exception as e:
            raise Exception(f"SQLite select failed: {e}") from e

    return sqlite_select_impl


# Platform function implementations
sqlite_impl = [
    PlatformFunction(
        name="sqlite_connect",
        inputs=[SqliteConfigType],
        output=ConnectionHandleType,
        type="async",
        fn=sqlite_connect_impl,
    ),
    PlatformFunction(
        name="sqlite_query",
        inputs=[ConnectionHandleType, StringType, SqlParametersType],
        output=SqlResultType,
        type="async",
        fn=sqlite_query_impl,
    ),
    PlatformFunction(
        name="sqlite_close",
        inputs=[ConnectionHandleType],
        output=NullType,
        type="async",
        fn=sqlite_close_impl,
    ),
    PlatformFunction(
        name="sqlite_close_all",
        inputs=[],
        output=NullType,
        type="async",
        fn=sqlite_close_all_impl,
    ),
    GenericPlatformFunction(
        name="sqlite_select",
        type_parameters=["T"],
        type="async",
        fn=sqlite_select_factory,
    ),
]


__all__ = [
    "sqlite_impl",
    "SqliteConfigType",
    "ConnectionHandleType",
    "SqlParametersType",
    "SqlResultType",
    "SqlParameterType",
    "SqlRowType",
]
