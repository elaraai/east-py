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

from east.runtime.platform import PlatformFunction
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
