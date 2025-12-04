"""MySQL platform functions for East.

Provides MySQL database operations for East programs, including
connection pooling and parameterized query execution.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import aiomysql
from east.runtime.platform import PlatformFunction
from east.types.types import NullType, StringType
from east.types.values import EastArray, EastDict, EastStruct, EastVariant, east_null

from .types import (
    ConnectionHandleType,
    MySqlConfigType,
    SqlParametersType,
    SqlParameterType,
    SqlResultType,
    SqlRowType,
)

# Connection pool storage
_pools: dict[str, aiomysql.Pool] = {}


def convert_param_to_native(param: EastVariant) -> Any:
    """Convert East SQL parameter to native Python value."""
    tag = param.type
    value = param.value

    if tag == "String":
        return value
    elif tag == "Integer":
        return int(value) if value is not None else 0
    elif tag == "Float" or tag == "Boolean":
        return value
    elif tag == "Null":
        return None
    elif tag == "Blob":
        return bytes(value) if value else b""
    elif tag == "DateTime":
        return value
    else:
        return None


# MySQL field type constants
MYSQL_TINY = 1  # TINYINT - used as BOOL
MYSQL_SHORT = 2
MYSQL_LONG = 3
MYSQL_FLOAT = 4
MYSQL_DOUBLE = 5
MYSQL_TIMESTAMP = 7
MYSQL_LONGLONG = 8
MYSQL_INT24 = 9
MYSQL_DATE = 10
MYSQL_TIME = 11
MYSQL_DATETIME = 12
MYSQL_YEAR = 13
MYSQL_BIT = 16
MYSQL_NEWDECIMAL = 246
MYSQL_BLOB = 252
MYSQL_VARCHAR = 253
MYSQL_STRING = 254


def convert_native_to_param(value: Any, field_type: int | None = None) -> EastVariant:
    """Convert native Python value to East SQL parameter variant.

    Args:
        value: Native Python value from MySQL
        field_type: MySQL field type code from cursor.description

    Returns:
        East SQL parameter variant
    """
    from east.types.values import EastBlob

    if value is None:
        return EastVariant("Null", east_null)

    # Boolean handling - TINYINT(1) and BIT are booleans
    if isinstance(value, bool) or (
        field_type in (MYSQL_TINY, MYSQL_BIT) and isinstance(value, int | float)
    ):
        return EastVariant("Boolean", bool(value))

    # Integer handling
    if isinstance(value, int):
        return EastVariant("Integer", value)

    # Float handling
    if isinstance(value, float):
        return EastVariant("Float", value)

    # String handling
    if isinstance(value, str):
        return EastVariant("String", value)

    # Bytes handling
    if isinstance(value, bytes):
        return EastVariant("Blob", EastBlob(value))

    # DateTime handling
    if isinstance(value, datetime):
        # Ensure UTC timezone and truncate to milliseconds to match TypeScript behavior
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        # Truncate microseconds to milliseconds (keep first 3 digits of microseconds)
        ms = (value.microsecond // 1000) * 1000
        value = value.replace(microsecond=ms)
        return EastVariant("DateTime", value)

    return EastVariant("Null", east_null)


def _convert_placeholders(sql: str) -> str:
    """Convert ? placeholders to %s for aiomysql.

    Handles quoted strings properly to avoid replacing ? inside strings.
    """
    result = []
    in_single_quote = False
    in_double_quote = False
    i = 0

    while i < len(sql):
        char = sql[i]

        # Handle escape sequences
        if char == "\\" and i + 1 < len(sql):
            result.append(char)
            result.append(sql[i + 1])
            i += 2
            continue

        # Toggle quote states
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        # Replace ? with %s only outside of quotes
        if char == "?" and not in_single_quote and not in_double_quote:
            result.append("%s")
        else:
            result.append(char)

        i += 1

    return "".join(result)


async def mysql_connect_impl(config: EastStruct) -> str:
    """Connect to a MySQL database.

    Creates a connection pool and returns a handle.
    """
    try:
        host = config["host"]
        port = int(config["port"])
        database = config["database"]
        user = config["user"]
        password = config["password"]

        max_conn_opt = config["maxConnections"]
        max_connections = int(max_conn_opt.value) if max_conn_opt.type == "some" else 10

        # Create connection pool
        pool = await aiomysql.create_pool(
            host=host,
            port=port,
            db=database,
            user=user,
            password=password,
            minsize=1,
            maxsize=max_connections,
            autocommit=True,
        )

        # Generate handle
        handle = str(uuid.uuid4())
        _pools[handle] = pool

        return handle
    except Exception as e:
        raise Exception(f"MySQL connection failed: {e}") from e


async def mysql_query_impl(handle: str, sql: str, params: EastArray) -> EastVariant:
    """Execute a SQL query with parameterized values."""
    try:
        if handle not in _pools:
            raise Exception(f"Invalid connection handle: {handle}")

        pool = _pools[handle]

        # Convert East parameters to native values
        native_params = tuple(convert_param_to_native(p) for p in params)

        # Convert ? placeholders to %s for aiomysql
        # Be careful not to replace ? inside quoted strings
        converted_sql = _convert_placeholders(sql)

        # Determine query type
        trimmed_sql = sql.strip().upper()

        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(converted_sql, native_params)

            if trimmed_sql.startswith("SELECT") or cursor.description:
                # SELECT query - return rows
                rows = await cursor.fetchall()

                # Build field type map from cursor.description
                # cursor.description is tuple of (name, type_code, display_size, internal_size, precision, scale, null_ok)
                field_type_map: dict[str, int | None] = {}
                if cursor.description:
                    for desc in cursor.description:
                        field_type_map[desc[0]] = desc[1]

                # Convert rows to East format
                east_rows = EastArray(SqlRowType, [])
                for row in rows:
                    row_dict = EastDict(StringType, SqlParameterType)
                    for key, value in row.items():
                        field_type = field_type_map.get(key)
                        row_dict[key] = convert_native_to_param(value, field_type)
                    east_rows.append(row_dict)

                return EastVariant("select", EastStruct({"rows": east_rows}))
            elif trimmed_sql.startswith("INSERT"):
                # INSERT query
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
                rows_affected = cursor.rowcount
                return EastVariant("update", EastStruct({"rowsAffected": rows_affected}))
            elif trimmed_sql.startswith("DELETE"):
                # DELETE query
                rows_affected = cursor.rowcount
                return EastVariant("delete", EastStruct({"rowsAffected": rows_affected}))
            else:
                # Other queries (CREATE, DROP, etc.)
                rows_affected = cursor.rowcount
                return EastVariant("update", EastStruct({"rowsAffected": rows_affected}))
    except Exception as e:
        raise Exception(f"MySQL query failed: {e}") from e


async def mysql_close_impl(handle: str) -> None:
    """Close a MySQL connection pool."""
    try:
        if handle not in _pools:
            raise Exception(f"Invalid connection handle: {handle}")

        pool = _pools[handle]
        pool.close()
        await pool.wait_closed()
        del _pools[handle]
    except Exception as e:
        raise Exception(f"MySQL close failed: {e}") from e


async def mysql_close_all_impl() -> None:
    """Close all MySQL connection pools."""
    for pool in _pools.values():
        pool.close()
        await pool.wait_closed()
    _pools.clear()


# Platform function implementations
mysql_impl = [
    PlatformFunction(
        name="mysql_connect",
        inputs=[MySqlConfigType],
        output=ConnectionHandleType,
        type="async",
        fn=mysql_connect_impl,
    ),
    PlatformFunction(
        name="mysql_query",
        inputs=[ConnectionHandleType, StringType, SqlParametersType],
        output=SqlResultType,
        type="async",
        fn=mysql_query_impl,
    ),
    PlatformFunction(
        name="mysql_close",
        inputs=[ConnectionHandleType],
        output=NullType,
        type="async",
        fn=mysql_close_impl,
    ),
    PlatformFunction(
        name="mysql_close_all",
        inputs=[],
        output=NullType,
        type="async",
        fn=mysql_close_all_impl,
    ),
]

__all__ = [
    "mysql_impl",
    "mysql_connect_impl",
    "mysql_query_impl",
    "mysql_close_impl",
    "mysql_close_all_impl",
]
