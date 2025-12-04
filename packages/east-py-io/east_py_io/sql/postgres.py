"""PostgreSQL platform functions for East.

Provides PostgreSQL database operations for East programs, including
connection pooling and parameterized query execution.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
from east.runtime.platform import PlatformFunction
from east.types.types import NullType, StringType
from east.types.values import EastArray, EastDict, EastStruct, EastVariant, east_null

from .types import (
    ConnectionHandleType,
    PostgresConfigType,
    SqlParametersType,
    SqlParameterType,
    SqlResultType,
    SqlRowType,
)

# Connection pool storage
_pools: dict[str, asyncpg.Pool] = {}


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
        # Strip timezone info to avoid asyncpg comparison issues
        if value is not None and hasattr(value, "tzinfo") and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value
    else:
        return None


def convert_native_to_param(value: Any) -> EastVariant:
    """Convert native Python value to East SQL parameter variant."""
    from east.types.values import EastBlob

    if value is None:
        return EastVariant("Null", east_null)
    elif isinstance(value, bool):
        return EastVariant("Boolean", value)
    elif isinstance(value, int):
        return EastVariant("Integer", value)
    elif isinstance(value, float):
        return EastVariant("Float", value)
    elif isinstance(value, str):
        return EastVariant("String", value)
    elif isinstance(value, bytes):
        return EastVariant("Blob", EastBlob(value))
    elif isinstance(value, datetime):
        # Ensure UTC timezone and truncate to milliseconds to match TypeScript behavior
        value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        # Truncate microseconds to milliseconds (keep first 3 digits of microseconds)
        ms = (value.microsecond // 1000) * 1000
        value = value.replace(microsecond=ms)
        return EastVariant("DateTime", value)
    else:
        return EastVariant("Null", east_null)


async def postgres_connect_impl(config: EastStruct) -> str:
    """Connect to a PostgreSQL database.

    Creates a connection pool and returns a handle.
    """
    try:
        host = config["host"]
        port = int(config["port"])
        database = config["database"]
        user = config["user"]
        password = config["password"]

        ssl_opt = config["ssl"]
        ssl = ssl_opt.value if ssl_opt.type == "some" else False

        max_conn_opt = config["maxConnections"]
        max_connections = int(max_conn_opt.value) if max_conn_opt.type == "some" else 10

        # Create connection pool
        pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            ssl=ssl if ssl else None,
            min_size=1,
            max_size=max_connections,
        )

        if pool is None:
            raise Exception("Failed to create connection pool")

        # Generate handle
        handle = str(uuid.uuid4())
        _pools[handle] = pool

        return handle
    except Exception as e:
        raise Exception(f"PostgreSQL connection failed: {e}") from e


async def postgres_query_impl(handle: str, sql: str, params: EastArray) -> EastVariant:
    """Execute a SQL query with parameterized values."""
    try:
        if handle not in _pools:
            raise Exception(f"Invalid connection handle: {handle}")

        pool = _pools[handle]

        # Convert East parameters to native values
        native_params = [convert_param_to_native(p) for p in params]

        # Determine query type
        trimmed_sql = sql.strip().upper()

        async with pool.acquire() as conn:
            if trimmed_sql.startswith("SELECT"):
                # SELECT query - return rows
                rows = await conn.fetch(sql, *native_params)

                # Convert rows to East format
                east_rows = EastArray(SqlRowType, [])
                for row in rows:
                    row_dict = EastDict(StringType, SqlParameterType)
                    for key, value in row.items():
                        row_dict[key] = convert_native_to_param(value)
                    east_rows.append(row_dict)

                return EastVariant("select", EastStruct({"rows": east_rows}))
            elif trimmed_sql.startswith("INSERT"):
                # INSERT query
                result = await conn.execute(sql, *native_params)
                # Parse result like "INSERT 0 1"
                parts = result.split()
                rows_affected = int(parts[-1]) if len(parts) >= 2 else 0

                return EastVariant(
                    "insert",
                    EastStruct(
                        {"rowsAffected": rows_affected, "lastInsertId": EastVariant("none", None)}
                    ),
                )
            elif trimmed_sql.startswith("UPDATE"):
                # UPDATE query
                result = await conn.execute(sql, *native_params)
                parts = result.split()
                rows_affected = int(parts[-1]) if len(parts) >= 2 else 0

                return EastVariant("update", EastStruct({"rowsAffected": rows_affected}))
            elif trimmed_sql.startswith("DELETE"):
                # DELETE query
                result = await conn.execute(sql, *native_params)
                parts = result.split()
                rows_affected = int(parts[-1]) if len(parts) >= 2 else 0

                return EastVariant("delete", EastStruct({"rowsAffected": rows_affected}))
            else:
                # Other queries (CREATE, DROP, etc.)
                await conn.execute(sql, *native_params)
                return EastVariant("update", EastStruct({"rowsAffected": 0}))
    except Exception as e:
        raise Exception(f"PostgreSQL query failed: {e}") from e


async def postgres_close_impl(handle: str) -> None:
    """Close a PostgreSQL connection pool."""
    try:
        if handle not in _pools:
            raise Exception(f"Invalid connection handle: {handle}")

        pool = _pools[handle]
        await pool.close()
        del _pools[handle]
    except Exception as e:
        raise Exception(f"PostgreSQL close failed: {e}") from e


async def postgres_close_all_impl() -> None:
    """Close all PostgreSQL connection pools."""
    for pool in _pools.values():
        await pool.close()
    _pools.clear()


# Platform function implementations
postgres_impl = [
    PlatformFunction(
        name="postgres_connect",
        inputs=[PostgresConfigType],
        output=ConnectionHandleType,
        type="async",
        fn=postgres_connect_impl,
    ),
    PlatformFunction(
        name="postgres_query",
        inputs=[ConnectionHandleType, StringType, SqlParametersType],
        output=SqlResultType,
        type="async",
        fn=postgres_query_impl,
    ),
    PlatformFunction(
        name="postgres_close",
        inputs=[ConnectionHandleType],
        output=NullType,
        type="async",
        fn=postgres_close_impl,
    ),
    PlatformFunction(
        name="postgres_close_all",
        inputs=[],
        output=NullType,
        type="async",
        fn=postgres_close_all_impl,
    ),
]

__all__ = [
    "postgres_impl",
    "postgres_connect_impl",
    "postgres_query_impl",
    "postgres_close_impl",
    "postgres_close_all_impl",
]
