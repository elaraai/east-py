"""MySQL platform functions for East.

Provides MySQL database operations for East programs, including
connection pooling and parameterized query execution.
"""

import uuid
from datetime import datetime
from typing import Any

import aiomysql
from east.runtime.platform import PlatformFunction
from east.types.types import NullType, StringType
from east.types.values import EastArray, EastDict, EastStruct, EastVariant

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


def convert_native_to_param(value: Any) -> EastVariant:
    """Convert native Python value to East SQL parameter variant."""
    if value is None:
        return EastVariant("Null", None)
    elif isinstance(value, bool):
        return EastVariant("Boolean", value)
    elif isinstance(value, int):
        return EastVariant("Integer", value)
    elif isinstance(value, float):
        return EastVariant("Float", value)
    elif isinstance(value, str):
        return EastVariant("String", value)
    elif isinstance(value, bytes):
        return EastVariant("Blob", value)
    elif isinstance(value, datetime):
        return EastVariant("DateTime", value)
    else:
        return EastVariant("Null", None)


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

        # Determine query type
        trimmed_sql = sql.strip().upper()

        async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(sql, native_params)

            if trimmed_sql.startswith("SELECT") or cursor.description:
                # SELECT query - return rows
                rows = await cursor.fetchall()

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
