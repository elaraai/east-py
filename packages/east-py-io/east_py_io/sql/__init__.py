"""SQL module - SQLite, PostgreSQL, and MySQL databases."""

from east_py_io.sql.mysql import (
    mysql_close_all_impl,
    mysql_close_impl,
    mysql_connect_impl,
    mysql_impl,
    mysql_query_impl,
)
from east_py_io.sql.postgres import (
    postgres_close_all_impl,
    postgres_close_impl,
    postgres_connect_impl,
    postgres_impl,
    postgres_query_impl,
)
from east_py_io.sql.sqlite import (
    sqlite_close_all_impl,
    sqlite_close_impl,
    sqlite_connect_impl,
    sqlite_impl,
    sqlite_query_impl,
)
from east_py_io.sql.types import (
    ConnectionHandleType,
    MySqlConfigType,
    PostgresConfigType,
    SqliteConfigType,
    SqlParametersType,
    SqlParameterType,
    SqlResultType,
    SqlRowType,
)

__all__ = [
    # Types
    "SqliteConfigType",
    "PostgresConfigType",
    "MySqlConfigType",
    "ConnectionHandleType",
    "SqlParametersType",
    "SqlParameterType",
    "SqlRowType",
    "SqlResultType",
    # SQLite
    "sqlite_impl",
    "sqlite_connect_impl",
    "sqlite_query_impl",
    "sqlite_close_impl",
    "sqlite_close_all_impl",
    # PostgreSQL
    "postgres_impl",
    "postgres_connect_impl",
    "postgres_query_impl",
    "postgres_close_impl",
    "postgres_close_all_impl",
    # MySQL
    "mysql_impl",
    "mysql_connect_impl",
    "mysql_query_impl",
    "mysql_close_impl",
    "mysql_close_all_impl",
]
