#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Microsoft Access platform functions for East.

Provides read-only Access database operations for East programs, including
opening databases, listing tables, and querying table data with user-defined
return types.

Supported formats:
- .mdb - Access 97, 2000, 2002/2003
- .accdb - Access 2007, 2010, 2013, 2016, 2019

Note: Requires mdb-parser package. Install with: pip install mdb-parser
or pip install east-py-io[access]
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from east.runtime.platform import GenericPlatformFunction, PlatformFunction
from east.types.types import NullType, StringType
from east.types.values import EastArray, EastStruct, EastVariant

from .types import (
    AccessBlobConfigType,
    AccessConfigType,
    AccessTablesResultType,
    ConnectionHandleType,
)

# Try to import mdb-parser
try:
    from mdb_parser import MDBParser, MDBTable  # type: ignore

    _HAS_MDB_SUPPORT = True
except ImportError:
    _HAS_MDB_SUPPORT = False
    MDBParser = None  # type: ignore
    MDBTable = None  # type: ignore


# Connection storage (maps handle -> MDBParser instance)
_access_connections: dict[str, Any] = {}


def _check_mdb_support() -> None:
    """Check if MDB support is available."""
    if not _HAS_MDB_SUPPORT:
        raise NotImplementedError(
            "Microsoft Access support requires mdb-parser. "
            "Install with: pip install mdb-parser or pip install east-py-io[access]"
        )


async def access_open_impl(config: EastStruct) -> str:
    """Open a Microsoft Access database file.

    Args:
        config: Access connection configuration with path and optional password

    Returns:
        Connection handle (opaque string)

    Raises:
        NotImplementedError: If mdb-parser is not installed
        Exception: If database open fails
    """
    _check_mdb_support()

    try:
        path = config["path"]
        password_opt = config["password"]
        password = password_opt.value if password_opt.type == "some" else None

        # Open the database
        reader = MDBParser(path, password=password) if password else MDBParser(path)

        handle = str(uuid.uuid4())
        _access_connections[handle] = reader

        return handle
    except Exception as e:
        raise Exception(f"Access database open failed: {e}") from e


async def access_open_blob_impl(config: EastStruct) -> str:
    """Open a Microsoft Access database from binary data.

    Args:
        config: Access blob configuration with data and optional password

    Returns:
        Connection handle (opaque string)

    Raises:
        NotImplementedError: If mdb-parser is not installed
        Exception: If database open fails
    """
    _check_mdb_support()

    try:
        data = config["data"]
        password_opt = config["password"]
        password = password_opt.value if password_opt.type == "some" else None

        # Convert to bytes if needed
        if hasattr(data, "data"):
            # EastBlob has a data property
            buffer = bytes(data.data)
        elif isinstance(data, (bytes, bytearray)):
            buffer = bytes(data)
        else:
            buffer = bytes(data)

        # Open from bytes using from_blob method
        if password:
            reader = MDBParser.from_blob(buffer, password=password)
        else:
            reader = MDBParser.from_blob(buffer)

        handle = str(uuid.uuid4())
        _access_connections[handle] = reader

        return handle
    except Exception as e:
        raise Exception(f"Access database open from blob failed: {e}") from e


async def access_tables_impl(handle: str) -> EastStruct:
    """List all table names in the database.

    Args:
        handle: Connection handle

    Returns:
        Struct with tables array containing table names

    Raises:
        NotImplementedError: If mdb-parser is not installed
        Exception: If operation fails or handle is invalid
    """
    _check_mdb_support()

    try:
        if handle not in _access_connections:
            raise Exception(f"Invalid connection handle: {handle}")

        reader = _access_connections[handle]
        tables = list(reader.tables)  # mdb-parser returns table names via .tables

        return EastStruct({"tables": EastArray(StringType, tables)})
    except Exception as e:
        raise Exception(f"Access tables list failed: {e}") from e


# Access type name to East type mapping
_ACCESS_TYPE_MAP = {
    "boolean": "Boolean",
    "byte": "Integer",
    "integer": "Integer",
    "int": "Integer",
    "long": "Integer",
    "longint": "Integer",
    "autoincrement": "Integer",
    "bigint": "Integer",
    "float": "Float",
    "single": "Float",
    "double": "Float",
    "text": "String",
    "memo": "String",
    "currency": "String",
    "numeric": "String",
    "repid": "String",
    "guid": "String",
    "datetime": "DateTime",
    "datetimeextended": "DateTime",
    "binary": "Blob",
    "ole": "Blob",
    "ole object": "Blob",
}


def _get_east_type_for_access(access_type: str) -> str:
    """Get East type name for Access column type."""
    return _ACCESS_TYPE_MAP.get(access_type.lower(), "String")


def _convert_access_value(value: Any, access_type: str) -> Any:
    """Convert an Access value to the appropriate East value."""
    if value is None:
        return None

    type_lower = access_type.lower()

    if type_lower in ("byte", "integer", "int", "long", "longint", "autoincrement", "bigint"):
        return int(value)
    elif type_lower in ("float", "single", "double"):
        return float(value)
    elif type_lower == "boolean":
        return bool(value)
    elif type_lower in ("binary", "ole", "ole object"):
        if isinstance(value, (bytes, bytearray)):
            from east.types.values import EastBlob

            return EastBlob(bytes(value))
        return value
    elif type_lower in ("datetime", "datetimeextended"):
        if isinstance(value, datetime):
            # Ensure UTC timezone and truncate to milliseconds
            value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
            ms = (value.microsecond // 1000) * 1000
            return value.replace(microsecond=ms)
        return value
    else:
        # String types - ensure string conversion
        return str(value) if value is not None else value


def _is_option_type(field_type: dict, base_type: str) -> bool:
    """Check if field_type is OptionType(base_type)."""
    if field_type.get("type") != "Option":
        return False
    inner = field_type.get("value")
    return inner is not None and inner.get("type") == base_type


def _is_matching_type(field_type: dict, base_type: str) -> bool:
    """Check if field_type matches base_type."""
    return field_type.get("type") == base_type


def access_query_factory(row_type: Any) -> Any:
    """Factory for access_query that captures the type parameter.

    Args:
        row_type: Row type parameter (East IR type format)

    Returns:
        Async implementation function for access_query
    """

    async def access_query_impl(handle: str, options: EastStruct) -> EastArray:
        """Query data from an Access table with typed results.

        Args:
            handle: Connection handle
            options: Query options with table name and optional columns/offset/limit

        Returns:
            Array of rows matching the type parameter T

        Raises:
            Exception: If query fails or types don't match
        """
        _check_mdb_support()

        try:
            if handle not in _access_connections:
                raise Exception(f"Invalid connection handle: {handle}")

            reader = _access_connections[handle]

            table_name = options["table"]
            row_offset_opt = options["rowOffset"]
            row_limit_opt = options["rowLimit"]

            row_offset = int(row_offset_opt.value) if row_offset_opt.type == "some" else None
            row_limit = int(row_limit_opt.value) if row_limit_opt.type == "some" else None

            # Get table
            table: MDBTable = reader.get_table(table_name)

            # Validate row type is a Struct
            if row_type.get("type") != "Struct":
                raise Exception(f"Expected row type must be a Struct, got {row_type.get('type')}")

            fields = row_type.get("value", [])

            # Get column metadata
            col_meta = {col.name: col for col in table.columns}

            # Validate field types match columns
            field_info: dict[str, dict[str, Any]] = {}
            for field in fields:
                field_name = field["name"]
                field_type = field["type"]

                if field_name not in col_meta:
                    raise Exception(f"Column '{field_name}' not found in table '{table_name}'")

                col = col_meta[field_name]
                access_type = col.type if hasattr(col, "type") else "text"
                expected_east = _get_east_type_for_access(access_type)

                # Check if field type matches expected type or OptionType(expected)
                is_option = _is_option_type(field_type, expected_east)
                is_base = _is_matching_type(field_type, expected_east)

                if not is_base and not is_option:
                    raise Exception(
                        f"Type mismatch for column '{field_name}': Access column is {access_type}, "
                        f"expected {expected_east} or OptionType({expected_east})"
                    )

                field_info[field_name] = {
                    "is_option": is_option,
                    "access_type": access_type,
                }

            # Read table data
            raw_data = list(table.records)

            # Apply offset
            if row_offset is not None and row_offset > 0:
                raw_data = raw_data[row_offset:]

            # Apply limit
            if row_limit is not None and row_limit >= 0:
                raw_data = raw_data[:row_limit]

            # Convert rows
            rows: list[EastStruct] = []
            for row_idx, raw_row in enumerate(raw_data):
                converted: dict[str, Any] = {}

                for field in fields:
                    field_name = field["name"]
                    info = field_info[field_name]

                    # Get value from row (raw_row is dict-like)
                    value = raw_row.get(field_name) if hasattr(raw_row, "get") else getattr(
                        raw_row, field_name, None
                    )

                    if value is None:
                        if info["is_option"]:
                            converted[field_name] = EastVariant("none", None)
                        else:
                            raise Exception(
                                f"null value at row[{row_idx}] for required field '{field_name}' - "
                                f"use OptionType for nullable columns"
                            )
                    else:
                        # Convert value based on Access type
                        converted_value = _convert_access_value(value, info["access_type"])

                        if info["is_option"]:
                            converted[field_name] = EastVariant("some", converted_value)
                        else:
                            converted[field_name] = converted_value

                rows.append(EastStruct(converted))

            return EastArray(row_type, rows)
        except Exception as e:
            raise Exception(f"Access query failed: {e}") from e

    return access_query_impl


async def access_close_impl(handle: str) -> None:
    """Close an Access database connection.

    Args:
        handle: Connection handle

    Raises:
        Exception: If handle is invalid
    """
    try:
        if handle not in _access_connections:
            raise Exception(f"Invalid connection handle: {handle}")

        # mdb-parser doesn't need explicit close, just remove reference
        del _access_connections[handle]
    except Exception as e:
        raise Exception(f"Access close failed: {e}") from e


async def access_close_all_impl() -> None:
    """Close all Access connections.

    Useful for test cleanup.
    """
    _access_connections.clear()


# Platform function implementations
access_impl = [
    PlatformFunction(
        name="access_open",
        inputs=[AccessConfigType],
        output=ConnectionHandleType,
        type="async",
        fn=access_open_impl,
    ),
    PlatformFunction(
        name="access_open_blob",
        inputs=[AccessBlobConfigType],
        output=ConnectionHandleType,
        type="async",
        fn=access_open_blob_impl,
    ),
    PlatformFunction(
        name="access_tables",
        inputs=[ConnectionHandleType],
        output=AccessTablesResultType,
        type="async",
        fn=access_tables_impl,
    ),
    GenericPlatformFunction(
        name="access_query",
        type_parameters=["T"],
        type="async",
        fn=access_query_factory,
    ),
    PlatformFunction(
        name="access_close",
        inputs=[ConnectionHandleType],
        output=NullType,
        type="async",
        fn=access_close_impl,
    ),
    PlatformFunction(
        name="access_close_all",
        inputs=[],
        output=NullType,
        type="async",
        fn=access_close_all_impl,
    ),
]

__all__ = [
    "access_impl",
    "access_open_impl",
    "access_open_blob_impl",
    "access_tables_impl",
    "access_query_factory",
    "access_close_impl",
    "access_close_all_impl",
]
