"""East Python I/O Platform Functions.

Python implementation of I/O platform functions for the East programming language.
Provides database, storage, file transfer, format, and compression operations.
"""

# Compression module
from east_py_io.compression import (
    GzipLevelType,
    GzipOptionsType,
    TarEntriesType,
    TarEntryType,
    TarExtractedType,
    ZipEntriesType,
    ZipEntryType,
    ZipExtractedType,
    ZipLevelType,
    ZipOptionsType,
    gzip_impl,
    tar_impl,
    zip_impl,
)

# Format module
from east_py_io.format import (
    LiteralValueType,
    XlsxCellType,
    XlsxInfoType,
    XlsxReadOptionsType,
    XlsxRowType,
    XlsxSheetInfoType,
    XlsxSheetType,
    XlsxWriteOptionsType,
    XmlNodeType,
    XmlParseConfigType,
    XmlSerializeConfigType,
    xlsx_impl,
    xml_impl,
)

# NoSQL module
from east_py_io.nosql import (
    BsonValueType,
    MongoConfigType,
    MongoDocumentType,
    MongoFindOptionsType,
    RedisConfigType,
    mongodb_impl,
    redis_impl,
)

# SQL module
from east_py_io.sql import (
    ConnectionHandleType,
    MySqlConfigType,
    PostgresConfigType,
    SqliteConfigType,
    SqlParametersType,
    SqlParameterType,
    SqlResultType,
    SqlRowType,
    mysql_impl,
    postgres_impl,
    sqlite_impl,
)
from east_py_io.storage import (
    S3ConfigType,
    S3ListResultType,
    S3ObjectMetadataType,
    s3_impl,
)

# Transfer module
from east_py_io.transfer import (
    FileEntryType,
    FileListType,
    FtpConfigType,
    SftpConfigType,
    ftp_impl,
    sftp_impl,
)

__version__ = "0.1.0"

# Complete Python I/O platform implementation
# Pass this list to compile_async() to enable all platform functions
python_io_platform = [
    # Storage
    *s3_impl,
    # SQL
    *sqlite_impl,
    *postgres_impl,
    *mysql_impl,
    # NoSQL
    *redis_impl,
    *mongodb_impl,
    # Format
    *xlsx_impl,
    *xml_impl,
    # Compression
    *gzip_impl,
    *tar_impl,
    *zip_impl,
    # Transfer
    *ftp_impl,
    *sftp_impl,
]

__all__ = [
    "__version__",
    # Main platform export
    "python_io_platform",
    # Storage types
    "S3ConfigType",
    "S3ObjectMetadataType",
    "S3ListResultType",
    "s3_impl",
    # SQL types
    "SqliteConfigType",
    "PostgresConfigType",
    "MySqlConfigType",
    "ConnectionHandleType",
    "SqlParametersType",
    "SqlParameterType",
    "SqlRowType",
    "SqlResultType",
    "sqlite_impl",
    "postgres_impl",
    "mysql_impl",
    # NoSQL types
    "RedisConfigType",
    "MongoConfigType",
    "MongoFindOptionsType",
    "BsonValueType",
    "MongoDocumentType",
    "redis_impl",
    "mongodb_impl",
    # Format types
    "LiteralValueType",
    "XlsxCellType",
    "XlsxRowType",
    "XlsxSheetType",
    "XlsxReadOptionsType",
    "XlsxWriteOptionsType",
    "XlsxSheetInfoType",
    "XlsxInfoType",
    "XmlNodeType",
    "XmlParseConfigType",
    "XmlSerializeConfigType",
    "xlsx_impl",
    "xml_impl",
    # Compression types
    "GzipLevelType",
    "GzipOptionsType",
    "ZipLevelType",
    "ZipOptionsType",
    "ZipEntryType",
    "ZipEntriesType",
    "ZipExtractedType",
    "TarEntryType",
    "TarEntriesType",
    "TarExtractedType",
    "gzip_impl",
    "tar_impl",
    "zip_impl",
    # Transfer types
    "FtpConfigType",
    "SftpConfigType",
    "FileEntryType",
    "FileListType",
    "ftp_impl",
    "sftp_impl",
]
