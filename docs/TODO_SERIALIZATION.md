# Serialization & DateTime Format

**Source Files (TypeScript)**:
- /home/crambelsoupy/src/East/src/serialization/beast.ts (561 lines)
- /home/crambelsoupy/src/East/src/serialization/beast.spec.ts (489 lines)
- /home/crambelsoupy/src/East/src/serialization/east.ts (1,360 lines)
- /home/crambelsoupy/src/East/src/serialization/east.spec.ts (1,633 lines)
- /home/crambelsoupy/src/East/src/serialization/json.ts (833 lines)
- /home/crambelsoupy/src/East/src/serialization/json.spec.ts (1,369 lines)
- /home/crambelsoupy/src/East/src/datetime_format/*.ts (8 files, 3,421 lines)

**Target Files (Python)**:
- east/serialization/beast.py (TO BE CREATED)
- east/serialization/east_printer.py (EXISTS - needs completion/validation)
- east/serialization/east_parser.py (EXISTS - needs completion/validation)
- east/serialization/json.py (EXISTS - 17 tests passing, needs remaining ~53 tests)
- east/serialization/datetime_format/ (TO BE CREATED as package with modules)
  - types.py
  - validate.py
  - tokenize.py
  - parse.py
  - print.py
- tests/serialization/test_beast.py (TO BE CREATED)
- tests/serialization/test_east.py (EXISTS - partially tested)
- tests/serialization/test_json.py (EXISTS - 17 tests, needs ~53 more)
- tests/serialization/datetime_format/ (TO BE CREATED)
  - test_tokenize.py
  - test_parse.py
  - test_print.py

---

## beast.ts → east/serialization/beast.py (TO BE CREATED)

**Binary format with byte-ordering preservation for database indexing**

#### Constants (2)
- [ ] Port BEAST_TYPE_TO_BYTE mapping (type names → byte values 0-13)
- [ ] Port BEAST_BYTE_TO_TYPE array (byte values → type names)
- [ ] Port MAGIC_BYTES signature: `[69, 97, 115, 116, 0, 234, 87, 255]`

#### Core Functions (8 functions)
- [ ] Port encodeTypeToBeastBuffer(type, writer) - Encode type schema to buffer
- [ ] Port decodeTypeBeast(buffer, offset) - Decode type schema from buffer (returns [type, offset])
- [ ] Port encodeBeastValueToBufferFor(type) - Create encoder closure for values
- [ ] Port decodeBeastValueFor(type) - Create decoder closure for values
- [ ] Port encodeBeastValueFor(type) - Header-free encoding (raw value only)
- [ ] Port encodeBeastFor(type) - Full Beast v1 with header (magic + type + value)
- [ ] Port decodeBeast(data) - Decode without type checking
- [ ] Port decodeBeastFor(type) - Decode with type validation

#### Implementation Details
- [ ] Single-byte type tags with nullable flag at bit 7
- [ ] Byte-ordering preservation (critical for indexing)
- [ ] Auto-convert nullable types to Variant on decode
- [ ] "Twiddled" encoding for signed integers/floats
- [ ] BufferWriter utility class

---

### beast.spec.ts → tests/serialization/test_beast.py (489 lines TypeScript)

#### Test Suite: Round-trip and Ordering Tests (12 type suites)
- [ ] Beast v1: Null type - round-trip and maintain ordering
- [ ] Beast v1: Boolean type - round-trip and maintain ordering
- [ ] Beast v1: Integer type - round-trip and maintain ordering
- [ ] Beast v1: Float type - round-trip and maintain ordering
- [ ] Beast v1: String type - round-trip and maintain ordering
- [ ] Beast v1: DateTime type - round-trip and maintain ordering
- [ ] Beast v1: Blob type - round-trip (NO ordering - length-first encoding)
- [ ] Beast v1: Array type - round-trip and maintain ordering
- [ ] Beast v1: Set type - round-trip and maintain ordering
- [ ] Beast v1: Dict type - round-trip and maintain ordering
- [ ] Beast v1: Struct type - round-trip and maintain ordering
- [ ] Beast v1: Variant type - round-trip and maintain ordering

#### Test Suite: Complex Structures (3 tests)
- [ ] Complex struct with nullable fields - production-like test
- [ ] Nested arrays - deep nesting
- [ ] Nested structs - deep struct composition

#### Test Suite: Byte Ordering Properties (4 tests)
- [ ] Integer byte ordering matches numeric ordering
- [ ] Float byte ordering matches numeric ordering (including NaN, Infinity)
- [ ] String byte ordering matches lexicographic ordering
- [ ] Array byte ordering matches lexicographic ordering

#### Test Suite: Error Handling (5 tests)
- [ ] Throw on truncated integer data
- [ ] Throw on truncated string (missing null terminator)
- [ ] Throw on invalid array continuation byte
- [ ] Throw on invalid variant tag
- [ ] Throw on excess data after value

---

### east.ts → east/serialization/east_printer.py & east_parser.py (1,360 lines TypeScript)

**NOTE: Partially implemented - validate and complete**

#### Types (4 types)
- [x] ParseResult<T> discriminated union (already exists in Python)
- [ ] EastPrintTypeContext - stack of printers for recursive types
- [ ] EastParseTypeContext - stack of parsers for recursive types
- [ ] EastPrintValueContext - refs map, currentPath for aliasing
- [ ] EastParseValueContext - refs map, currentPath for aliasing

#### Main API Functions (4 functions)
- [x] encodeEastFor(type) - encode to UTF-8 bytes (validate match)
- [x] decodeEastFor(type, frozen) - decode from UTF-8 bytes (validate match)
- [x] printFor(type, typeCtx) - print to text format (validate match)
- [x] parseFor(type, frozen) - parse from text format (validate match)

#### Aliasing Support (3 helper functions)
- [ ] _commonPrefixLength(a, b) - find common path prefix
- [ ] _encodeRelativeRef(currentPath, targetPath) - create "2#.foo[0]" syntax
- [ ] _decodeRelativeRef(refStr, currentPath) - parse relative references

#### Parser Creators (6 major parsers)
- [ ] createArrayParser - with alias detection
- [ ] createSetParser - with alias detection
- [ ] createDictParser - with alias detection
- [ ] createStructParser - field validation
- [ ] createVariantParser - case matching
- [ ] createParser - main factory dispatch

#### Primitive Parsers (7 parsers)
- [ ] parseNull
- [ ] parseBoolean
- [ ] parseInteger
- [ ] parseFloat
- [ ] parseString
- [ ] parseDateTime
- [ ] parseBlob

#### Parser Utilities (5 utilities)
- [ ] parseReference<T> - handle "#..." aliasing syntax
- [ ] parseIdentifier - unquoted identifiers
- [ ] parseQuotedIdentifier - backtick-quoted identifiers
- [ ] consumeWhitespace - skip whitespace
- [ ] isTokenTerminator - validate token boundaries

---

### east.spec.ts → tests/serialization/test_east.py (1,633 lines TypeScript)

**NOTE: Partially tested - need full coverage validation**

#### Test Suite: parseFor - Primitive Values (10 tests)
- [x] should parse null (validate exists)
- [x] should parse booleans (validate exists)
- [x] should parse integers (validate exists)
- [x] should parse floats (validate exists)
- [x] should parse strings (validate exists)
- [ ] should parse strings with basic content
- [ ] should error on unsupported escape sequence \n
- [ ] should error on unsupported escape sequence \t
- [x] should parse datetime (validate exists)
- [x] should parse blobs (validate exists)

#### Test Suite: parseFor - Collection Values (6 tests)
- [x] should parse empty array
- [x] should parse array with elements
- [ ] should error on array with trailing comma
- [x] should parse empty set
- [x] should parse set with elements
- [ ] should error on set with trailing comma
- [x] should parse empty dict
- [x] should parse dict with entries
- [ ] should error on dict with trailing comma

#### Test Suite: parseFor - Struct Values (5 tests)
- [x] should parse empty struct
- [x] should parse struct with fields
- [ ] should parse struct with quoted field names
- [ ] should error on missing required field
- [ ] should error on unknown field

#### Test Suite: parseFor - Variant Values (7 tests)
- [x] should parse nullary variant without explicit null
- [ ] should parse nullary variant with null provided
- [x] should parse variant with data
- [ ] should error on unknown variant case
- [ ] should error on variant case with incorrect payload
- [ ] should error when data is provided for nullary case
- [ ] should error when no data is provided for data case

#### Test Suite: parseFor - Complex Nested Values (2 tests)
- [ ] should parse complex nested structure
- [ ] should parse deeply nested structure with multiple collection types

#### Test Suite: parseFor - Error Cases (3 tests)
- [ ] should return error for type mismatch
- [ ] should return error for malformed input
- [ ] should return error for extra tokens

#### Test Suite: printFor and Round-trip - Primitives (7 tests)
- [x] null should round-trip
- [x] booleans should round-trip
- [x] integers should round-trip
- [x] floats should round-trip
- [x] strings should round-trip
- [x] datetime should round-trip
- [x] blobs should round-trip

#### Test Suite: printFor and Round-trip - Collections (3 tests)
- [x] arrays should round-trip
- [x] sets should round-trip
- [x] dicts should round-trip

#### Test Suite: printFor and Round-trip - Struct/Variant (5 tests)
- [x] empty struct should round-trip
- [x] struct with fields should round-trip
- [ ] struct with quoted field names should round-trip
- [x] variant nullary case should round-trip
- [x] variant with data should round-trip

#### Test Suite: printFor and Round-trip - Complex Structures (3 tests)
- [ ] nested arrays should round-trip
- [ ] struct with array field should round-trip
- [ ] deeply nested structure should round-trip

#### Test Suite: printFor Edge Cases (4 suites)
- [ ] float formatting edge cases
- [ ] string escaping
- [ ] identifier quoting in structs
- [ ] empty collections formatting
- [ ] variant case formatting

#### Test Suite: Uint8Array Encoding (3 tests)
- [ ] should encode/decode integers with Uint8Array
- [ ] should encode/decode structs with Uint8Array
- [ ] should throw error when decoding invalid East format

#### Test Suite: Never/Function Type Handling (4 tests)
- [ ] should throw when printing Never type
- [ ] should throw when parsing Never type
- [ ] should print Function type as λ
- [ ] should throw when creating parser for Function type

#### Test Suite: Frozen Parameter (8 tests)
- [ ] should freeze DateTime when frozen=true
- [ ] should not freeze DateTime when frozen=false
- [ ] should attempt to freeze Blob when frozen=true
- [ ] should freeze Array when frozen=true
- [ ] should freeze Set when frozen=true
- [ ] should freeze Dict when frozen=true
- [ ] should freeze Struct when frozen=true
- [ ] should freeze Variant when frozen=true

#### Test Suite: Additional Error Cases (25+ error tests)
- [ ] All struct parsing errors (missing fields, wrong fields, etc.)
- [ ] All variant parsing errors (unknown case, wrong payload, etc.)
- [ ] All number parsing errors (missing exponent, etc.)
- [ ] All datetime parsing errors (invalid format, invalid value)
- [ ] All blob parsing errors (odd hex digits, missing 0x)
- [ ] All collection parsing errors (missing delimiters, etc.)

#### Test Suite: Fuzz Tests (2 tests)
- [ ] should round-trip random types and values (100 types × 10 samples)
- [ ] should round-trip with Uint8Array encoding for random types

#### Test Suite: Aliasing (4 tests)
- [ ] should detect array aliases in struct
- [ ] should detect set aliases in struct
- [ ] should detect dict aliases in struct
- [ ] should detect nested array aliases

#### Test Suite: Recursive Types (3 tests)
- [ ] should print tree without cycles
- [ ] should print larger tree without cycles
- [ ] should print linked list without cycles

---

### json.ts → east/serialization/json.py (833 lines TypeScript)

**NOTE: Core functions implemented - need completion and validation**

#### Types (4 types)
- [ ] JSONEncodeTypeContext - stack of encoders for recursive types
- [ ] JSONEncodeValueContext - refs map, currentPath for aliasing
- [ ] JSONDecodeTypeContext - stack of decoders for recursive types
- [ ] JSONDecodeValueContext - refs map, currentPath for aliasing
- [ ] JSONDecodeError - error class with path information

#### Main API Functions (4 functions)
- [x] encodeJSONFor(type) - encode to JSON UTF-8 bytes (validate complete)
- [x] decodeJSONFor(type, frozen) - decode from JSON UTF-8 bytes (validate complete)
- [x] toJSONFor(type, typeCtx) - convert to JSON-serializable (validate complete)
- [x] fromJSONFor(type, frozen) - convert from JSON value (validate complete)

#### JSON Pointer Support (3 helper functions)
- [ ] _encodeJSONPointerComponent(component) - RFC 6901 encoding (~ → ~0, / → ~1)
- [ ] _decodeJSONPointerComponent(component) - RFC 6901 decoding
- [ ] encodeRelativeRef(currentPath, targetPath) - create "2#foo/bar" syntax
- [ ] _decodeRelativeRef(refStr, currentPath) - parse relative JSON pointer
- [ ] _commonPrefixLength(a, b) - find common prefix

#### Decoder Factory (1 major function)
- [ ] createJSONDecoder(type, frozen, typeCtx) - internal decoder factory

#### JSON Format Validation
- [x] Integers: string representation "42"
- [x] Floats: number or string for NaN/Infinity/-0.0
- [x] DateTime: RFC 3339 with timezone
- [x] Blob: hex string "0x..."
- [x] Set: JSON array
- [x] Dict: array of {key, value} objects
- [x] Struct: JSON object
- [x] Variant: {type, value} object
- [ ] References: {"$ref": "2#path/to/target"} for shared containers

---

### json.spec.ts → tests/serialization/test_json.py (1,369 lines TypeScript)

**NOTE: 17 tests passing - need to complete remaining ~60 tests**

#### Test Suite: Basic Type Encoding/Decoding (11 tests)
- [x] should encode/decode null
- [x] should encode/decode boolean
- [x] should encode/decode integer
- [x] should encode/decode float
- [x] should encode/decode string
- [x] should encode/decode date
- [x] should encode/decode array
- [x] should encode/decode set
- [x] should encode/decode dict
- [x] should encode/decode struct
- [x] should encode/decode variant
- [x] should encode/decode blob

#### Test Suite: Recursive Type Encoding (5 tests)
- [x] should encode/decode simple linked list (RecursiveType)
- [x] should encode/decode binary tree
- [x] should encode/decode tree with array children
- [x] should encode/decode graph with string labels
- [x] should encode/decode nested variant structures

#### Test Suite: Error Message Formatting (15 specific error tests)
- [ ] Error at root level
- [ ] Error in array element
- [ ] Error in nested array
- [ ] Error in set element
- [ ] Error in dict key
- [ ] Error in dict value
- [ ] Error in struct field
- [ ] Error in nested struct field
- [ ] Error in variant value
- [ ] Error in variant nested field
- [ ] Error with numeric field names
- [ ] Error with special characters in field names
- [ ] Error with escaped characters in field names
- [ ] Error in deeply nested structure
- [ ] Error with mixed collection types

#### Test Suite: Uint8Array Encoding (2 tests)
- [ ] should encode/decode using Uint8Array with encodeJSONFor/decodeJSONFor
- [ ] should encode/decode complex types with encodeJSONFor/decodeJSONFor

#### Test Suite: Never Type Handling (3 tests)
- [ ] should throw when encoding Never type
- [ ] should throw when decoding Never type with fromJSONFor
- [ ] should throw when decoding Never type with decodeJSONFor

#### Test Suite: Function Type Handling (3 tests)
- [ ] should throw when encoding Function type
- [ ] should throw when decoding Function type with fromJSONFor
- [ ] should throw when creating decoder for Function type with decodeJSONFor

#### Test Suite: Frozen Parameter (16 tests - 2 per container type)
- [ ] should freeze decoded Date when frozen=true
- [ ] should not freeze decoded Date when frozen=false
- [ ] should attempt to freeze decoded Blob when frozen=true
- [ ] should not freeze decoded Blob when frozen=false
- [ ] should freeze decoded Array when frozen=true
- [ ] should not freeze decoded Array when frozen=false
- [ ] should freeze decoded Set when frozen=true
- [ ] should not freeze decoded Set when frozen=false
- [ ] should freeze decoded Dict when frozen=true
- [ ] should not freeze decoded Dict when frozen=false
- [ ] should freeze decoded Struct when frozen=true
- [ ] should not freeze decoded Struct when frozen=false
- [ ] should freeze decoded Variant when frozen=true
- [ ] should not freeze decoded Variant when frozen=false

#### Test Suite: JSON Parse Errors (4 tests)
- [ ] should handle JSON.parse syntax errors in decodeJSONFor
- [ ] should track line and column numbers in JSON parse errors
- [ ] should handle JSON parse errors without position info
- [ ] should handle non-SyntaxError exceptions in decodeJSONFor

#### Test Suite: Frozen Parameter Edge Cases (1 test)
- [ ] should use decodeJSONFor with frozen parameter

#### Test Suite: Fuzz Tests (2 tests)
- [x] should round-trip random types and values (100 types × 10 samples)
- [ ] should round-trip with Uint8Array encoding for random types

#### Test Suite: Error Propagation (6 tests)
- [ ] should re-throw non-JSONDecodeError exceptions in Array decoding
- [ ] should re-throw non-JSONDecodeError exceptions in Set decoding
- [ ] should re-throw non-JSONDecodeError exceptions in Dict key decoding
- [ ] should re-throw non-JSONDecodeError exceptions in Dict value decoding
- [ ] should re-throw non-JSONDecodeError exceptions in Struct decoding
- [ ] should re-throw non-JSONDecodeError exceptions in Variant decoding

#### Test Suite: Shared References (3 tests)
- [ ] should encode shared array references within RecursiveType
- [ ] should encode shared set references within RecursiveType
- [ ] should encode shared dict references within RecursiveType

#### Test Suite: JSON Pointer Escaping (1 test)
- [ ] should handle JSON Pointer escaping in field names

---

### datetime_format/types.ts → east/datetime_format/types.py (145 lines TypeScript)

#### Types (2 types)
- [ ] Port DateTimeFormatTokenType - variant with 22 cases
- [ ] Port DateTimeFormatToken - Python representation

#### Token Cases (22 cases)
- [ ] literal: String
- [ ] Year tokens: year4, year2
- [ ] Month tokens: month1, month2, monthNameShort, monthNameFull
- [ ] Day tokens: day1, day2
- [ ] Weekday tokens: weekdayNameMin, weekdayNameShort, weekdayNameFull
- [ ] Hour 24h tokens: hour24_1, hour24_2
- [ ] Hour 12h tokens: hour12_1, hour12_2
- [ ] Minute tokens: minute1, minute2
- [ ] Second tokens: second1, second2
- [ ] Millisecond token: millisecond3
- [ ] AM/PM tokens: ampmUpper, ampmLower

---

### datetime_format/validate.ts → east/datetime_format/validate.py (239 lines TypeScript)

#### Types (1 type)
- [ ] Port DateTimeFormatValidationResult - discriminated union

#### Functions (1 function)
- [ ] Port validateDateTimeFormatTokens(tokens) - validates contiguous prefix

#### Validation Rules
- [ ] Component hierarchy: Year → Month → Day → Hour → Minute → Second → Millisecond
- [ ] Cannot skip levels (e.g., can't have Year + Hour without Month + Day)
- [ ] Weekday, AM/PM, and literals don't affect hierarchy
- [ ] Time-only formats (no date components) are allowed

---

### datetime_format/tokenize.ts → east/datetime_format/tokenize.py (305 lines TypeScript)

#### Functions (2 functions)
- [ ] Port tokenizeDateTimeFormat(format) - parse format string to tokens
- [ ] Port formatTokensToString(tokens, colorize) - tokens to canonical format string

#### Format Patterns (Longest to Shortest Matching)
- [ ] Year: YYYY (4-digit), YY (2-digit)
- [ ] Month: MMMM (full name), MMM (short), MM (2-digit), M (1-2 digit)
- [ ] Day: DD (2-digit), D (1-2 digit)
- [ ] Weekday: dddd (full), ddd (short), dd (minimal)
- [ ] Hour 24h: HH, H
- [ ] Hour 12h: hh, h
- [ ] Minute: mm, m
- [ ] Second: ss, s
- [ ] Millisecond: SSS
- [ ] AM/PM: A (upper), a (lower)

#### Escaping Support
- [ ] \x produces literal x for any character
- [ ] Terminating backslash treated as literal backslash
- [ ] Unicode support (correctly handles surrogate pairs)

---

### datetime_format/tokenize.spec.ts → tests/datetime_format/test_tokenize.py (573 lines TypeScript)

#### Test Suite: parseDateTimeFormat (50+ tests)
- [ ] empty string
- [ ] year tokens (YYYY, YY)
- [ ] month tokens (M, MM, MMM, MMMM)
- [ ] day tokens (D, DD)
- [ ] weekday tokens (dd, ddd, dddd)
- [ ] hour tokens - 24h (H, HH)
- [ ] hour tokens - 12h (h, hh)
- [ ] minute tokens (m, mm)
- [ ] second tokens (s, ss)
- [ ] millisecond token (SSS)
- [ ] AM/PM tokens (A, a)
- [ ] ISO 8601 date format (YYYY-MM-DD)
- [ ] ISO 8601 datetime format (YYYY-MM-DD HH:mm:ss)
- [ ] datetime with milliseconds
- [ ] 12-hour format with AM/PM
- [ ] long date format with month name
- [ ] weekday format
- [ ] pure literal string - must escape
- [ ] literal grouping - escape letters that are format codes
- [ ] escape single character
- [ ] escape within tokens
- [ ] escape backslash
- [ ] multiple escaped backslashes
- [ ] escape non-token characters
- [ ] terminating backslash
- [ ] unicode - CJK characters
- [ ] unicode - emoji
- [ ] unicode - emoji in escape
- [ ] mixed padding - unpadded time
- [ ] ambiguous MM vs M - longer pattern wins
- [ ] ambiguous MMMM vs MMM - longer pattern wins
- [ ] adjacent tokens without separators
- [ ] adjacent different token types
- [ ] partial token followed by literal
- [ ] token at end
- [ ] complex real-world format 1
- [ ] complex real-world format 2
- [ ] format with newlines
- [ ] format with tabs
- [ ] escape n as literal n (not newline)
- [ ] escape t as literal t (not tab)

#### Test Suite: formatTokensToString (20+ tests)
- [ ] empty token array
- [ ] single format token
- [ ] ISO 8601 format
- [ ] minimal escaping - only escapes when necessary
- [ ] minimal escaping - format tokens separate literals
- [ ] escapes backslashes
- [ ] escapes multiple backslashes
- [ ] unicode - CJK characters need no escaping
- [ ] unicode - emoji need no escaping
- [ ] round-trip property - ISO format
- [ ] round-trip property - escaped format
- [ ] round-trip property - mixed
- [ ] round-trip property - complex format
- [ ] YYYY as literal requires multiple escapes
- [ ] pattern at end of literal with format codes in middle
- [ ] adjacent literals and tokens
- [ ] all token types round-trip with separators
- [ ] debugging use case - show interpretation

---

### datetime_format/parse.ts → east/datetime_format/parse.py (958 lines TypeScript)

#### Types (1 type)
- [ ] Port DateTimeParseResult - discriminated union

#### Constants (5 constants)
- [ ] Port MONTH_NAMES_FULL - 12 full month names (English)
- [ ] Port MONTH_NAMES_SHORT - 12 short month names (English)
- [ ] Port WEEKDAY_NAMES_FULL - 7 full weekday names (English)
- [ ] Port WEEKDAY_NAMES_SHORT - 7 short weekday names (English)
- [ ] Port WEEKDAY_NAMES_MIN - 7 minimal weekday names (English)

#### Functions (1 function)
- [ ] Port parseDateTimeFormatted(input, tokens) - parse datetime string

#### Features
- [ ] All dates treated as UTC
- [ ] Weekday tokens consumed but validated against actual date
- [ ] Handles redundant specifications (detects conflicts)
- [ ] 2-digit year: 00-99 → 2000-2099
- [ ] 12-hour format requires AM/PM
- [ ] Validates date is actually valid (e.g., rejects Feb 31)
- [ ] Fills in defaults while checking for hierarchy gaps
- [ ] Supports unpadded formats (M, D, H, m, s)
- [ ] Case-insensitive month/weekday name matching

---

### datetime_format/parse.spec.ts → tests/datetime_format/test_parse.py (457 lines TypeScript)

#### Test Suite: parse (45 tests)
- [ ] basic ISO 8601 date
- [ ] ISO 8601 datetime
- [ ] with milliseconds
- [ ] 12-hour format with PM
- [ ] 12-hour format with AM
- [ ] 12 AM (midnight)
- [ ] 12 PM (noon)
- [ ] month names (full)
- [ ] month names (short)
- [ ] month names case insensitive
- [ ] weekday names (ignored but consumed)
- [ ] unpadded single-digit month
- [ ] unpadded double-digit month
- [ ] 2-digit year
- [ ] error: missing year
- [ ] error: missing month
- [ ] missing day defaults to 1st
- [ ] error: month out of range
- [ ] error: day out of range
- [ ] error: invalid date (Feb 31)
- [ ] valid leap year date
- [ ] error: invalid leap year date
- [ ] error: literal mismatch
- [ ] error: trailing characters
- [ ] error: unexpected end of input
- [ ] error: expected 4-digit year
- [ ] error: expected 2-digit month
- [ ] complex format with weekday
- [ ] round-trip: format then parse
- [ ] all months
- [ ] all weekdays
- [ ] lowercase am/pm
- [ ] defaults for missing time components
- [ ] unpadded time components
- [ ] error: hour 24-hour out of range
- [ ] error: hour 12-hour out of range
- [ ] error: minute out of range
- [ ] error: second out of range

---

### datetime_format/print.ts → east/datetime_format/print.py (199 lines TypeScript)

#### Constants (5 constants)
- [ ] Port MONTH_NAMES_FULL - 12 full month names (English)
- [ ] Port MONTH_NAMES_SHORT - 12 short month names (English)
- [ ] Port WEEKDAY_NAMES_FULL - 7 full weekday names (English)
- [ ] Port WEEKDAY_NAMES_SHORT - 7 short weekday names (English)
- [ ] Port WEEKDAY_NAMES_MIN - 7 minimal weekday names (English)

#### Functions (1 function)
- [ ] Port formatDateTime(date, tokens) - format Date according to tokens

#### Features
- [ ] All dates treated as UTC (uses getUTC* methods)
- [ ] 12-hour conversion: hour % 12 || 12
- [ ] Zero-padding with toString().padStart()
- [ ] Straightforward token-to-string conversion

---

### datetime_format/print.spec.ts → tests/datetime_format/test_print.py (545 lines TypeScript)

#### Test Suite: formatDateTime (45 tests)
- [ ] year tokens
- [ ] year2 padding
- [ ] month tokens
- [ ] month names for all months
- [ ] day tokens
- [ ] weekday tokens
- [ ] weekday names for all days
- [ ] 24-hour format
- [ ] 12-hour format
- [ ] 12-hour format edge cases (midnight, noon, 1 AM, 1 PM)
- [ ] minute tokens
- [ ] second tokens
- [ ] millisecond token
- [ ] AM/PM tokens
- [ ] literal token
- [ ] ISO 8601 date format
- [ ] ISO 8601 datetime format
- [ ] ISO 8601 with milliseconds
- [ ] 12-hour format with AM/PM
- [ ] long date format
- [ ] weekday with date
- [ ] compact format
- [ ] with escaped literals
- [ ] complex real-world format
- [ ] edge case: year 2000
- [ ] edge case: leap year
- [ ] edge case: end of year
- [ ] empty token array
- [ ] only literals
- [ ] all 24-hour hours
- [ ] all 12-hour hours with AM/PM
- [ ] unicode in literals
- [ ] newlines in format

---

## Phase 2.5 UPDATED Summary Statistics

**Original Scope** (already in TODO.md):
- types.ts: 1,575 lines → 18 functions, 30+ types, 95+ tests
- comparison.ts: 943 lines → 8 functions, 50+ tests
- default.ts: 97 lines → 2 functions, 31 tests
- analyze.ts: 1,542 lines → 1 main function + infrastructure, 23 tests

**Additional Scope** (this document):

**Serialization Files** (6 files, ~6,245 lines):
- beast.ts: 561 lines → 8 functions, 2 constants
- beast.spec.ts: 489 lines → 15 test suites, 24+ tests
- east.ts: 1,360 lines → 4 main API, 15+ helpers (partially done)
- east.spec.ts: 1,633 lines → 11 test suites, 147+ tests (partially done)
- json.ts: 833 lines → 4 main API, 5+ helpers (partially done)
- json.spec.ts: 1,369 lines → 1 test suite, 70+ tests (17 passing, ~53 remaining)

**DateTime Format Files** (8 files, ~3,421 lines):
- types.ts: 145 lines → 1 variant type with 22 cases
- validate.ts: 239 lines → 1 validation function
- tokenize.ts: 305 lines → 2 functions
- tokenize.spec.ts: 573 lines → 2 test suites, 70+ tests
- parse.ts: 958 lines → 1 parse function, 5 constants
- parse.spec.ts: 457 lines → 1 test suite, 45 tests
- print.ts: 199 lines → 1 format function, 5 constants
- print.spec.ts: 545 lines → 1 test suite, 45 tests

**GRAND TOTAL FOR PHASE 2.5:**
- **22 TypeScript files** (11 implementation + 11 test files)
- **~13,823 lines** of TypeScript to port
- **~400+ test cases** to validate
- **~50+ functions** to implement
