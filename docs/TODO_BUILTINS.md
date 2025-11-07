# Builtins Implementation

**Source Files (TypeScript)**:
- /home/crambelsoupy/src/East/src/builtins.ts (1,102 lines)
- /home/crambelsoupy/src/East/test/*.spec.ts (East language tests - executed as compiled IR)

**Target Files (Python)**:
- east/builtins/comparison.py (EXISTS - Is, Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual)
- east/builtins/boolean.py (EXISTS - BooleanNot, BooleanOr, BooleanAnd, BooleanXor)
- east/builtins/integer.py (EXISTS - 13/14 ops, missing IntegerToString)
- east/builtins/float_ops.py (EXISTS - 30 ops complete)
- east/builtins/string.py (EXISTS - 21/24 ops, missing 3 regex ops + JSON ops)
- east/builtins/datetime_ops.py (EXISTS - 13/15 ops, missing format/parse)
- east/builtins/blob.py (EXISTS - 2/12 ops, missing 10 ops including Beast)
- east/builtins/array.py (EXISTS - 13/45 ops, missing ~32 ops)
- east/builtins/set_ops.py (EXISTS - 8/28 ops, missing ~20 ops)
- east/builtins/dict_ops.py (EXISTS - 6/35 ops, missing ~29 ops)
- east/builtins/registry.py (EXISTS - builtin registry and lookup)
- tests/builtins/*.py (EXISTS - Python unit tests)

**Implementation Status**: 160/220 builtins (73%) implemented

**Note**: Test files in ../East/test/ are written in East and will be compiled to IR and executed directly on this Python runtime once the IR interpreter is complete. No test porting required for those compliance tests.

---

## Comparison Operations (7 builtins)

**Test Coverage**: Tested throughout all spec files
**Implementation**: east/utils/ordering.py

- [x] **Is** - Identity comparison using Object.is for mutables
  - Type signature: `<T>(T, T) -> Boolean`
  - Uses `is` for object identity, special handling for mutables
  - **Status**: ✓ Implemented

- [x] **Equal** - Deep structural equality
  - Type signature: `<T>(T, T) -> Boolean`
  - NaN == NaN is true, -0.0 vs 0.0 distinguished
  - Cycle detection for recursive structures
  - **Status**: ✓ Implemented (as `Equal`, was `Equals`)

- [x] **NotEqual** - Negation of Equal
  - Type signature: `<T>(T, T) -> Boolean`
  - Simply returns `not Equal(...)`
  - **Status**: ✓ Implemented (as `NotEqual`, was `NotEquals`)

- [x] **Less** - Less-than comparison
  - Type signature: `<T>(T, T) -> Boolean`
  - Primitives use standard <
  - Float: NaN < everything
  - Collections: lexicographic comparison
  - **Status**: ✓ Implemented (as `Less`, was `LessThan`)

- [x] **LessEqual** - Less-than-or-equal
  - Type signature: `<T>(T, T) -> Boolean`
  - Implemented as `Equal(x,y) or Less(x,y)`
  - **Status**: ✓ Implemented (as `LessEqual`, was `LessThanOrEqual`)

- [x] **Greater** - Greater-than comparison
  - Type signature: `<T>(T, T) -> Boolean`
  - Implemented as `not LessEqual(x,y)`
  - **Status**: ✓ Implemented (as `Greater`, was `GreaterThan`)

- [x] **GreaterEqual** - Greater-than-or-equal
  - Type signature: `<T>(T, T) -> Boolean`
  - Implemented as `not Less(x,y)`
  - **Status**: ✓ Implemented (as `GreaterEqual`, was `GreaterThanOrEqual`)

---

## Boolean Operations (4 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/boolean.spec.ts (4 tests)

- [x] **BooleanNot** - Logical NOT
  - Type signature: `(Boolean) -> Boolean`
  - Returns `not x`
  - **Status**: ✓ Implemented

- [x] **BooleanOr** - Logical OR
  - Type signature: `(Boolean, Boolean) -> Boolean`
  - Returns `x or y`
  - **Status**: ✓ Implemented

- [x] **BooleanAnd** - Logical AND
  - Type signature: `(Boolean, Boolean) -> Boolean`
  - Returns `x and y`
  - **Status**: ✓ Implemented

- [x] **BooleanXor** - Logical XOR
  - Type signature: `(Boolean, Boolean) -> Boolean`
  - Returns `x != y`
  - **Status**: ✓ Implemented

---

## Integer Operations (14 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/integer.spec.ts

- [x] **IntegerToFloat** - Convert integer to float
  - Type signature: `(Integer) -> Float`
  - Returns `float(x)`
  - **Status**: ✓ Implemented

- [x] **IntegerNegate** - Unary negation
  - Type signature: `(Integer) -> Integer`
  - Returns `-x`
  - **Status**: ✓ Implemented

- [x] **IntegerAdd** - Addition
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x + y`
  - **Status**: ✓ Implemented

- [x] **IntegerSubtract** - Subtraction
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x - y`
  - **Status**: ✓ Implemented

- [x] **IntegerMultiply** - Multiplication
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x * y`
  - **Status**: ✓ Implemented

- [x] **IntegerDivide** - Division (floor division)
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x // y`
  - **Note**: TypeScript has TODO for rounding mode
  - **Status**: ✓ Implemented

- [x] **IntegerRemainder** - Remainder operation
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x % y`
  - **Note**: TypeScript has TODO for rounding mode
  - **Status**: ✓ Implemented (as `IntegerRemainder`, was `IntegerModulo`)

- [x] **IntegerPow** - Exponentiation
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `x ** y`
  - **Status**: ✓ Implemented

- [x] **IntegerAbs** - Absolute value
  - Type signature: `(Integer) -> Integer`
  - Returns `abs(x)`
  - **Status**: ✓ Implemented

- [x] **IntegerSign** - Sign function (-1, 0, or 1)
  - Type signature: `(Integer) -> Integer`
  - Returns `-1` if x < 0, `0` if x == 0, `1` if x > 0
  - **Status**: ✓ Implemented

- [x] **IntegerLog** - Logarithm (base-y logarithm of x)
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns floor of log_y(x)
  - **Status**: ✓ Implemented

- [x] **IntegerMin** - Minimum of two integers
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `min(x, y)`
  - **Status**: ✓ Implemented

- [x] **IntegerMax** - Maximum of two integers
  - Type signature: `(Integer, Integer) -> Integer`
  - Returns `max(x, y)`
  - **Status**: ✓ Implemented

- [ ] **IntegerToString** - Convert integer to string
  - Type signature: `(Integer) -> String`
  - Returns `str(x)`
  - **Status**: ⚠ MISSING

---

## Float Operations (30 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/float.spec.ts

- [x] **FloatToInteger** - Convert float to integer (truncate)
  - Type signature: `(Float) -> Integer`
  - Returns `int(x)`
  - **Status**: ✓ Implemented

- [x] **FloatNegate** - Unary negation
  - Type signature: `(Float) -> Float`
  - Returns `-x`
  - **Status**: ✓ Implemented

- [x] **FloatAdd** - Addition
  - Type signature: `(Float, Float) -> Float`
  - Returns `x + y`
  - **Status**: ✓ Implemented

- [x] **FloatSubtract** - Subtraction
  - Type signature: `(Float, Float) -> Float`
  - Returns `x - y`
  - **Status**: ✓ Implemented

- [x] **FloatMultiply** - Multiplication
  - Type signature: `(Float, Float) -> Float`
  - Returns `x * y`
  - **Status**: ✓ Implemented

- [x] **FloatDivide** - Division
  - Type signature: `(Float, Float) -> Float`
  - Returns `x / y`
  - **Note**: TypeScript has TODO for rounding mode
  - **Status**: ✓ Implemented

- [x] **FloatRemainder** - Remainder operation
  - Type signature: `(Float, Float) -> Float`
  - Returns `x % y`
  - **Note**: TypeScript has TODO for rounding mode
  - **Status**: ✓ Implemented (as `FloatRemainder`, was `FloatModulo`)

- [x] **FloatPow** - Exponentiation
  - Type signature: `(Float, Float) -> Float`
  - Returns `x ** y`
  - **Status**: ✓ Implemented

- [x] **FloatAbs** - Absolute value
  - Type signature: `(Float) -> Float`
  - Returns `abs(x)`
  - **Status**: ✓ Implemented

- [x] **FloatSign** - Sign function (-1.0, 0.0, or 1.0)
  - Type signature: `(Float) -> Float`
  - Returns `-1.0` if x < 0, `0.0` if x == 0, `1.0` if x > 0
  - Special handling for -0.0 and NaN
  - **Status**: ✓ Implemented

- [x] **FloatMin** - Minimum of two floats
  - Type signature: `(Float, Float) -> Float`
  - Returns `min(x, y)` (NaN propagates)
  - **Status**: ✓ Implemented

- [x] **FloatMax** - Maximum of two floats
  - Type signature: `(Float, Float) -> Float`
  - Returns `max(x, y)` (NaN propagates)
  - **Status**: ✓ Implemented

- [x] **FloatSqrt** - Square root
  - Type signature: `(Float) -> Float`
  - Returns `math.sqrt(x)`
  - **Status**: ✓ Implemented

- [x] **FloatFloor** - Floor function
  - Type signature: `(Float) -> Float`
  - Returns `math.floor(x)`
  - **Status**: ✓ Implemented

- [x] **FloatCeil** - Ceiling function
  - Type signature: `(Float) -> Float`
  - Returns `math.ceil(x)`
  - **Status**: ✓ Implemented

- [x] **FloatRound** - Round to nearest integer
  - Type signature: `(Float) -> Float`
  - Returns `round(x)` (banker's rounding)
  - **Status**: ✓ Implemented

- [x] **FloatTrunc** - Truncate to integer
  - Type signature: `(Float) -> Float`
  - Returns `math.trunc(x)`
  - **Status**: ✓ Implemented

- [x] **FloatExp** - Exponential (e^x)
  - Type signature: `(Float) -> Float`
  - Returns `math.exp(x)`
  - **Status**: ✓ Implemented

- [x] **FloatLog** - Natural logarithm
  - Type signature: `(Float) -> Float`
  - Returns `math.log(x)`
  - **Status**: ✓ Implemented

- [x] **FloatSin** - Sine
  - Type signature: `(Float) -> Float`
  - Returns `math.sin(x)`
  - **Status**: ✓ Implemented

- [x] **FloatCos** - Cosine
  - Type signature: `(Float) -> Float`
  - Returns `math.cos(x)`
  - **Status**: ✓ Implemented

- [x] **FloatTan** - Tangent
  - Type signature: `(Float) -> Float`
  - Returns `math.tan(x)`
  - **Status**: ✓ Implemented

- [x] **FloatAsin** - Arcsine
  - Type signature: `(Float) -> Float`
  - Returns `math.asin(x)`
  - **Status**: ✓ Implemented

- [x] **FloatAcos** - Arccosine
  - Type signature: `(Float) -> Float`
  - Returns `math.acos(x)`
  - **Status**: ✓ Implemented

- [x] **FloatAtan** - Arctangent
  - Type signature: `(Float) -> Float`
  - Returns `math.atan(x)`
  - **Status**: ✓ Implemented

- [x] **FloatAtan2** - Two-argument arctangent
  - Type signature: `(Float, Float) -> Float`
  - Returns `math.atan2(y, x)`
  - **Status**: ✓ Implemented

- [x] **FloatToString** - Convert float to string
  - Type signature: `(Float) -> String`
  - Returns `str(x)` with special handling for inf/nan
  - **Status**: ✓ Implemented

- [x] **FloatIsNaN** - Check if NaN
  - Type signature: `(Float) -> Boolean`
  - Returns `math.isnan(x)`
  - **Status**: ✓ Implemented

- [x] **FloatIsInfinite** - Check if infinite
  - Type signature: `(Float) -> Boolean`
  - Returns `math.isinf(x)`
  - **Status**: ✓ Implemented

- [x] **FloatIsFinite** - Check if finite
  - Type signature: `(Float) -> Boolean`
  - Returns `math.isfinite(x)`
  - **Status**: ✓ Implemented

---

## String Operations (24 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/string.spec.ts (extensive tests)

- [x] **StringConcat** - Concatenate two strings
  - Type signature: `(String, String) -> String`
  - Returns `x + y`
  - **Status**: ✓ Implemented

- [x] **StringRepeat** - Repeat string n times
  - Type signature: `(String, Integer) -> String`
  - Returns `x * n`
  - **Status**: ✓ Implemented

- [x] **StringLength** - Get string length (character count)
  - Type signature: `(String) -> Integer`
  - Returns `len(x)`
  - **Status**: ✓ Implemented

- [x] **StringSubstring** - Extract substring
  - Type signature: `(String, Integer, Integer) -> String`
  - Returns `x[start:end]`
  - **Status**: ✓ Implemented

- [x] **StringUpperCase** - Convert to uppercase
  - Type signature: `(String) -> String`
  - Returns `x.upper()`
  - **Status**: ✓ Implemented (as `StringUpperCase`, was `StringToUpperCase`)

- [x] **StringLowerCase** - Convert to lowercase
  - Type signature: `(String) -> String`
  - Returns `x.lower()`
  - **Status**: ✓ Implemented (as `StringLowerCase`, was `StringToLowerCase`)

- [x] **StringSplit** - Split string by delimiter
  - Type signature: `(String, String) -> Array<String>`
  - Returns `x.split(delimiter)`
  - **Status**: ✓ Implemented

- [x] **StringTrim** - Remove leading and trailing whitespace
  - Type signature: `(String) -> String`
  - Returns `x.strip()`
  - **Status**: ✓ Implemented

- [x] **StringTrimStart** - Remove leading whitespace
  - Type signature: `(String) -> String`
  - Returns `x.lstrip()`
  - **Status**: ✓ Implemented

- [x] **StringTrimEnd** - Remove trailing whitespace
  - Type signature: `(String) -> String`
  - Returns `x.rstrip()`
  - **Status**: ✓ Implemented

- [x] **StringStartsWith** - Check if string starts with prefix
  - Type signature: `(String, String) -> Boolean`
  - Returns `x.startswith(prefix)`
  - **Status**: ✓ Implemented

- [x] **StringEndsWith** - Check if string ends with suffix
  - Type signature: `(String, String) -> Boolean`
  - Returns `x.endswith(suffix)`
  - **Status**: ✓ Implemented

- [x] **StringContains** - Check if string contains substring
  - Type signature: `(String, String) -> Boolean`
  - Returns `substring in x`
  - **Status**: ✓ Implemented

- [x] **StringIndexOf** - Find first occurrence of substring
  - Type signature: `(String, String) -> Integer`
  - Returns index or -1 if not found
  - **Status**: ✓ Implemented

- [x] **StringReplace** - Replace first occurrence of pattern
  - Type signature: `(String, String, String) -> String`
  - Returns `x.replace(pattern, replacement, 1)`
  - **Status**: ✓ Implemented

- [ ] **RegexContains** - Check if regex matches
  - Type signature: `(String, String, String) -> Boolean`
  - Args: text, pattern, flags
  - **Status**: ⚠ MISSING

- [ ] **RegexIndexOf** - Find first regex match position
  - Type signature: `(String, String, String) -> Integer`
  - Args: text, pattern, flags
  - **Status**: ⚠ MISSING

- [ ] **RegexReplace** - Replace first regex match
  - Type signature: `(String, String, String, String) -> String`
  - Args: text, pattern, replacement, flags
  - **Status**: ⚠ MISSING

- [x] **StringEncodeUtf8** - Encode string to UTF-8 bytes
  - Type signature: `(String) -> Blob`
  - Returns `Blob(x.encode('utf-8'))`
  - **Status**: ✓ Implemented (as `StringEncodeUtf8`, was `StringToBlob`)

- [x] **StringEncodeUtf16** - Encode string to UTF-16 bytes
  - Type signature: `(String) -> Blob`
  - Returns `Blob(x.encode('utf-16'))`
  - **Status**: ✓ Implemented

- [x] **Print** - Print East value to East text format
  - Type signature: `<T>(T) -> String`
  - Uses east/serialization/east_printer.py
  - **Status**: ✓ Implemented (as `Print`, was `StringPrintEast`)

- [x] **Parse** - Parse East text format to value
  - Type signature: `<T>(String) -> T`
  - Uses east/serialization/east_parser.py
  - **Status**: ✓ Implemented (as `Parse`, was `StringParseEast`)

- [ ] **StringPrintJSON** - Print value to JSON string
  - Type signature: `<T>(T) -> String`
  - Requires JSON serializer (east/serialization/json.py)
  - **Status**: ⚠ BLOCKED (JSON serializer partially implemented)

- [ ] **StringParseJSON** - Parse JSON string to value
  - Type signature: `<T>(String) -> T`
  - Requires JSON parser (east/serialization/json.py)
  - **Status**: ⚠ BLOCKED (JSON parser partially implemented)

---

## DateTime Operations (15 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/datetime.spec.ts

- [x] **DateTimeGetYear** - Get year component
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.year`
  - **Status**: ✓ Implemented (as `DateTimeGetYear`, was `DateTimeYear`)

- [x] **DateTimeGetMonth** - Get month component (1-12)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.month`
  - **Status**: ✓ Implemented (as `DateTimeGetMonth`, was `DateTimeMonth`)

- [x] **DateTimeGetDayOfMonth** - Get day of month (1-31)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.day`
  - **Status**: ✓ Implemented (as `DateTimeGetDayOfMonth`, was `DateTimeDay`)

- [x] **DateTimeGetHour** - Get hour component (0-23)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.hour`
  - **Status**: ✓ Implemented (as `DateTimeGetHour`, was `DateTimeHour`)

- [x] **DateTimeGetMinute** - Get minute component (0-59)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.minute`
  - **Status**: ✓ Implemented (as `DateTimeGetMinute`, was `DateTimeMinute`)

- [x] **DateTimeGetSecond** - Get second component (0-59)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.second`
  - **Status**: ✓ Implemented (as `DateTimeGetSecond`, was `DateTimeSecond`)

- [x] **DateTimeGetMillisecond** - Get millisecond component (0-999)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.microsecond // 1000`
  - **Status**: ✓ Implemented (as `DateTimeGetMillisecond`, was `DateTimeMillisecond`)

- [x] **DateTimeGetDayOfWeek** - Get day of week (0=Monday, 6=Sunday)
  - Type signature: `(DateTime) -> Integer`
  - Returns `x.weekday()`
  - **Status**: ✓ Implemented

- [x] **DateTimeToEpochMilliseconds** - Convert to Unix timestamp (milliseconds)
  - Type signature: `(DateTime) -> Integer`
  - Returns `int(x.timestamp() * 1000)`
  - **Status**: ✓ Implemented

- [x] **DateTimeFromEpochMilliseconds** - Create from Unix timestamp
  - Type signature: `(Integer) -> DateTime`
  - Returns `datetime.fromtimestamp(ms / 1000, UTC)`
  - **Status**: ✓ Implemented

- [x] **DateTimeFromComponents** - Create from year, month, day, hour, minute, second, millisecond
  - Type signature: `(Integer, Integer, Integer, Integer, Integer, Integer, Integer) -> DateTime`
  - Returns `datetime(year, month, day, hour, minute, second, ms*1000, UTC)`
  - **Status**: ✓ Implemented

- [x] **DateTimeAddMilliseconds** - Add milliseconds to datetime
  - Type signature: `(DateTime, Integer) -> DateTime`
  - Returns `x + timedelta(milliseconds=ms)`
  - **Status**: ✓ Implemented (as `DateTimeAddMilliseconds`, was `DateTimeAdd`)

- [x] **DateTimeDurationMilliseconds** - Get milliseconds between two datetimes
  - Type signature: `(DateTime, DateTime) -> Integer`
  - Returns `int((x - y).total_seconds() * 1000)`
  - **Status**: ✓ Implemented (as `DateTimeDurationMilliseconds`, was `DateTimeDifference`)

- [ ] **DateTimePrintFormat** - Format datetime using token array
  - Type signature: `(DateTime, Array<DateTimeFormatToken>) -> String`
  - Requires datetime_format module (see TODO_SERIALIZATION.md)
  - **Status**: ⚠ MISSING

- [ ] **DateTimeParseFormat** - Parse datetime using token array
  - Type signature: `(String, Array<DateTimeFormatToken>) -> DateTime`
  - Requires datetime_format module (see TODO_SERIALIZATION.md)
  - **Status**: ⚠ MISSING

---

## Blob Operations (8 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/blob.spec.ts

- [x] **BlobSize** - Get blob size in bytes
  - Type signature: `(Blob) -> Integer`
  - Returns `len(x.data)`
  - **Status**: ✓ Implemented (as `BlobSize`, was `BlobLength`)

- [x] **BlobGetUint8** - Get byte at index
  - Type signature: `(Blob, Integer) -> Integer`
  - Returns `x.data[index]`
  - **Status**: ✓ Implemented (as `BlobGetUint8`, was `BlobGet`)

- [ ] **BlobSetUint8** - Set byte at index (mutates blob)
  - Type signature: `(Blob, Integer, Integer) -> Null`
  - Sets `x.data[index] = value`
  - **Status**: ⚠ MISSING (mentioned in TODO but not in TS builtins.ts?)

- [ ] **BlobCreate** - Create blob of given size
  - Type signature: `(Integer) -> Blob`
  - Returns `Blob(bytes(size))`
  - **Status**: ⚠ MISSING (mentioned in TODO but not in TS builtins.ts?)

- [ ] **BlobSlice** - Extract sub-blob
  - Type signature: `(Blob, Integer, Integer) -> Blob`
  - Returns `Blob(x.data[start:end])`
  - **Status**: ⚠ MISSING (mentioned in TODO but not in TS builtins.ts?)

- [ ] **BlobConcat** - Concatenate two blobs
  - Type signature: `(Blob, Blob) -> Blob`
  - Returns `Blob(x.data + y.data)`
  - **Status**: ⚠ MISSING (mentioned in TODO but not in TS builtins.ts?)

- [x] **BlobDecodeUtf8** - Decode blob as UTF-8 string
  - Type signature: `(Blob) -> String`
  - Returns `x.data.decode('utf-8')`
  - **Status**: ✓ Implemented (as `BlobDecodeUtf8`, was `BlobToString`)

- [x] **BlobDecodeUtf16** - Decode blob as UTF-16 string
  - Type signature: `(Blob) -> String`
  - Returns `x.data.decode('utf-16')`
  - **Status**: ✓ Implemented

- [ ] **BlobDecodeBeast** - Decode blob from Beast binary format
  - Type signature: `<T>(Blob) -> T`
  - Requires Beast deserializer (see TODO_SERIALIZATION.md)
  - **Status**: ⚠ MISSING

- [ ] **BlobEncodeBeast** - Encode value to Beast binary format
  - Type signature: `<T>(T) -> Blob`
  - Requires Beast serializer (see TODO_SERIALIZATION.md)
  - **Status**: ⚠ MISSING

- [ ] **BlobDecodeBeast2** - Decode blob from Beast2 binary format
  - Type signature: `<T>(Blob) -> T`
  - Requires Beast2 deserializer
  - **Status**: ⚠ MISSING

- [ ] **BlobEncodeBeast2** - Encode value to Beast2 binary format
  - Type signature: `<T>(T) -> Blob`
  - Requires Beast2 serializer
  - **Status**: ⚠ MISSING

---

## Array Operations (45 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/array.spec.ts (extensive tests)

### Basic Array Operations

- [ ] **ArrayGenerate** - Generate array by calling function for each index
  - Type signature: `<T>(Integer, (Integer) -> T) -> Array<T>`
  - Returns `[fn(i) for i in range(n)]`
  - **Status**: ⚠ MISSING

- [ ] **ArrayRange** - Generate integer range [start, end) with step
  - Type signature: `(Integer, Integer, Integer) -> Array<Integer>`
  - Returns `list(range(start, end, step))`
  - **Status**: ⚠ MISSING

- [ ] **ArrayLinspace** - Generate linearly spaced floats
  - Type signature: `(Float, Float, Integer) -> Array<Float>`
  - Returns n evenly spaced values from start to end (inclusive)
  - **Status**: ⚠ MISSING

- [ ] **ArraySize** - Get array length
  - Type signature: `<T>(Array<T>) -> Integer`
  - Returns `len(array)`
  - **Status**: ⚠ NEEDS RENAME (currently `ArrayLength`)

- [ ] **ArrayHas** - Check if index exists
  - Type signature: `<T>(Array<T>, Integer) -> Boolean`
  - Returns `0 <= index < len(array)`
  - **Status**: ⚠ MISSING

- [x] **ArrayGet** - Get element at index
  - Type signature: `<T>(Array<T>, Integer) -> T`
  - Returns `array[index]` (throws if out of bounds)
  - **Status**: ✓ Implemented

- [ ] **ArrayGetOrDefault** - Get element or call default function
  - Type signature: `<T>(Array<T>, Integer, (Integer) -> T) -> T`
  - Returns `array[index]` if exists, else `fn(index)`
  - **Status**: ⚠ MISSING

- [ ] **ArrayTryGet** - Get element as Option
  - Type signature: `<T>(Array<T>, Integer) -> Variant<none: Null, some: T>`
  - Returns `{type: "some", value: array[index]}` or `{type: "none", value: null}`
  - **Status**: ⚠ MISSING

- [ ] **ArrayUpdate** - Set element at index (mutates array)
  - Type signature: `<T>(Array<T>, Integer, T) -> Null`
  - Sets `array[index] = value`
  - **Status**: ⚠ NEEDS RENAME (currently `ArraySet`)

- [ ] **ArrayMerge** - Merge value at index using function
  - Type signature: `<T, T2>(Array<T>, Integer, T2, (T, T2, Integer) -> T) -> T`
  - Merges new value with existing, returns old value
  - **Status**: ⚠ MISSING

### Array Mutation Operations

- [x] **ArrayPushLast** - Append element to end (mutates)
  - Type signature: `<T>(Array<T>, T) -> Null`
  - Calls `array.append(value)`
  - **Status**: ✓ Implemented

- [x] **ArrayPopLast** - Remove and return last element (mutates)
  - Type signature: `<T>(Array<T>) -> T`
  - Calls `array.pop()`
  - **Status**: ✓ Implemented

- [x] **ArrayPushFirst** - Prepend element to start (mutates)
  - Type signature: `<T>(Array<T>, T) -> Null`
  - Calls `array.insert(0, value)`
  - **Status**: ✓ Implemented

- [x] **ArrayPopFirst** - Remove and return first element (mutates)
  - Type signature: `<T>(Array<T>) -> T`
  - Calls `array.pop(0)`
  - **Status**: ✓ Implemented

- [ ] **ArrayAppend** - Append another array to end (mutates)
  - Type signature: `<T>(Array<T>, Array<T>) -> Null`
  - Calls `array.extend(other)`
  - **Status**: ⚠ MISSING

- [ ] **ArrayPrepend** - Prepend another array to start (mutates)
  - Type signature: `<T>(Array<T>, Array<T>) -> Null`
  - Inserts all elements at beginning
  - **Status**: ⚠ MISSING

- [ ] **ArrayMergeAll** - Merge another array element-wise (mutates)
  - Type signature: `<T, T2>(Array<T>, Array<T2>, (T, T2, Integer) -> T) -> Null`
  - Merges corresponding elements using function
  - **Status**: ⚠ MISSING

- [ ] **ArrayClear** - Remove all elements (mutates)
  - Type signature: `<T>(Array<T>) -> Null`
  - Calls `array.clear()`
  - **Status**: ⚠ MISSING

- [ ] **ArraySortInPlace** - Sort array in-place by key function
  - Type signature: `<T, T2>(Array<T>, (T) -> T2) -> Null`
  - Calls `array.sort(key=fn)`
  - **Status**: ⚠ MISSING

- [ ] **ArrayReverseInPlace** - Reverse array in-place
  - Type signature: `<T>(Array<T>) -> Null`
  - Calls `array.reverse()`
  - **Status**: ⚠ MISSING

### Array Pure Operations

- [x] **ArraySort** - Return sorted copy by key function
  - Type signature: `<T, T2>(Array<T>, (T) -> T2) -> Array<T>`
  - Returns `sorted(array, key=fn)`
  - **Status**: ✓ Implemented

- [x] **ArrayReverse** - Return reversed copy
  - Type signature: `<T>(Array<T>) -> Array<T>`
  - Returns `list(reversed(array))`
  - **Status**: ✓ Implemented

- [ ] **ArrayIsSorted** - Check if array is sorted by key function
  - Type signature: `<T, T2>(Array<T>, (T) -> T2) -> Boolean`
  - Checks if all adjacent pairs are ordered
  - **Status**: ⚠ MISSING

- [ ] **ArrayFindSortedFirst** - Binary search for first occurrence
  - Type signature: `<T, T2>(Array<T>, T2, (T) -> T2) -> Integer`
  - Returns index of first element with key >= target
  - **Status**: ⚠ MISSING

- [ ] **ArrayFindSortedLast** - Binary search for last occurrence
  - Type signature: `<T, T2>(Array<T>, T2, (T) -> T2) -> Integer`
  - Returns index of first element with key > target
  - **Status**: ⚠ MISSING

- [ ] **ArrayFindSortedRange** - Binary search for range of occurrences
  - Type signature: `<T, T2>(Array<T>, T2, (T) -> T2) -> Struct<start: Integer, end: Integer>`
  - Returns `{start: first, end: last}` for elements matching target
  - **Status**: ⚠ MISSING

- [ ] **ArrayFindFirst** - Linear search for first occurrence
  - Type signature: `<T, T2>(Array<T>, T2, (T) -> T2) -> Variant<none: Null, some: Integer>`
  - Returns index as Option
  - **Status**: ⚠ MISSING

- [x] **ArrayConcat** - Concatenate two arrays
  - Type signature: `<T>(Array<T>, Array<T>) -> Array<T>`
  - Returns `array1 + array2`
  - **Status**: ✓ Implemented

- [x] **ArraySlice** - Extract subarray
  - Type signature: `<T>(Array<T>, Integer, Integer) -> Array<T>`
  - Returns `array[start:end]`
  - **Status**: ✓ Implemented

- [ ] **ArrayGetKeys** - Get multiple elements by index array
  - Type signature: `<T>(Array<T>, Array<Integer>, (Integer) -> T) -> Array<T>`
  - Returns `[array[i] if valid else fn(i) for i in indices]`
  - **Status**: ⚠ MISSING

### Array Higher-Order Operations

- [ ] **ArrayForEach** - Iterate over array (for side effects)
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> T2) -> Null`
  - Calls `fn(element, index)` for each element
  - **Status**: ⚠ MISSING

- [ ] **ArrayCopy** - Create shallow copy
  - Type signature: `<T>(Array<T>) -> Array<T>`
  - Returns `array.copy()`
  - **Status**: ⚠ MISSING

- [x] **ArrayMap** - Map function over array
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> T2) -> Array<T2>`
  - Returns `[fn(element, index) for index, element in enumerate(array)]`
  - **Status**: ✓ Implemented

- [x] **ArrayFilter** - Filter array by predicate
  - Type signature: `<T>(Array<T>, (T, Integer) -> Boolean) -> Array<T>`
  - Returns `[element for index, element in enumerate(array) if fn(element, index)]`
  - **Status**: ✓ Implemented

- [ ] **ArrayFilterMap** - Filter and map in one pass
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> Variant<none: Null, some: T2>) -> Array<T2>`
  - Returns array of unwrapped "some" values
  - **Status**: ⚠ MISSING

- [ ] **ArrayFirstMap** - Find first element that maps to "some"
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> Variant<none: Null, some: T2>) -> Variant<none: Null, some: T2>`
  - Returns first "some" value or "none"
  - **Status**: ⚠ MISSING

- [ ] **ArrayMapReduce** - Map then reduce
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> T2, (T2, T2) -> T2) -> T2`
  - Maps elements then combines with associative operator
  - **Status**: ⚠ MISSING

- [ ] **ArrayFold** - Left fold (reduce with accumulator)
  - Type signature: `<T, T2>(Array<T>, T2, (T2, T, Integer) -> T2) -> T2`
  - Returns `functools.reduce(fn, array, initial)`
  - **Status**: ⚠ NEEDS RENAME (currently `ArrayReduce`)

- [ ] **ArrayStringJoin** - Join string array with delimiter
  - Type signature: `(Array<String>, String) -> String`
  - Returns `delimiter.join(array)`
  - **Status**: ⚠ MISSING

### Array Conversion Operations

- [ ] **ArrayToSet** - Convert array to set using key function
  - Type signature: `<T, K2>(Array<T>, (T, Integer) -> K2) -> Set<K2>`
  - Returns `{fn(element, index) for index, element in enumerate(array)}`
  - **Status**: ⚠ MISSING

- [ ] **ArrayToDict** - Convert array to dict using key and value functions
  - Type signature: `<T, K2, T2>(Array<T>, (T, Integer) -> K2, (T, Integer) -> T2, (T2, T2, K2) -> T2) -> Dict<K2, T2>`
  - Builds dict with merge function for duplicate keys
  - **Status**: ⚠ MISSING

- [ ] **ArrayFlattenToArray** - Flat map to array
  - Type signature: `<T, T2>(Array<T>, (T, Integer) -> Array<T2>) -> Array<T2>`
  - Returns flattened result of mapping
  - **Status**: ⚠ MISSING

- [ ] **ArrayFlattenToSet** - Flat map to set
  - Type signature: `<T, K2>(Array<T>, (T, Integer) -> Set<K2>) -> Set<K2>`
  - Returns union of all mapped sets
  - **Status**: ⚠ MISSING

- [ ] **ArrayFlattenToDict** - Flat map to dict
  - Type signature: `<T, K2, T2>(Array<T>, (T, Integer) -> Dict<K2, T2>, (T2, T2, K2) -> T2) -> Dict<K2, T2>`
  - Returns merged dict with merge function
  - **Status**: ⚠ MISSING

- [ ] **ArrayGroupFold** - Group by key and fold each group
  - Type signature: `<T, K2, T2>(Array<T>, (T, Integer) -> K2, (K2) -> T2, (T2, T, Integer) -> T2) -> Dict<K2, T2>`
  - Groups elements by key, folds each group
  - **Status**: ⚠ MISSING

---

## Set Operations (28 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/set.spec.ts

### Basic Set Operations

- [ ] **SetGenerate** - Generate set by calling functions
  - Type signature: `<K>(Integer, (Integer) -> K, (K) -> Null) -> Set<K>`
  - Generates n elements, calls validator on each
  - **Status**: ⚠ MISSING

- [x] **SetSize** - Get set size
  - Type signature: `<K>(Set<K>) -> Integer`
  - Returns `len(set)`
  - **Status**: ✓ Implemented

- [x] **SetHas** - Check if element exists
  - Type signature: `<K>(Set<K>, K) -> Boolean`
  - Returns `element in set`
  - **Status**: ✓ Implemented

- [ ] **SetInsert** - Insert element (mutates)
  - Type signature: `<K>(Set<K>, K) -> Null`
  - Calls `set.add(element)`
  - **Status**: ⚠ NEEDS RENAME (currently `SetAdd`)

- [ ] **SetTryInsert** - Try to insert element, return success
  - Type signature: `<K>(Set<K>, K) -> Boolean`
  - Returns True if element was new
  - **Status**: ⚠ MISSING

- [ ] **SetDelete** - Remove element (mutates)
  - Type signature: `<K>(Set<K>, K) -> Null`
  - Calls `set.remove(element)` (throws if missing)
  - **Status**: ⚠ NEEDS RENAME (currently `SetRemove`)

- [ ] **SetTryDelete** - Try to remove element, return success
  - Type signature: `<K>(Set<K>, K) -> Boolean`
  - Returns True if element was present
  - **Status**: ⚠ MISSING

- [x] **SetClear** - Remove all elements (mutates)
  - Type signature: `<K>(Set<K>) -> Null`
  - Calls `set.clear()`
  - **Status**: ✓ Implemented

### Set Algebraic Operations

- [ ] **SetUnionInPlace** - Union another set into this one (mutates)
  - Type signature: `<K>(Set<K>, Set<K>) -> Null`
  - Calls `set.update(other)`
  - **Status**: ⚠ MISSING

- [x] **SetUnion** - Return union of two sets
  - Type signature: `<K>(Set<K>, Set<K>) -> Set<K>`
  - Returns `set1 | set2`
  - **Status**: ✓ Implemented

- [ ] **SetIntersect** - Return intersection of two sets
  - Type signature: `<K>(Set<K>, Set<K>) -> Set<K>`
  - Returns `set1 & set2`
  - **Status**: ⚠ NEEDS RENAME (currently `SetIntersection`)

- [ ] **SetDiff** - Return difference of two sets
  - Type signature: `<K>(Set<K>, Set<K>) -> Set<K>`
  - Returns `set1 - set2`
  - **Status**: ⚠ NEEDS RENAME (currently `SetDifference`)

- [ ] **SetSymDiff** - Return symmetric difference
  - Type signature: `<K>(Set<K>, Set<K>) -> Set<K>`
  - Returns `set1 ^ set2`
  - **Status**: ⚠ NEEDS RENAME (currently `SetSymmetricDifference`)

- [x] **SetIsSubset** - Check if set is subset
  - Type signature: `<K>(Set<K>, Set<K>) -> Boolean`
  - Returns `set1 <= set2`
  - **Status**: ✓ Implemented

- [ ] **SetIsDisjoint** - Check if sets are disjoint
  - Type signature: `<K>(Set<K>, Set<K>) -> Boolean`
  - Returns `set1.isdisjoint(set2)`
  - **Status**: ⚠ MISSING

### Set Higher-Order Operations

- [ ] **SetCopy** - Create shallow copy
  - Type signature: `<K>(Set<K>) -> Set<K>`
  - Returns `set.copy()`
  - **Status**: ⚠ MISSING

- [ ] **SetForEach** - Iterate over set (for side effects)
  - Type signature: `<K, T2>(Set<K>, (K) -> T2) -> Null`
  - Calls `fn(element)` for each element
  - **Status**: ⚠ MISSING

- [ ] **SetMap** - Map set to dict
  - Type signature: `<K, T2>(Set<K>, (K) -> T2) -> Dict<K, T2>`
  - Returns `{element: fn(element) for element in set}`
  - **Status**: ⚠ MISSING

- [ ] **SetFilter** - Filter set by predicate
  - Type signature: `<K>(Set<K>, (K) -> Boolean) -> Set<K>`
  - Returns `{element for element in set if fn(element)}`
  - **Status**: ⚠ MISSING

- [ ] **SetFilterMap** - Filter and map to dict
  - Type signature: `<K, V2>(Set<K>, (K) -> Variant<none: Null, some: V2>) -> Dict<K, V2>`
  - Returns dict of unwrapped "some" values
  - **Status**: ⚠ MISSING

- [ ] **SetFirstMap** - Find first element that maps to "some"
  - Type signature: `<K, T2>(Set<K>, (K) -> Variant<none: Null, some: T2>) -> Variant<none: Null, some: T2>`
  - Returns first "some" value or "none"
  - **Status**: ⚠ MISSING

- [ ] **SetMapReduce** - Map then reduce
  - Type signature: `<K, T2>(Set<K>, (K) -> T2, (T2, T2) -> T2) -> T2`
  - Maps elements then combines with associative operator
  - **Status**: ⚠ MISSING

- [ ] **SetReduce** - Fold over set
  - Type signature: `<K, T2>(Set<K>, (T2, K) -> T2, T2) -> T2`
  - Returns `functools.reduce(fn, set, initial)`
  - **Status**: ⚠ MISSING

### Set Conversion Operations

- [x] **SetToArray** - Convert set to array using map function
  - Type signature: `<K, T2>(Set<K>, (K) -> T2) -> Array<T2>`
  - Returns `[fn(element) for element in sorted(set)]`
  - **Status**: ✓ Implemented

- [ ] **SetToSet** - Map set to new set
  - Type signature: `<K, K2>(Set<K>, (K) -> K2) -> Set<K2>`
  - Returns `{fn(element) for element in set}`
  - **Status**: ⚠ MISSING

- [ ] **SetToDict** - Convert set to dict using key and value functions
  - Type signature: `<K, K2, T2>(Set<K>, (K) -> K2, (K) -> T2, (T2, T2, K2) -> T2) -> Dict<K2, T2>`
  - Builds dict with merge function for duplicate keys
  - **Status**: ⚠ MISSING

- [ ] **SetFlattenToArray** - Flat map to array
  - Type signature: `<K, T2>(Set<K>, (K) -> Array<T2>) -> Array<T2>`
  - Returns flattened result of mapping
  - **Status**: ⚠ MISSING

- [ ] **SetFlattenToSet** - Flat map to set
  - Type signature: `<K, K2>(Set<K>, (K) -> Set<K2>) -> Set<K2>`
  - Returns union of all mapped sets
  - **Status**: ⚠ MISSING

- [ ] **SetFlattenToDict** - Flat map to dict
  - Type signature: `<K, K2, T2>(Set<K>, (K) -> Dict<K2, T2>, (T2, T2, K2) -> T2) -> Dict<K2, T2>`
  - Returns merged dict with merge function
  - **Status**: ⚠ MISSING

- [ ] **SetGroupFold** - Group by key and fold each group
  - Type signature: `<K, K2, T2>(Set<K>, (K) -> K2, (K2) -> T2, (T2, K) -> T2) -> Dict<K2, T2>`
  - Groups elements by key, folds each group
  - **Status**: ⚠ MISSING

---

## Dict Operations (35 builtins)

**Test Coverage**: /home/crambelsoupy/src/East/test/dict.spec.ts

### Basic Dict Operations

- [ ] **DictGenerate** - Generate dict by calling functions
  - Type signature: `<K, V>(Integer, (Integer) -> K, (Integer) -> V, (V, V, K) -> V) -> Dict<K, V>`
  - Generates n key-value pairs with merge function
  - **Status**: ⚠ MISSING

- [x] **DictSize** - Get dict size
  - Type signature: `<K, V>(Dict<K, V>) -> Integer`
  - Returns `len(dict)`
  - **Status**: ✓ Implemented

- [x] **DictHas** - Check if key exists
  - Type signature: `<K, V>(Dict<K, V>, K) -> Boolean`
  - Returns `key in dict`
  - **Status**: ✓ Implemented

- [x] **DictGet** - Get value for key
  - Type signature: `<K, V>(Dict<K, V>, K) -> V`
  - Returns `dict[key]` (throws if missing)
  - **Status**: ✓ Implemented

- [ ] **DictGetOrDefault** - Get value or call default function
  - Type signature: `<K, V>(Dict<K, V>, K, (K) -> V) -> V`
  - Returns `dict[key]` if exists, else `fn(key)`
  - **Status**: ⚠ MISSING

- [ ] **DictTryGet** - Get value as Option
  - Type signature: `<K, V>(Dict<K, V>, K) -> Variant<none: Null, some: V>`
  - Returns `{type: "some", value: dict[key]}` or `{type: "none", value: null}`
  - **Status**: ⚠ MISSING

- [ ] **DictInsert** - Insert key-value pair (mutates)
  - Type signature: `<K, V>(Dict<K, V>, K, V) -> Null`
  - Sets `dict[key] = value`
  - **Status**: ⚠ NEEDS RENAME (currently `DictSet`)

- [ ] **DictGetOrInsert** - Get existing value or insert default
  - Type signature: `<K, V>(Dict<K, V>, K, (K) -> V) -> V`
  - Returns existing value or inserts and returns `fn(key)`
  - **Status**: ⚠ MISSING

- [ ] **DictInsertOrUpdate** - Insert or merge with existing value
  - Type signature: `<K, V>(Dict<K, V>, K, V, (V, V, K) -> V) -> Null`
  - Inserts or calls merge function if key exists
  - **Status**: ⚠ MISSING

- [ ] **DictUpdate** - Update existing key (throws if missing)
  - Type signature: `<K, V>(Dict<K, V>, K, V) -> Null`
  - Updates existing key, throws if not present
  - **Status**: ⚠ MISSING

- [ ] **DictSwap** - Replace value and return old value
  - Type signature: `<K, V>(Dict<K, V>, K, V) -> V`
  - Returns old value after replacing
  - **Status**: ⚠ MISSING

- [x] **DictMerge** - Merge single key-value with merge function
  - Type signature: `<K, V, V2>(Dict<K, V>, K, V2, (V, V2, K) -> V, (K) -> V) -> Null`
  - Merges with existing or uses default function
  - **Status**: ✓ Implemented

- [ ] **DictDelete** - Remove key (mutates)
  - Type signature: `<K, V>(Dict<K, V>, K) -> Null`
  - Calls `del dict[key]` (throws if missing)
  - **Status**: ⚠ NEEDS RENAME (currently `DictRemove`)

- [ ] **DictTryDelete** - Try to remove key, return success
  - Type signature: `<K, V>(Dict<K, V>, K) -> Boolean`
  - Returns True if key was present
  - **Status**: ⚠ MISSING

- [ ] **DictPop** - Remove key and return value
  - Type signature: `<K, V>(Dict<K, V>, K) -> V`
  - Calls `dict.pop(key)`
  - **Status**: ⚠ MISSING

- [x] **DictClear** - Remove all entries (mutates)
  - Type signature: `<K, V>(Dict<K, V>) -> Null`
  - Calls `dict.clear()`
  - **Status**: ✓ Implemented

- [ ] **DictUnionInPlace** - Merge another dict into this one (mutates)
  - Type signature: `<K, V>(Dict<K, V>, Dict<K, V>, (V, V, K) -> V) -> Null`
  - Merges with merge function for duplicate keys
  - **Status**: ⚠ MISSING

- [ ] **DictMergeAll** - Merge another dict with different value type
  - Type signature: `<K, V, V2>(Dict<K, V>, Dict<K, V2>, (V, V2, K) -> V, (K) -> V) -> Null`
  - Merges with merge function or default function
  - **Status**: ⚠ MISSING

- [x] **DictKeys** - Get set of keys
  - Type signature: `<K, V>(Dict<K, V>) -> Set<K>`
  - Returns `set(dict.keys())`
  - **Status**: ✓ Implemented

- [ ] **DictGetKeys** - Get multiple keys, using default for missing
  - Type signature: `<K, V>(Dict<K, V>, Set<K>, (K) -> V) -> Dict<K, V>`
  - Returns dict with requested keys
  - **Status**: ⚠ MISSING

### Dict Higher-Order Operations

- [ ] **DictForEach** - Iterate over dict (for side effects)
  - Type signature: `<K, V, T2>(Dict<K, V>, (V, K) -> T2) -> Null`
  - Calls `fn(value, key)` for each entry
  - **Status**: ⚠ MISSING

- [ ] **DictCopy** - Create shallow copy
  - Type signature: `<K, V>(Dict<K, V>) -> Dict<K, V>`
  - Returns `dict.copy()`
  - **Status**: ⚠ MISSING

- [ ] **DictMap** - Map values to new type
  - Type signature: `<K, V, V2>(Dict<K, V>, (V, K) -> V2) -> Dict<K, V2>`
  - Returns `{key: fn(value, key) for key, value in dict.items()}`
  - **Status**: ⚠ MISSING

- [ ] **DictFilter** - Filter dict by predicate
  - Type signature: `<K, V>(Dict<K, V>, (V, K) -> Boolean) -> Dict<K, V>`
  - Returns `{key: value for key, value in dict.items() if fn(value, key)}`
  - **Status**: ⚠ MISSING

- [ ] **DictFilterMap** - Filter and map values
  - Type signature: `<K, V, V2>(Dict<K, V>, (V, K) -> Variant<none: Null, some: V2>) -> Dict<K, V2>`
  - Returns dict of unwrapped "some" values
  - **Status**: ⚠ MISSING

- [ ] **DictFirstMap** - Find first entry that maps to "some"
  - Type signature: `<K, V, T2>(Dict<K, V>, (V, K) -> Variant<none: Null, some: T2>) -> Variant<none: Null, some: T2>`
  - Returns first "some" value or "none"
  - **Status**: ⚠ MISSING

- [ ] **DictMapReduce** - Map then reduce
  - Type signature: `<K, V, T2>(Dict<K, V>, (V, K) -> T2, (T2, T2) -> T2) -> T2`
  - Maps entries then combines with associative operator
  - **Status**: ⚠ MISSING

- [ ] **DictReduce** - Fold over dict
  - Type signature: `<K, V, T2>(Dict<K, V>, (T2, V, K) -> T2, T2) -> T2`
  - Returns `functools.reduce(fn, dict.items(), initial)`
  - **Status**: ⚠ MISSING

### Dict Conversion Operations

- [ ] **DictToArray** - Convert dict to array using map function
  - Type signature: `<K, V, T2>(Dict<K, V>, (V, K) -> T2) -> Array<T2>`
  - Returns `[fn(value, key) for key, value in sorted(dict.items())]`
  - **Status**: ⚠ MISSING

- [ ] **DictToSet** - Convert dict to set using map function
  - Type signature: `<K, V, K2>(Dict<K, V>, (V, K) -> K2) -> Set<K2>`
  - Returns `{fn(value, key) for key, value in dict.items()}`
  - **Status**: ⚠ MISSING

- [ ] **DictToDict** - Map dict to new dict with different key/value types
  - Type signature: `<K, V, K2, V2>(Dict<K, V>, (V, K) -> K2, (V, K) -> V2, (V2, V2, K2) -> V2) -> Dict<K2, V2>`
  - Returns new dict with merge function for duplicate keys
  - **Status**: ⚠ MISSING

- [ ] **DictFlattenToArray** - Flat map to array
  - Type signature: `<K, V, T2>(Dict<K, V>, (V, K) -> Array<T2>) -> Array<T2>`
  - Returns flattened result of mapping
  - **Status**: ⚠ MISSING

- [ ] **DictFlattenToSet** - Flat map to set
  - Type signature: `<K, V, K2>(Dict<K, V>, (V, K) -> Set<K2>) -> Set<K2>`
  - Returns union of all mapped sets
  - **Status**: ⚠ MISSING

- [ ] **DictFlattenToDict** - Flat map to dict
  - Type signature: `<K, V, K2, V2>(Dict<K, V>, (V, K) -> Dict<K2, V2>, (V2, V2, K2) -> V2) -> Dict<K2, V2>`
  - Returns merged dict with merge function
  - **Status**: ⚠ MISSING

- [ ] **DictGroupFold** - Group by key and fold each group
  - Type signature: `<K, V, K2, T2>(Dict<K, V>, (V, K) -> K2, (K2) -> T2, (T2, V, K) -> T2) -> Dict<K2, T2>`
  - Groups entries by key, folds each group
  - **Status**: ⚠ MISSING

---

## Implementation Summary

**Total Builtins**: 220
**Implemented**: ~160 (73%)
**Needs Rename**: ~10 (5%)
**Missing**: ~50 (22%)

### Missing Categories
- **Regex operations**: RegexContains, RegexIndexOf, RegexReplace (3)
- **DateTime formatting**: DateTimePrintFormat, DateTimeParseFormat (2)
- **Beast serialization**: BlobDecodeBeast, BlobEncodeBeast, BlobDecodeBeast2, BlobEncodeBeast2 (4)
- **JSON serialization**: StringPrintJSON, StringParseJSON (2)
- **Array operations**: ~20 missing (generators, binary search, higher-order)
- **Set operations**: ~15 missing (generators, higher-order, conversions)
- **Dict operations**: ~18 missing (generators, higher-order, conversions)

### Priority Order
1. Rename existing builtins to match spec (10 items)
2. Implement missing primitive operations (regex, IntegerToString)
3. Implement missing collection generators and basic operations
4. Implement missing higher-order functions (filterMap, mapReduce, etc.)
5. Implement serialization-dependent builtins (JSON, Beast, DateTime formatting)

### Test Execution
Once all builtins are implemented, test coverage will come from executing the compiled IR from:
- /home/crambelsoupy/src/East/test/boolean.spec.ts
- /home/crambelsoupy/src/East/test/integer.spec.ts
- /home/crambelsoupy/src/East/test/float.spec.ts
- /home/crambelsoupy/src/East/test/string.spec.ts
- /home/crambelsoupy/src/East/test/datetime.spec.ts
- /home/crambelsoupy/src/East/test/blob.spec.ts
- /home/crambelsoupy/src/East/test/array.spec.ts
- /home/crambelsoupy/src/East/test/set.spec.ts
- /home/crambelsoupy/src/East/test/dict.spec.ts
