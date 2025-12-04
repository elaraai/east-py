"""Fuzz testing utilities for East types.

Generates random types and values for property-based testing.
"""

import random
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from east.types.types import (
    ArrayType,
    BlobType,
    BooleanType,
    DateTimeType,
    DictType,
    EastType,
    FloatType,
    IntegerType,
    NullType,
    RefType,
    SetType,
    StringType,
    StructType,
    VariantType,
)
from east.types.values import EastArray, EastBlob, EastDict, EastSet


def random_type(depth: int = 0, exclude_types: list[str] | None = None) -> EastType:
    """Generate a random East type for fuzz testing.

    Args:
        depth: Current nesting depth (used internally to limit recursion)
        exclude_types: List of type names to exclude (e.g., ["Ref", "Function"])

    Returns:
        A randomly generated EastType

    Note:
        Types are kept reasonably simple to avoid generating huge nested structures:
        - Maximum nesting depth of 3 levels
        - Higher chance of primitives at deeper levels
        - Sets and Dicts use StringType keys (immutability constraint)
        - Structs have 0-4 random fields
        - Variants have 1-3 random cases
    """
    if exclude_types is None:
        exclude_types = []

    # Limit nesting to avoid stack overflow and keep tests fast
    max_depth = 3

    # Higher chance of primitives at deeper levels
    primitive_weight = 0.9 if depth >= max_depth else 0.5

    if random.random() < primitive_weight:
        # Primitive type
        r = random.random() * 7
        if r < 1:
            return NullType
        if r < 2:
            return BooleanType
        if r < 3:
            return IntegerType
        if r < 4:
            return FloatType
        if r < 5:
            return StringType
        if r < 6:
            return DateTimeType
        return BlobType

    # Build list of available complex types
    available_types = []
    if "Array" not in exclude_types:
        available_types.append("Array")
    if "Set" not in exclude_types:
        available_types.append("Set")
    if "Dict" not in exclude_types:
        available_types.append("Dict")
    if "Ref" not in exclude_types:
        available_types.append("Ref")
    if "Struct" not in exclude_types:
        available_types.append("Struct")
    if "Variant" not in exclude_types:
        available_types.append("Variant")

    if not available_types:
        # Fall back to primitive if all complex types excluded
        return IntegerType

    # Choose random complex type
    choice = random.choice(available_types)

    if choice == "Array":
        return ArrayType(random_type(depth + 1, exclude_types))
    if choice == "Set":
        return SetType(StringType)
    if choice == "Dict":
        return DictType(StringType, random_type(depth + 1, exclude_types))
    if choice == "Ref":
        return RefType(random_type(depth + 1, exclude_types))
    if choice == "Struct":
        field_count = random.randint(0, 4)
        fields = [(f"field{i}", random_type(depth + 1, exclude_types)) for i in range(field_count)]
        return StructType(fields)
    # Variant
    case_count = random.randint(1, 3)
    cases = [(f"case{i}", random_type(depth + 1, exclude_types)) for i in range(case_count)]
    return VariantType(cases)


def random_value_for(type_val: EastType) -> Callable[[], Any]:
    """Create a function that generates random values of a given type.

    Args:
        type_val: The type to generate values for

    Returns:
        A function that returns a new random value each time it's called

    Raises:
        ValueError: When the type is NeverType, FunctionType, or RecursiveType

    Note:
        Generates diverse test values for each type:
        - Floats include special values (NaN, ±Infinity, ±0.0)
        - Integers range from -100 to 100
        - Strings use random alphanumeric sequences (0-20 chars)
        - DateTimes are within one year of 2025-01-01
        - Collections have 0-4 elements (kept small for performance)
        - Variants randomly select one of their cases
    """
    type_kind = type_val.type

    if type_kind == "Never":
        raise ValueError("Cannot generate values for Never type")

    if type_kind == "Null":
        from east.types.values import east_null

        return lambda: east_null

    if type_kind == "Boolean":
        return lambda: random.random() < 0.5

    if type_kind == "Integer":
        return lambda: random.randint(-100, 100)

    if type_kind == "Float":

        def random_float() -> float:
            r = random.random()
            if r < 0.05:
                return float("nan")
            if r < 0.10:
                return float("inf")
            if r < 0.15:
                return float("-inf")
            if r < 0.20:
                return 0.0
            if r < 0.25:
                return -0.0
            return random.random() * 200 - 100

        return random_float

    if type_kind == "String":

        def random_string() -> str:
            length = random.randint(0, 20)
            if length == 0:
                return ""
            # Generate random alphanumeric string
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            return "".join(random.choice(chars) for _ in range(length))

        return random_string

    if type_kind == "DateTime":

        def random_datetime() -> datetime:
            year_2025 = datetime(2025, 1, 1, tzinfo=UTC)
            one_year_seconds = 365 * 24 * 60 * 60
            random_seconds = random.randint(0, one_year_seconds)
            return year_2025 + timedelta(seconds=random_seconds)

        return random_datetime

    if type_kind == "Blob":

        def random_blob() -> EastBlob:
            length = random.randint(0, 100)  # Keep small for speed
            data = bytes(random.randint(0, 255) for _ in range(length))
            return EastBlob(data)

        return random_blob

    if type_kind == "Array":
        item_fn = random_value_for(type_val.value)  # type: ignore[arg-type]

        def random_array() -> EastArray:
            length = random.randint(0, 4)
            items = [item_fn() for _ in range(length)]
            return EastArray(type_val.value, items)  # type: ignore[arg-type]

        return random_array

    if type_kind == "Set":
        item_fn = random_value_for(type_val.value)  # type: ignore[arg-type]

        def random_set() -> EastSet:
            length = random.randint(0, 4)
            items = [item_fn() for _ in range(length)]
            return EastSet(type_val.value, items)  # type: ignore[arg-type]

        return random_set

    if type_kind == "Dict":
        key_fn = random_value_for(type_val.value["key"])  # type: ignore[arg-type]
        value_fn = random_value_for(type_val.value["value"])  # type: ignore[arg-type]

        def random_dict() -> EastDict:
            length = random.randint(0, 4)
            items = {key_fn(): value_fn() for _ in range(length)}
            return EastDict(type_val.value["key"], type_val.value["value"], items)  # type: ignore[arg-type]

        return random_dict

    if type_kind == "Ref":
        from east.types.values import east_ref

        inner_fn = random_value_for(type_val.value)  # type: ignore[arg-type]

        def random_ref():
            # Generate random value for inner type
            inner_value = inner_fn()
            return east_ref(inner_value)

        return random_ref

    if type_kind == "Struct":
        # Get field names and create generators for each
        field_fns = {}
        for field_struct in type_val.value:
            field_name = field_struct["name"]
            field_type = field_struct["type"]
            field_fns[field_name] = random_value_for(field_type)

        def random_struct() -> dict[str, Any]:
            return {name: fn() for name, fn in field_fns.items()}

        return random_struct

    if type_kind == "Variant":
        # Get case names and create generators for each
        case_keys = []
        case_fns = {}
        cases = []
        for case_struct in type_val.value:
            case_name = case_struct["name"]
            case_type = case_struct["type"]
            case_keys.append(case_name)
            case_fns[case_name] = random_value_for(case_type)
            cases.append((case_name, case_type))

        def random_variant():
            case_key = random.choice(case_keys)
            case_value = case_fns[case_key]()
            return {"type": case_key, "value": case_value}

        return random_variant

    if type_kind == "Recursive":
        raise ValueError("Cannot generate values for Recursive type")

    if type_kind == "Function":
        raise ValueError("Cannot generate values for Function type")

    raise ValueError(f"Unhandled type: {type_kind}")


async def fuzzer_test(
    fn: Callable[[EastType], Callable[[Any], None]],
    n_types: int = 100,
    n_samples: int = 10,
    exclude_types: list[str] | None = None,
) -> bool:
    """Run a fuzz test over a generic function parameterized by a type.

    Args:
        fn: Factory function that takes a type and returns a test function for values of that type
        n_types: Number of random types to test
        n_samples: Number of random values to test per type
        exclude_types: List of type names to exclude from generation (e.g., ["Ref"])

    Returns:
        True if all tests passed, False if any failed

    Note:
        For each randomly generated type:
        1. Creates a test function using the provided factory
        2. Generates random values of that type
        3. Runs the test function on each value
        4. Reports any failures to stderr with type, value, and error details

        Attempts to generate unique types (up to 100 attempts per type) to maximize
        test coverage. Prints summary statistics showing success/failure counts.

    Example:
        >>> from east.serialization.json import to_json_for, from_json_for
        >>> async def test_json(type_val):
        ...     to_json = to_json_for(type_val)
        ...     from_json = from_json_for(type_val)
        ...     def test_value(value):
        ...         encoded = to_json(value)
        ...         decoded = from_json(encoded)
        ...         assert decoded == value
        ...     return test_value
        >>> await fuzzer_test(test_json, 100, 10)
        True
    """
    from east.serialization.east_printer import print_for, print_type

    n_type_success = 0
    n_type_fail = 0
    type_cache: set[str] = set()

    for _i in range(n_types):
        n_success = 0
        n_fail = 0

        # Generate a unique random type
        attempts = 0
        while True:
            type_val = random_type(exclude_types=exclude_types)
            type_str = print_type(type_val)
            if type_str not in type_cache:
                type_cache.add(type_str)
                break
            attempts += 1
            if attempts > 100:
                # Give up and allow duplicates
                break

        type_fn = fn(type_val)
        random_value = random_value_for(type_val)
        print_value = print_for(type_val)

        for _j in range(n_samples):
            value = random_value()
            try:
                type_fn(value)
                n_success += 1
            except Exception:
                n_fail += 1
                print(f"    Test failed for type {print_type(type_val)}", file=sys.stderr)
                print(f"    Value: {print_value(value)}", file=sys.stderr)
                import traceback

                print(f"    Error: {traceback.format_exc()}", file=sys.stderr)

        if n_fail > 0:
            n_type_fail += 1
            print(
                f"  FAILED: {n_success}/{n_samples} samples passed for type {print_type(type_val)}",
                file=sys.stderr,
            )
        else:
            n_type_success += 1

    if n_type_fail > 0:
        print(f"FAILED: {n_type_success}/{n_types} types passed", file=sys.stderr)
        return False
    return True


__all__ = ["random_type", "random_value_for", "fuzzer_test"]
