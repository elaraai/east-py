"""Integer builtin functions."""

from east.builtins.registry import register_builtin


def integer_add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        a + b
    """
    return a + b


def integer_subtract(a: int, b: int) -> int:
    """Subtract two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        a - b
    """
    return a - b


def integer_multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        a * b
    """
    return a * b


def integer_divide(a: int, b: int) -> int:
    """Integer division (floor division).

    Args:
        a: Dividend
        b: Divisor

    Returns:
        a // b (floor division)

    Raises:
        ZeroDivisionError: If b is zero
    """
    return a // b


def integer_modulo(a: int, b: int) -> int:
    """Integer modulo.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        a % b

    Raises:
        ZeroDivisionError: If b is zero
    """
    return a % b


def integer_power(a: int, b: int) -> int:
    """Integer exponentiation.

    Args:
        a: Base
        b: Exponent

    Returns:
        a ** b
    """
    return a**b


def integer_negate(a: int) -> int:
    """Negate an integer.

    Args:
        a: Integer to negate

    Returns:
        -a
    """
    return -a


def integer_abs(a: int) -> int:
    """Absolute value of an integer.

    Args:
        a: Integer

    Returns:
        |a|
    """
    return abs(a)


def integer_sign(a: int) -> int:
    """Sign of an integer.

    Args:
        a: Integer

    Returns:
        -1 if a < 0, 0 if a == 0, 1 if a > 0
    """
    if a < 0:
        return -1
    if a > 0:
        return 1
    return 0


def integer_log(a: int, base: int) -> int:
    """Integer logarithm (floor of log base b of a).

    Args:
        a: Value (must be positive)
        base: Logarithm base (must be > 1)

    Returns:
        floor(log_base(a))

    Raises:
        ValueError: If a <= 0 or base <= 1
    """
    if a <= 0:
        raise ValueError(f"IntegerLog requires positive value, got {a}")
    if base <= 1:
        raise ValueError(f"IntegerLog requires base > 1, got {base}")

    # Compute integer log using bit length for efficiency
    result = 0
    power = base
    while power <= a:
        power *= base
        result += 1
    return result


def integer_min(a: int, b: int) -> int:
    """Minimum of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        min(a, b)
    """
    return min(a, b)


def integer_max(a: int, b: int) -> int:
    """Maximum of two integers.

    Args:
        a: First integer
        b: Second integer

    Returns:
        max(a, b)
    """
    return max(a, b)


def integer_to_float(a: int) -> float:
    """Convert integer to float.

    Args:
        a: Integer

    Returns:
        Float representation of a
    """
    return float(a)


def integer_to_string(a: int) -> str:
    """Convert integer to string.

    Args:
        a: Integer

    Returns:
        String representation of a
    """
    return str(a)


# Register all integer builtins
register_builtin("IntegerAdd", integer_add)
register_builtin("IntegerSubtract", integer_subtract)
register_builtin("IntegerMultiply", integer_multiply)
register_builtin("IntegerDivide", integer_divide)
register_builtin("IntegerRemainder", integer_modulo)  # Renamed from IntegerModulo
register_builtin("IntegerPow", integer_power)
register_builtin("IntegerNegate", integer_negate)
register_builtin("IntegerAbs", integer_abs)
register_builtin("IntegerSign", integer_sign)
register_builtin("IntegerLog", integer_log)
register_builtin("IntegerToFloat", integer_to_float)

# Note: Removed IntegerMin/Max/ToString as they're not in the spec


__all__ = [
    "integer_add",
    "integer_subtract",
    "integer_multiply",
    "integer_divide",
    "integer_modulo",
    "integer_power",
    "integer_negate",
    "integer_abs",
    "integer_sign",
    "integer_log",
    "integer_min",
    "integer_max",
    "integer_to_float",
    "integer_to_string",
]
