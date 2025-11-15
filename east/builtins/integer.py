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
    """Integer remainder (JavaScript-style).

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Remainder with same sign as dividend (matches JavaScript % operator)

    Note:
        Python's % uses floored division (result has sign of divisor),
        but JavaScript's % uses truncated division (result has sign of dividend).

    Raises:
        ZeroDivisionError: If b is zero
    """
    # Get Python's floored division result
    r = a % b
    # If signs differ and there's a remainder, adjust to truncated division semantics
    if r != 0 and (a < 0) != (b < 0):
        r = r - b
    return r


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
        a: Value
        base: Logarithm base

    Returns:
        floor(log_base(abs(a))), or 0 if a == 0 or base <= 1

    Note:
        Matches JavaScript behavior: returns 0 for edge cases instead of throwing errors.
        Uses absolute value of input.
    """
    # Match TypeScript behavior: return 0 for edge cases
    if a == 0:
        return 0
    if base <= 1:
        return 0

    # Use absolute value (matches TypeScript)
    abs_value = abs(a)

    # Count how many times we can divide by base
    result = 0
    while abs_value >= base:
        abs_value = abs_value // base
        result += 1
    return result


def integer_to_float(a: int) -> float:
    """Convert integer to float.

    Args:
        a: Integer

    Returns:
        Float representation of a
    """
    return float(a)


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
    "integer_to_float",
]
