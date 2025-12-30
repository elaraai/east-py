#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Float builtin functions."""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from east.builtins.registry import register_builtin


def float_add(a: float, b: float) -> float:
    """Add two floats.

    Args:
        a: First float
        b: Second float

    Returns:
        a + b
    """
    return a + b


def float_subtract(a: float, b: float) -> float:
    """Subtract two floats.

    Args:
        a: First float
        b: Second float

    Returns:
        a - b
    """
    return a - b


def float_multiply(a: float, b: float) -> float:
    """Multiply two floats.

    Args:
        a: First float
        b: Second float

    Returns:
        a * b
    """
    return a * b


def float_divide(a: float, b: float) -> float:
    """Divide two floats.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        a / b
    """
    return a / b


def float_modulo(a: float, b: float) -> float:
    """Float remainder (JavaScript-style).

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Remainder with same sign as dividend (matches JavaScript % operator)
        Returns NaN if divisor is zero (matches JavaScript behavior)

    Note: Python's % uses floored division (result has sign of divisor),
          but JavaScript's % uses truncated division (result has sign of dividend).
    """
    import math

    # JavaScript returns NaN for modulo by zero
    if b == 0.0:
        return float("nan")

    # JavaScript-style remainder: result = a - trunc(a/b) * b
    result = a - (math.trunc(a / b) * b)

    # Preserve signed zero: if result is exactly zero, use sign of dividend
    if result == 0.0:
        return math.copysign(0.0, a)
    return result


def float_power(a: float, b: float) -> float:
    """Float exponentiation.

    Args:
        a: Base
        b: Exponent

    Returns:
        a ** b
    """
    return a**b


def float_negate(a: float) -> float:
    """Negate a float.

    Args:
        a: Float to negate

    Returns:
        -a
    """
    return -a


def float_abs(a: float) -> float:
    """Absolute value of a float.

    Args:
        a: Float

    Returns:
        |a|
    """
    return abs(a)


def float_sign(a: float) -> float:
    """Sign of a float.

    Args:
        a: Float

    Returns:
        -1.0 if a < 0, 0.0 if a == 0 or NaN, 1.0 if a > 0
    """
    if math.isnan(a):
        return 0.0
    if a < 0:
        return -1.0
    if a > 0:
        return 1.0
    return 0.0


def float_sqrt(a: float) -> float:
    """Square root of a float.

    Args:
        a: Float

    Returns:
        sqrt(a)
    """
    return math.sqrt(a)


def float_log(a: float) -> float:
    """Natural logarithm of a float.

    Args:
        a: Float

    Returns:
        ln(a)
    """
    return math.log(a)


def float_exp(a: float) -> float:
    """Exponential of a float.

    Args:
        a: Float

    Returns:
        e^a
    """
    return math.exp(a)


def float_sin(a: float) -> float:
    """Sine of a float (in radians).

    Args:
        a: Angle in radians

    Returns:
        sin(a)
    """
    return math.sin(a)


def float_cos(a: float) -> float:
    """Cosine of a float (in radians).

    Args:
        a: Angle in radians

    Returns:
        cos(a)
    """
    return math.cos(a)


def float_tan(a: float) -> float:
    """Tangent of a float (in radians).

    Args:
        a: Angle in radians

    Returns:
        tan(a)
    """
    return math.tan(a)


def float_to_integer(a: float) -> int:
    """Convert float to integer (truncate).

    Args:
        a: Float

    Returns:
        Integer truncation of a
    """
    return int(a)


# Register all float builtins as factories (no type params, so return impl directly)
register_builtin("FloatAdd", lambda _platform: float_add)
register_builtin("FloatSubtract", lambda _platform: float_subtract)
register_builtin("FloatMultiply", lambda _platform: float_multiply)
register_builtin("FloatDivide", lambda _platform: float_divide)
register_builtin("FloatRemainder", lambda _platform: float_modulo)  # Renamed from FloatModulo
register_builtin("FloatPow", lambda _platform: float_power)
register_builtin("FloatNegate", lambda _platform: float_negate)
register_builtin("FloatAbs", lambda _platform: float_abs)
register_builtin("FloatSign", lambda _platform: float_sign)
register_builtin("FloatSqrt", lambda _platform: float_sqrt)
register_builtin("FloatLog", lambda _platform: float_log)
register_builtin("FloatExp", lambda _platform: float_exp)
register_builtin("FloatSin", lambda _platform: float_sin)
register_builtin("FloatCos", lambda _platform: float_cos)
register_builtin("FloatTan", lambda _platform: float_tan)
register_builtin("FloatToInteger", lambda _platform: float_to_integer)


__all__ = [
    "float_add",
    "float_subtract",
    "float_multiply",
    "float_divide",
    "float_modulo",
    "float_power",
    "float_negate",
    "float_abs",
    "float_sign",
    "float_sqrt",
    "float_log",
    "float_exp",
    "float_sin",
    "float_cos",
    "float_tan",
    "float_to_integer",
]
