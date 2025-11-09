"""Float builtin functions."""

import math

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
    """Float modulo.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        a % b
    """
    return a % b


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


# Register all float builtins
register_builtin("FloatAdd", float_add)
register_builtin("FloatSubtract", float_subtract)
register_builtin("FloatMultiply", float_multiply)
register_builtin("FloatDivide", float_divide)
register_builtin("FloatRemainder", float_modulo)  # Renamed from FloatModulo
register_builtin("FloatPow", float_power)
register_builtin("FloatNegate", float_negate)
register_builtin("FloatAbs", float_abs)
register_builtin("FloatSign", float_sign)
register_builtin("FloatSqrt", float_sqrt)
register_builtin("FloatLog", float_log)
register_builtin("FloatExp", float_exp)
register_builtin("FloatSin", float_sin)
register_builtin("FloatCos", float_cos)
register_builtin("FloatTan", float_tan)
register_builtin("FloatToInteger", float_to_integer)


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
