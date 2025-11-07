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


def float_min(a: float, b: float) -> float:
    """Minimum of two floats.

    Args:
        a: First float
        b: Second float

    Returns:
        min(a, b)
    """
    return min(a, b)


def float_max(a: float, b: float) -> float:
    """Maximum of two floats.

    Args:
        a: First float
        b: Second float

    Returns:
        max(a, b)
    """
    return max(a, b)


def float_floor(a: float) -> float:
    """Floor of a float.

    Args:
        a: Float

    Returns:
        floor(a) as float
    """
    return math.floor(a)


def float_ceil(a: float) -> float:
    """Ceiling of a float.

    Args:
        a: Float

    Returns:
        ceil(a) as float
    """
    return math.ceil(a)


def float_round(a: float) -> float:
    """Round a float to nearest integer.

    Args:
        a: Float

    Returns:
        round(a) as float
    """
    return round(a)


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


def float_asin(a: float) -> float:
    """Arcsine of a float.

    Args:
        a: Value

    Returns:
        asin(a) in radians
    """
    return math.asin(a)


def float_acos(a: float) -> float:
    """Arccosine of a float.

    Args:
        a: Value

    Returns:
        acos(a) in radians
    """
    return math.acos(a)


def float_atan(a: float) -> float:
    """Arctangent of a float.

    Args:
        a: Value

    Returns:
        atan(a) in radians
    """
    return math.atan(a)


def float_atan2(y: float, x: float) -> float:
    """Two-argument arctangent.

    Args:
        y: Y coordinate
        x: X coordinate

    Returns:
        atan2(y, x) in radians
    """
    return math.atan2(y, x)


def float_to_integer(a: float) -> int:
    """Convert float to integer (truncate).

    Args:
        a: Float

    Returns:
        Integer truncation of a
    """
    return int(a)


def float_to_string(a: float) -> str:
    """Convert float to string.

    Args:
        a: Float

    Returns:
        String representation of a
    """
    if math.isnan(a):
        return "NaN"
    if math.isinf(a):
        return "Infinity" if a > 0 else "-Infinity"
    return str(a)


def float_is_nan(a: float) -> bool:
    """Check if float is NaN.

    Args:
        a: Float

    Returns:
        True if a is NaN
    """
    return math.isnan(a)


def float_is_infinite(a: float) -> bool:
    """Check if float is infinite.

    Args:
        a: Float

    Returns:
        True if a is infinite
    """
    return math.isinf(a)


def float_is_finite(a: float) -> bool:
    """Check if float is finite.

    Args:
        a: Float

    Returns:
        True if a is finite (not NaN or infinite)
    """
    return math.isfinite(a)


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
register_builtin("FloatFloor", float_floor)
register_builtin("FloatCeil", float_ceil)
register_builtin("FloatRound", float_round)
register_builtin("FloatLog", float_log)
register_builtin("FloatExp", float_exp)
register_builtin("FloatSin", float_sin)
register_builtin("FloatCos", float_cos)
register_builtin("FloatTan", float_tan)
register_builtin("FloatAsin", float_asin)
register_builtin("FloatAcos", float_acos)
register_builtin("FloatAtan", float_atan)
register_builtin("FloatAtan2", float_atan2)
register_builtin("FloatToInteger", float_to_integer)
register_builtin("FloatToString", float_to_string)
register_builtin("FloatIsNaN", float_is_nan)
register_builtin("FloatIsInfinite", float_is_infinite)
register_builtin("FloatIsFinite", float_is_finite)


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
    "float_min",
    "float_max",
    "float_floor",
    "float_ceil",
    "float_round",
    "float_sqrt",
    "float_log",
    "float_exp",
    "float_sin",
    "float_cos",
    "float_tan",
    "float_asin",
    "float_acos",
    "float_atan",
    "float_atan2",
    "float_to_integer",
    "float_to_string",
    "float_is_nan",
    "float_is_infinite",
    "float_is_finite",
]
