"""Boolean builtin functions."""

from east.builtins.registry import register_builtin


def boolean_and(a: bool, b: bool) -> bool:
    """Logical AND.

    Args:
        a: First boolean
        b: Second boolean

    Returns:
        a AND b
    """
    return a and b


def boolean_or(a: bool, b: bool) -> bool:
    """Logical OR.

    Args:
        a: First boolean
        b: Second boolean

    Returns:
        a OR b
    """
    return a or b


def boolean_not(a: bool) -> bool:
    """Logical NOT.

    Args:
        a: Boolean to negate

    Returns:
        NOT a
    """
    return not a


# Register all boolean builtins
register_builtin("BooleanAnd", boolean_and)
register_builtin("BooleanOr", boolean_or)
register_builtin("BooleanNot", boolean_not)


__all__ = ["boolean_and", "boolean_or", "boolean_not"]
