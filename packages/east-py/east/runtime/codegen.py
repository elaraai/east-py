#
# Copyright (c) 2025 Elara AI Pty Ltd
# Licensed under the Business Source License 1.1. See LICENSE.md for details.
#
"""Code generators for specialized IR closures.

Uses exec at compile time to produce a closure that calls sub-expressions
directly by position — no list comprehension, no *-unpacking. The code
strings are constructed from integer indices only; no user input is involved.
"""

from east.runtime.errors import EastError, _wrap_exception_with_location


def gen_builtin_sync(specialized_fn, arg_fns, ir_location):
    """Builtin call — direct positional args, no list/unpack."""
    n = len(arg_fns)
    ns = {f"_f{i}": f for i, f in enumerate(arg_fns)}
    ns.update(
        _fn=specialized_fn,
        _loc=ir_location,
        _EastError=EastError,
        _wrap=_wrap_exception_with_location,
    )
    args = ", ".join(f"_f{i}(env)" for i in range(n))
    code = f"""\
def _call(env):
    try:
        return _fn({args})
    except _EastError as e:
        e.location.extend(_loc)
        raise
    except Exception as e:
        raise _wrap(e, _loc) from e
"""
    exec(code, ns)  # noqa: S102
    return ns["_call"]
