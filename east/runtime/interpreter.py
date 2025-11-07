"""East interpreter - tree-walking evaluation of IR.

The interpreter evaluates IR nodes to produce values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from east.runtime.environment import Environment, EnvironmentError
from east.types.primitives import null

if TYPE_CHECKING:
    from east.ir.nodes import IR, Location


class EastError(Exception):
    """Runtime error in East program.

    Attributes:
        message: Error message
        location: Source location
        stack: Call stack trace
    """

    def __init__(self, message: str, location: Location | None = None):
        """Initialize error.

        Args:
            message: Error message
            location: Source location
        """
        super().__init__(message)
        self.message = message
        self.location = location
        self.stack: list[Location] = []

    def add_stack_frame(self, location: Location) -> None:
        """Add a stack frame to the trace.

        Args:
            location: Source location
        """
        self.stack.append(location)

    def __str__(self) -> str:
        """Return error message with stack trace."""
        parts = [f"EastError: {self.message}"]
        if self.location:
            parts.append(f"  at {self.location}")
        for loc in reversed(self.stack):
            parts.append(f"  at {loc}")
        return "\n".join(parts)


class BreakException(Exception):  # noqa: N818
    """Break statement control flow.

    Attributes:
        label: Optional loop label
    """

    def __init__(self, label: str | None = None):
        """Initialize break.

        Args:
            label: Optional loop label
        """
        super().__init__()
        self.label = label


class ContinueException(Exception):  # noqa: N818
    """Continue statement control flow.

    Attributes:
        label: Optional loop label
    """

    def __init__(self, label: str | None = None):
        """Initialize continue.

        Args:
            label: Optional loop label
        """
        super().__init__()
        self.label = label


class ReturnException(Exception):  # noqa: N818
    """Return statement control flow.

    Attributes:
        value: Return value
    """

    def __init__(self, value: Any):
        """Initialize return.

        Args:
            value: Return value
        """
        super().__init__()
        self.value = value


class Interpreter:
    """Tree-walking interpreter for East IR.

    Evaluates IR nodes to produce values.
    """

    def __init__(self, platform: Any = None):
        """Initialize interpreter.

        Args:
            platform: Optional platform for platform function calls
        """
        self.platform = platform
        self.global_env = Environment()

    def eval(self, node: IR, env: Environment | None = None) -> Any:
        """Evaluate an IR node.

        Args:
            node: IR node to evaluate
            env: Environment (uses global if None)

        Returns:
            Evaluated value

        Raises:
            EastError: If evaluation fails
        """
        if env is None:
            env = self.global_env

        # Dispatch based on node type
        from east.ir.nodes import (
            Assign,
            Block,
            Break,
            Builtin,
            Call,
            Continue,
            Error,
            ForArray,
            ForDict,
            ForSet,
            Function,
            GetField,
            IfElse,
            Let,
            Match,
            NewArray,
            NewDict,
            NewSet,
            Platform,
            Return,
            StructNode,
            TryCatch,
            Value,
            Variable,
            VariantNode,
            While,
        )

        try:
            if isinstance(node, Value):
                return node.value

            if isinstance(node, Variable):
                return self.eval_variable(node, env)

            if isinstance(node, Block):
                return self.eval_block(node, env)

            if isinstance(node, IfElse):
                return self.eval_if_else(node, env)

            if isinstance(node, While):
                return self.eval_while(node, env)

            if isinstance(node, Break):
                raise BreakException(node.label)

            if isinstance(node, Continue):
                raise ContinueException(node.label)

            if isinstance(node, Return):
                value = self.eval(node.value, env)
                raise ReturnException(value)

            if isinstance(node, Let):
                return self.eval_let(node, env)

            if isinstance(node, Assign):
                return self.eval_assign(node, env)

            if isinstance(node, NewArray):
                return self.eval_new_array(node, env)

            if isinstance(node, NewSet):
                return self.eval_new_set(node, env)

            if isinstance(node, NewDict):
                return self.eval_new_dict(node, env)

            if isinstance(node, ForArray):
                return self.eval_for_array(node, env)

            if isinstance(node, ForSet):
                return self.eval_for_set(node, env)

            if isinstance(node, ForDict):
                return self.eval_for_dict(node, env)

            if isinstance(node, StructNode):
                return self.eval_struct(node, env)

            if isinstance(node, GetField):
                return self.eval_get_field(node, env)

            if isinstance(node, VariantNode):
                return self.eval_variant(node, env)

            if isinstance(node, Match):
                return self.eval_match(node, env)

            if isinstance(node, Function):
                return self.eval_function(node, env)

            if isinstance(node, Call):
                return self.eval_call(node, env)

            if isinstance(node, Platform):
                return self.eval_platform(node, env)

            if isinstance(node, Builtin):
                return self.eval_builtin(node, env)

            if isinstance(node, Error):
                return self.eval_error(node, env)

            if isinstance(node, TryCatch):
                return self.eval_try_catch(node, env)

            raise EastError(f"Unknown IR node type: {type(node).__name__}")

        except EastError:
            raise
        except EnvironmentError as e:
            raise EastError(str(e), node.location) from e

    def eval_variable(self, node: Any, env: Environment) -> Any:
        """Evaluate variable reference."""
        return env.lookup(node.name)

    def eval_block(self, node: Any, env: Environment) -> Any:
        """Evaluate block of statements."""
        result = null
        for stmt in node.statements:
            result = self.eval(stmt, env)
        return result

    def eval_if_else(self, node: Any, env: Environment) -> Any:
        """Evaluate if-else conditional."""
        condition = self.eval(node.condition, env)
        if condition:
            return self.eval(node.then_branch, env)
        return self.eval(node.else_branch, env)

    def eval_while(self, node: Any, env: Environment) -> Any:
        """Evaluate while loop."""
        result = null
        while True:
            condition = self.eval(node.condition, env)
            if not condition:
                break

            try:
                result = self.eval(node.body, env)
            except BreakException as e:
                if e.label is None or e.label == node.label:
                    break
                raise
            except ContinueException as e:
                if e.label is None or e.label == node.label:
                    continue
                raise

        return result

    def eval_let(self, node: Any, env: Environment) -> Any:
        """Evaluate variable declaration."""
        value = self.eval(node.value, env)
        env.bind(node.name, value, node.mutable)
        return null

    def eval_assign(self, node: Any, env: Environment) -> Any:
        """Evaluate variable assignment."""
        value = self.eval(node.value, env)
        env.assign(node.name, value)
        return null

    def eval_new_array(self, node: Any, env: Environment) -> Any:
        """Evaluate array construction."""
        from east.types.containers import EastArray

        elements = [self.eval(elem, env) for elem in node.elements]
        return EastArray(node.element_type, elements)

    def eval_new_set(self, node: Any, env: Environment) -> Any:
        """Evaluate set construction."""
        from east.types.containers import EastSet

        elements = [self.eval(elem, env) for elem in node.elements]
        return EastSet(node.element_type, elements)

    def eval_new_dict(self, node: Any, env: Environment) -> Any:
        """Evaluate dict construction."""
        from east.types.containers import EastDict

        entries = {}
        for key_node, val_node in node.entries:
            key = self.eval(key_node, env)
            val = self.eval(val_node, env)
            entries[key] = val

        return EastDict(node.key_type, node.value_type, entries)

    def eval_for_array(self, node: Any, env: Environment) -> Any:
        """Evaluate for loop over array."""
        array = self.eval(node.array, env)
        result = null

        for index, element in enumerate(array):
            loop_env = env.extend()
            loop_env.bind(node.index_var, index, False)
            loop_env.bind(node.element_var, element, False)

            try:
                result = self.eval(node.body, loop_env)
            except BreakException as e:
                if e.label is None or e.label == node.label:
                    break
                raise
            except ContinueException as e:
                if e.label is None or e.label == node.label:
                    continue
                raise

        return result

    def eval_for_set(self, node: Any, env: Environment) -> Any:
        """Evaluate for loop over set."""
        set_val = self.eval(node.set_expr, env)
        result = null

        for element in set_val:
            loop_env = env.extend()
            loop_env.bind(node.element_var, element, False)

            try:
                result = self.eval(node.body, loop_env)
            except BreakException as e:
                if e.label is None or e.label == node.label:
                    break
                raise
            except ContinueException as e:
                if e.label is None or e.label == node.label:
                    continue
                raise

        return result

    def eval_for_dict(self, node: Any, env: Environment) -> Any:
        """Evaluate for loop over dict."""
        dict_val = self.eval(node.dict_expr, env)
        result = null

        for key, value in dict_val.items():
            loop_env = env.extend()
            loop_env.bind(node.key_var, key, False)
            loop_env.bind(node.value_var, value, False)

            try:
                result = self.eval(node.body, loop_env)
            except BreakException as e:
                if e.label is None or e.label == node.label:
                    break
                raise
            except ContinueException as e:
                if e.label is None or e.label == node.label:
                    continue
                raise

        return result

    def eval_struct(self, node: Any, env: Environment) -> Any:
        """Evaluate struct construction."""
        from east.types.type_system import StructType

        # Evaluate field values
        field_values = {}
        for field_name, field_node in node.fields:
            field_values[field_name] = self.eval(field_node, env)

        # Build runtime StructType and create instance
        field_specs = node.struct_type.value
        fields = [(field.name, field.type) for field in field_specs]
        runtime_type = StructType(tuple(fields))

        return runtime_type.create(**field_values)

    def eval_get_field(self, node: Any, env: Environment) -> Any:
        """Evaluate struct field access."""
        struct = self.eval(node.struct, env)
        return getattr(struct, node.field_name)

    def eval_variant(self, node: Any, env: Environment) -> Any:
        """Evaluate variant construction."""
        from east.types.type_system import VariantType

        value = self.eval(node.value, env)

        # Build runtime VariantType and create instance
        case_specs = node.variant_type.value
        cases = [(case.name, case.type) for case in case_specs]
        runtime_type = VariantType(tuple(cases))

        return runtime_type.create(node.tag, value)

    def eval_match(self, node: Any, env: Environment) -> Any:
        """Evaluate pattern matching."""
        variant = self.eval(node.value, env)

        for case in node.cases:
            if variant.tag == case.tag:
                case_env = env.extend()
                case_env.bind(case.var_name, variant.value, False)
                return self.eval(case.body, case_env)

        raise EastError(f"No match case for variant tag: {variant.tag}", node.location)

    def eval_function(self, node: Any, env: Environment) -> Any:
        """Evaluate function definition (creates closure)."""
        # Capture current environment
        return {"type": "closure", "node": node, "env": env}

    def eval_call(self, node: Any, env: Environment) -> Any:
        """Evaluate function call."""
        func = self.eval(node.function, env)

        if not isinstance(func, dict) or func.get("type") != "closure":
            raise EastError("Cannot call non-function value", node.location)

        # Evaluate arguments
        args = [self.eval(arg, env) for arg in node.arguments]

        # Create new environment from closure environment
        func_node = func["node"]
        func_env = func["env"].extend()

        # Bind parameters
        for param_name, arg_value in zip(func_node.param_names, args, strict=False):
            func_env.bind(param_name, arg_value, False)

        # Execute function body
        try:
            return self.eval(func_node.body, func_env)
        except ReturnException as e:
            return e.value

    def eval_platform(self, node: Any, _env: Environment) -> NoReturn:
        """Evaluate platform function call."""
        raise EastError("Platform functions not yet implemented", node.location)

    def eval_builtin(self, node: Any, _env: Environment) -> NoReturn:
        """Evaluate builtin function call."""
        raise EastError("Builtin functions not yet implemented", node.location)

    def eval_error(self, node: Any, env: Environment) -> NoReturn:
        """Evaluate error throw."""
        message = self.eval(node.message, env)
        raise EastError(str(message), node.location)

    def eval_try_catch(self, node: Any, env: Environment) -> Any:
        """Evaluate try-catch error handling."""
        try:
            return self.eval(node.try_body, env)
        except EastError as e:
            catch_env = env.extend()
            catch_env.bind(node.error_var, e.message, False)
            return self.eval(node.catch_body, catch_env)


__all__: list[str] = [
    "Interpreter",
    "EastError",
    "BreakException",
    "ContinueException",
    "ReturnException",
]
