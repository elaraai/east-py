"""IR analysis and validation for East runtime.

This module provides analysis of IR before compilation, computing metadata
needed for efficient execution (e.g., async propagation).
"""

from dataclasses import dataclass

from east.runtime.platform import PlatformFunction
from east.types.structural import EastStruct, EastVariant
from east.types.type_system import EastType


@dataclass
class VariableMetadata:
    """Metadata about a variable tracked during IR analysis.

    Attributes:
        type: The East type of the variable
        mutable: Whether the variable can be mutated
        defined_by: The IR node that defined this variable (Let or Function parameter)
        defined_at: Location where variable was defined
        captured: Whether this variable is captured by a nested function
    """

    type: EastType
    mutable: bool
    defined_by: EastVariant
    defined_at: EastStruct  # Location struct
    captured: bool = False


# Variable context mapping variable names to their metadata
VariableContext = dict[str, VariableMetadata]


def analyze_ir(
    ir: EastVariant, platform: list[PlatformFunction], ctx: VariableContext | None = None
) -> tuple[EastVariant, dict[int, bool]]:
    """Analyze IR tree and compute is_async metadata.

    This function:
    - Validates that all platform functions referenced in IR exist
    - Validates that all variable references are defined
    - Computes which nodes require async compilation
    - Tracks variable captures in closures
    - Returns IR and is_async mapping

    Args:
        ir: The IR tree to analyze (EastVariant)
        platform: List of platform functions available
        ctx: Variable context (optional, used for nested scopes)

    Returns:
        Tuple of (IR tree, is_async_map) where is_async_map maps id(node) -> bool

    Raises:
        ValueError: If IR references unknown platform functions
        NameError: If IR references undefined variables

    Example:
        >>> platform = [
        ...     PlatformFunction(name="log", inputs=[StringType], output=NullType,
        ...                      type='sync', fn=print)
        ... ]
        >>> ir, is_async_map = analyze_ir(function_ir, platform, {})
    """
    # Build platform function lookup
    platform_map: dict[str, bool] = {}  # name -> is_async
    for pf in platform:
        platform_map[pf["name"]] = pf["type"] == "async"

    # Track which platform functions we've seen (for validation)
    referenced_platforms: set[str] = set()

    # Map from id(IR node) -> is_async
    is_async_map: dict[int, bool] = {}

    # Initialize variable context if not provided
    if ctx is None:
        ctx = {}

    def visit_ir(node: EastVariant, var_ctx: VariableContext) -> bool:
        """Visit IR node, validate, and compute is_async.

        Args:
            node: IR node to visit
            var_ctx: Variable context for this scope

        Returns:
            True if this node requires async execution, False otherwise
        """
        tag = node.tag
        is_async = False

        if tag == "Platform":
            # Validate platform function exists
            platform_name = node.value.name
            referenced_platforms.add(platform_name)

            if platform_name not in platform_map:
                available = ", ".join(sorted(platform_map.keys())) if platform_map else "(none)"
                raise ValueError(
                    f"Platform function '{platform_name}' not found. "
                    f"Available platform functions: {available}"
                )

            # Platform is async if function itself is async OR any argument is async
            is_async = platform_map[platform_name]
            for arg in node.value.arguments:
                if visit_ir(arg, var_ctx):
                    is_async = True

        elif tag == "Function":
            # Validate that function's required platforms are available
            func_type = node.value.type
            if func_type.tag == "Function":
                required_platforms = func_type.value.platforms
                if required_platforms:
                    missing_platforms = [p for p in required_platforms if p not in platform_map]
                    if missing_platforms:
                        func_loc = node.value.location
                        available = (
                            ", ".join(sorted(platform_map.keys())) if platform_map else "(none)"
                        )
                        raise ValueError(
                            f"Function at {func_loc.filename}:{func_loc.line}:{func_loc.column} "
                            f"requires platform function(s) {missing_platforms} which are not available. "
                            f"Available platforms: {available}"
                        )

            # Create child context for function body
            # Child inherits parent variables (potential captures)
            child_ctx = var_ctx.copy()

            # Add parameters to child context
            for param in node.value.parameters:
                param_var = param.value
                param_name = param_var.name
                param_type = param_var.type
                param_mutable = param_var.mutable
                param_location = param_var.location

                child_ctx[param_name] = VariableMetadata(
                    type=param_type,
                    mutable=param_mutable,
                    defined_by=node,  # Defined by this Function node
                    defined_at=param_location,
                    captured=False,
                )

            # Visit body with child context
            visit_ir(node.value.body, child_ctx)

            # Mark captured variables in parent context
            # A variable is captured if it's in parent context and referenced in child
            for var_name, var_meta in child_ctx.items():
                if var_name in var_ctx and var_meta is var_ctx[var_name]:
                    # Same object means it was inherited from parent and used in child
                    var_ctx[var_name].captured = True

            # Function node itself is sync (is_async = False)

        elif tag == "Block":
            # Block is async if any statement is async
            for stmt in node.value.statements:
                if visit_ir(stmt, var_ctx):
                    is_async = True

        elif tag == "Let":
            # Let is async if value expression is async
            # First visit the value expression
            if visit_ir(node.value.value, var_ctx):
                is_async = True

            # Add variable to context after evaluating value
            var_node = node.value.variable
            var_name = var_node.value.name
            var_type = var_node.value.type
            var_mutable = var_node.value.mutable
            var_location = var_node.value.location

            var_ctx[var_name] = VariableMetadata(
                type=var_type,
                mutable=var_mutable,
                defined_by=node,  # Defined by this Let node
                defined_at=var_location,
                captured=False,
            )

            # Visit the variable node itself (will now succeed validation)
            visit_ir(node.value.variable, var_ctx)

        elif tag == "Builtin":
            # Builtins are always sync, but arguments might be async
            for arg in node.value.arguments:
                if visit_ir(arg, var_ctx):
                    is_async = True

        elif tag == "IfElse":
            # IfElse is async if predicate, any if body, or else body is async
            for if_case in node.value.ifs:
                if visit_ir(if_case.predicate, var_ctx):
                    is_async = True
                if visit_ir(if_case.body, var_ctx):
                    is_async = True
            if visit_ir(node.value.else_body, var_ctx):
                is_async = True

        elif tag == "While":
            # While is async if predicate or body is async
            if visit_ir(node.value.predicate, var_ctx):
                is_async = True
            if visit_ir(node.value.body, var_ctx):
                is_async = True

        elif tag == "Value":
            # Value nodes are always sync
            is_async = False

        elif tag == "Variable":
            # Validate variable reference exists in context
            var_name = node.value.name
            if var_name not in var_ctx:
                var_loc = node.value.location
                raise NameError(
                    f"Variable '{var_name}' is not defined. "
                    f"Referenced at {var_loc.filename}:{var_loc.line}:{var_loc.column}"
                )
            # Variable references are always sync
            is_async = False

        elif tag == "Call":
            # Call is async if function or any argument is async
            if visit_ir(node.value.function, var_ctx):
                is_async = True
            for arg in node.value.arguments:
                if visit_ir(arg, var_ctx):
                    is_async = True

        elif tag == "Assign":
            # Assign is async if value is async
            visit_ir(node.value.variable, var_ctx)
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "Return":
            # Return is async if value is async
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "Break":
            # Break is always sync
            is_async = False

        elif tag == "Continue":
            # Continue is always sync
            is_async = False

        elif tag == "Match":
            # Match is async if variant or any case body is async
            if visit_ir(node.value.variant, var_ctx):
                is_async = True
            for case in node.value.cases:
                if visit_ir(case.body, var_ctx):
                    is_async = True

        elif tag == "Struct":
            # Struct is async if any field value is async
            for field_value in node.value.fields:
                if visit_ir(field_value, var_ctx):
                    is_async = True

        elif tag == "GetField":
            # GetField is async if struct expression is async
            if visit_ir(node.value.struct, var_ctx):
                is_async = True

        elif tag == "Variant":
            # Variant is async if value is async
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "NewArray":
            # NewArray is async if any element is async
            for elem in node.value.values:
                if visit_ir(elem, var_ctx):
                    is_async = True

        elif tag == "NewSet":
            # NewSet is async if any element is async
            for elem in node.value.values:
                if visit_ir(elem, var_ctx):
                    is_async = True

        elif tag == "NewDict":
            # NewDict is async if any key or value is async
            for entry in node.value.entries:
                if visit_ir(entry.key, var_ctx):
                    is_async = True
                if visit_ir(entry.value, var_ctx):
                    is_async = True

        elif tag == "ForArray":
            # ForArray is async if collection or body is async
            if visit_ir(node.value.array, var_ctx):
                is_async = True
            visit_ir(node.value.element_variable, var_ctx)
            if visit_ir(node.value.body, var_ctx):
                is_async = True

        elif tag == "ForSet":
            # ForSet is async if collection or body is async
            if visit_ir(node.value.set, var_ctx):
                is_async = True
            visit_ir(node.value.element_variable, var_ctx)
            if visit_ir(node.value.body, var_ctx):
                is_async = True

        elif tag == "ForDict":
            # ForDict is async if collection or body is async
            if visit_ir(node.value.dict, var_ctx):
                is_async = True
            visit_ir(node.value.key_variable, var_ctx)
            visit_ir(node.value.value_variable, var_ctx)
            if visit_ir(node.value.body, var_ctx):
                is_async = True

        elif tag == "As":
            # As is async if value is async
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "UnwrapRecursive":
            # UnwrapRecursive is async if value is async
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "WrapRecursive":
            # WrapRecursive is async if value is async
            if visit_ir(node.value.value, var_ctx):
                is_async = True

        elif tag == "Error":
            # Error is async if message is async
            if visit_ir(node.value.message, var_ctx):
                is_async = True

        elif tag == "TryCatch":
            # TryCatch is async if try_body, catch_body, or finally_body is async
            if visit_ir(node.value.try_body, var_ctx):
                is_async = True
            visit_ir(node.value.message, var_ctx)
            visit_ir(node.value.stack, var_ctx)
            if visit_ir(node.value.catch_body, var_ctx):
                is_async = True
            # Process finally block if present (not null)
            from east.types.primitives import Null

            if (
                hasattr(node.value, "finally_body")
                and node.value.finally_body is not None
                and not isinstance(node.value.finally_body, Null)
                and visit_ir(node.value.finally_body, var_ctx)
            ):
                is_async = True

        else:
            # Unknown IR node type
            raise NotImplementedError(f"Analysis for IR node type '{tag}' not implemented")

        # Store is_async for this node
        is_async_map[id(node)] = is_async
        return is_async

    # Visit the IR tree to compute is_async for all nodes
    visit_ir(ir, ctx)

    # Return IR and is_async mapping
    return ir, is_async_map


__all__ = ["analyze_ir", "VariableMetadata", "VariableContext"]
