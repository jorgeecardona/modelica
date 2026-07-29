"""IR -> Modelica source text.

The inverse of the parser, and useful long before the parser exists: it is how a model
built with :mod:`modelica.build` is shown to someone who reads Modelica, and how the test
suite checks that a Python model and a ``.mo`` file say the same thing.

Output is deterministic and normalised -- one space around binary operators, two-space
indentation, parentheses only where precedence requires them.
"""

from __future__ import annotations

from modelica import ir
from modelica.lang.precedence import BINARY, IF, NON_ASSOCIATIVE, PRIMARY, UNARY_SIGN

__all__ = ["unparse", "unparse_class", "unparse_equation", "unparse_expr"]


def _real(value: float) -> str:
    text = repr(float(value))
    return text.replace("e-0", "e-").replace("e+0", "e+") if "e" in text else text


def _string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _precedence(node: ir.Expr) -> int:
    match node:
        case ir.IfExpr():
            return IF
        case ir.Binary(op=op):
            return BINARY[op]
        case ir.Unary(op="not"):
            return BINARY["not"]
        case ir.Unary():
            return UNARY_SIGN
        case _:
            return PRIMARY


def unparse_expr(node: ir.Expr, *, minimum: int = IF) -> str:
    """Render an expression, parenthesising it if it binds looser than ``minimum``."""
    text = _render_expr(node)
    return f"({text})" if _precedence(node) < minimum else text


def _render_expr(node: ir.Expr) -> str:  # noqa: C901, PLR0911 -- one case per node type
    match node:
        case ir.Ref(parts=parts):
            return ".".join(parts)
        case ir.RealLiteral(value=value):
            return _real(value)
        case ir.IntLiteral(value=value):
            return str(value)
        case ir.BoolLiteral(value=value):
            return "true" if value else "false"
        case ir.StringLiteral(value=value):
            return _string(value)
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            level = BINARY[op]
            left = unparse_expr(lhs, minimum=level + (op in NON_ASSOCIATIVE))
            right = unparse_expr(rhs, minimum=level + 1)
            return f"{left} {op} {right}"
        case ir.Unary(op="not", operand=operand):
            return f"not {unparse_expr(operand, minimum=BINARY['not'])}"
        case ir.Unary(op=op, operand=operand):
            return f"{op}{unparse_expr(operand, minimum=UNARY_SIGN)}"
        case ir.Call(name=name, args=args):
            return f"{name}({', '.join(unparse_expr(a) for a in args)})"
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return (
                f"if {unparse_expr(condition, minimum=IF + 1)}"
                f" then {unparse_expr(then, minimum=IF + 1)}"
                f" else {unparse_expr(otherwise, minimum=IF + 1)}"
            )


def _modification(mod: ir.Modification) -> str:
    """Render a modification the way it attaches to a name: ``(a = 1, b(c = 2)) = 3``."""
    parts: list[str] = []
    if mod.entries:
        inner = ", ".join(f"{name}{_modification(m)}" for name, m in mod.entries.items())
        parts.append(f"({inner})")
    if mod.value is not None:
        parts.append(f" = {unparse_expr(mod.value)}")
    return "".join(parts)


def _comment(text: str) -> str:
    return f" {_string(text)}" if text else ""


def _statement(node: ir.Statement) -> str:
    match node:
        case ir.Assign(target=target, value=value):
            return f"{target.name} = {unparse_expr(value)};"
        case ir.Reinit(target=target, value=value):
            return f"reinit({target.name}, {unparse_expr(value)});"


def unparse_equation(node: ir.Equation, indent: int = 0) -> list[str]:
    """Render one equation as a list of lines, indented by ``indent`` levels."""
    pad = "  " * indent
    match node:
        case ir.Equal(lhs=lhs, rhs=rhs):
            return [f"{pad}{unparse_expr(lhs)} = {unparse_expr(rhs)};"]
        case ir.Connect(a=a, b=b):
            return [f"{pad}connect({a.name}, {b.name});"]
        case ir.When(branches=branches):
            lines: list[str] = []
            for position, branch in enumerate(branches):
                keyword = "when" if position == 0 else "elsewhen"
                lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
                lines += [f"{pad}  {_statement(s)}" for s in branch.body]
            lines.append(f"{pad}end when;")
            return lines
        case ir.IfEquation(branches=branches, otherwise=otherwise):
            lines = []
            for position, branch in enumerate(branches):
                keyword = "if" if position == 0 else "elseif"
                lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
                for equation in branch.body:
                    lines += unparse_equation(equation, indent + 1)
            if otherwise:
                lines.append(f"{pad}else")
                for equation in otherwise:
                    lines += unparse_equation(equation, indent + 1)
            lines.append(f"{pad}end if;")
            return lines


def _component(component: ir.Component) -> str:
    prefix = ""
    if component.variability in ("parameter", "constant"):
        prefix += f"{component.variability} "
    elif component.variability == "discrete":
        prefix += "discrete "
    if component.connector != "none":
        prefix += f"{component.connector} "
    if component.causality != "none":
        prefix += f"{component.causality} "
    modification = _modification(component.modification)
    comment = _comment(component.comment)
    return f"{prefix}{component.type_name} {component.name}{modification}{comment};"


def _is_short_definition(node: ir.Class) -> bool:
    """``type Voltage = Real(unit = "V");`` -- one extends and nothing else."""
    return (
        node.restriction == "type"
        and len(node.extends) == 1
        and not node.components
        and not node.equations
        and not node.classes
    )


def unparse_class(node: ir.Class, indent: int = 0) -> list[str]:
    """Render a class definition as a list of lines."""
    pad = "  " * indent
    if _is_short_definition(node):
        base = node.extends[0]
        modification = _modification(base.modification)
        comment = _comment(node.comment)
        return [f"{pad}type {node.name} = {base.type_name}{modification}{comment};"]

    header = f"{pad}{'partial ' if node.partial else ''}{node.restriction} {node.name}"
    lines = [f"{header}{_comment(node.comment)}"]

    for nested in node.classes:
        lines.append("")
        lines += unparse_class(nested, indent + 1)
    if node.classes:
        lines.append("")

    for base in node.extends:
        lines.append(f"{pad}  extends {base.type_name}{_modification(base.modification)};")
    for component in node.components:
        lines.append(f"{pad}  {_component(component)}")

    if node.initial_equations:
        lines.append(f"{pad}initial equation")
        for equation in node.initial_equations:
            lines += unparse_equation(equation, indent + 1)
    if node.equations:
        lines.append(f"{pad}equation")
        for equation in node.equations:
            lines += unparse_equation(equation, indent + 1)

    lines.append(f"{pad}end {node.name};")
    return lines


def unparse(node: ir.Program | ir.Class) -> str:
    """Render a program or a single class as Modelica source, newline-terminated."""
    classes = node.classes if isinstance(node, ir.Program) else (node,)
    blocks = ["\n".join(unparse_class(c)) for c in classes]
    return "\n\n".join(blocks) + "\n"
