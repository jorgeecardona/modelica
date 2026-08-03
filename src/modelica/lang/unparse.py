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
        case ir.Array(elements=elements):
            return f"{{{', '.join(unparse_expr(e) for e in elements)}}}"
        case ir.Call(name=name, args=args, named=named):
            written = [unparse_expr(a) for a in args]
            written += [f"{key} = {unparse_expr(value)}" for key, value in named]
            return f"{name}({', '.join(written)})"
        case ir.IfExpr():
            return _if_chain(node)


def _if_chain(node: ir.IfExpr) -> str:
    """``if a then x elseif b then y else z``, however deeply the IR nests it.

    A conditional whose else-branch is another conditional is one chain, not a tree, and
    Modelica has ``elseif`` to say so. Emitting ``else (if ...)`` instead is legal and
    parses back to the same IR, but a staged schedule written that way -- nine gas levels
    on a roaster, say -- comes out as nine opening parentheses and a wall of closing ones,
    which nobody can read and nobody would have written by hand.

    Flattening here rather than at the call sites means every front-end gets it: a chain
    built by folding a table in Python prints exactly like the ``.mo`` a person would type.
    """
    parts: list[str] = []
    branch = node
    while True:
        keyword = "if" if not parts else "elseif"
        parts.append(
            f"{keyword} {unparse_expr(branch.condition, minimum=IF + 1)}"
            f" then {unparse_expr(branch.then, minimum=IF + 1)}"
        )
        if not isinstance(branch.otherwise, ir.IfExpr):
            parts.append(f"else {unparse_expr(branch.otherwise, minimum=IF + 1)}")
            return " ".join(parts)
        branch = branch.otherwise


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


def _annotation(mod: ir.Modification) -> str:
    """`` annotation(Icon(...))``, or nothing at all when there is none to write."""
    return f" annotation{_modification(mod)}" if mod else ""


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
        case ir.Connect(a=a, b=b, annotation=annotation):
            return [f"{pad}connect({a.name}, {b.name}){_annotation(annotation)};"]
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
    annotation = _annotation(component.annotation)
    return f"{prefix}{component.type_name} {component.name}{modification}{comment}{annotation};"


def _is_short_definition(node: ir.Class) -> bool:
    """``type Voltage = Real(unit = "V");`` -- one extends and nothing else."""
    return (
        node.restriction == "type"
        and len(node.extends) == 1
        and not node.components
        and not node.equations
        and not node.classes
    )


def _members(node: ir.Class, indent: int) -> list[str]:
    """The declarations: nested classes, then ``extends``, then components."""
    pad = "  " * indent
    lines: list[str] = []
    for nested in node.classes:
        lines.append("")
        lines += unparse_class(nested, indent + 1)
    if node.classes:
        lines.append("")
    lines += [
        f"{pad}  extends {base.type_name}{_modification(base.modification)};"
        for base in node.extends
    ]
    lines += [f"{pad}  {_component(component)}" for component in node.components]
    return lines


def _sections(node: ir.Class, indent: int) -> list[str]:
    """The equation sections, initial first, each with its keyword line."""
    pad = "  " * indent
    lines: list[str] = []
    for keyword, equations in (
        ("initial equation", node.initial_equations),
        ("equation", node.equations),
    ):
        if not equations:
            continue
        lines.append(f"{pad}{keyword}")
        for equation in equations:
            lines += unparse_equation(equation, indent + 1)
    return lines


def unparse_class(node: ir.Class, indent: int = 0) -> list[str]:
    """Render a class definition as a list of lines."""
    pad = "  " * indent
    if _is_short_definition(node):
        base = node.extends[0]
        modification = _modification(base.modification)
        comment = _comment(node.comment)
        annotation = _annotation(node.annotation)
        return [f"{pad}type {node.name} = {base.type_name}{modification}{comment}{annotation};"]

    header = f"{pad}{'partial ' if node.partial else ''}{node.restriction} {node.name}"
    lines = [f"{header}{_comment(node.comment)}"]
    lines += _members(node, indent)
    lines += _sections(node, indent)

    # The class's own annotation goes last, where a tool writes it: after the equations
    # and before `end`. That is also where the parser looks for it.
    if node.annotation:
        lines.append(f"{pad}  annotation{_modification(node.annotation)};")

    lines.append(f"{pad}end {node.name};")
    return lines


def unparse(node: ir.Program | ir.Class) -> str:
    """Render a program or a single class as Modelica source, newline-terminated."""
    classes = node.classes if isinstance(node, ir.Program) else (node,)
    blocks = ["\n".join(unparse_class(c)) for c in classes]
    return "\n\n".join(blocks) + "\n"
