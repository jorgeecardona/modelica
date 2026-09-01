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

__all__ = [
    "unparse",
    "unparse_class",
    "unparse_equation",
    "unparse_expr",
    "unparse_statement",
]


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


def _render_expr(node: ir.Expr) -> str:  # noqa: C901, PLR0911, PLR0912 -- one per node type
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
        case ir.Matrix(rows=rows):
            body = "; ".join(", ".join(unparse_expr(e) for e in row) for row in rows)
            return f"[{body}]"
        case ir.Colon():
            return ":"
        case ir.Subscript(base=base, indices=indices):
            return f"{base.name}[{', '.join(unparse_expr(i) for i in indices)}]"
        case ir.Range(start=start, stop=stop, step=step):
            # The step is the middle term, which is the one thing about `a:s:b` that
            # everybody gets wrong once.
            parts = [start, stop] if step is None else [start, step, stop]
            return ":".join(unparse_expr(p, minimum=IF + 1) for p in parts)
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


def _entry(name: str, mod: ir.Modification) -> str:
    """One item inside a class-modification: a redeclaration, or a name and what it says."""
    if mod.redeclare is None:
        return f"{name}{_modification(mod)}"
    if isinstance(mod.redeclare, ir.Class):
        # A redeclared class is a short definition, and its own printer ends it with a
        # semicolon that has no place inside a parenthesised list.
        return f"redeclare {unparse_class(mod.redeclare)[0].rstrip(';')}"
    return f"redeclare {_component(mod.redeclare).rstrip(';')}"


def _modification(mod: ir.Modification) -> str:
    """Render a modification the way it attaches to a name: ``(a = 1, b(c = 2)) = 3``."""
    parts: list[str] = []
    if mod.entries:
        inner = ", ".join(_entry(name, m) for name, m in mod.entries.items())
        parts.append(f"({inner})")
    if mod.value is not None:
        parts.append(f" = {unparse_expr(mod.value)}")
    return "".join(parts)


def _comment(text: str) -> str:
    return f" {_string(text)}" if text else ""


def _annotation(mod: ir.Modification) -> str:
    """`` annotation(Icon(...))``, or nothing at all when there is none to write."""
    return f" annotation{_modification(mod)}" if mod else ""


def unparse_statement(  # noqa: C901, PLR0911, PLR0912 -- one arm per statement form
    node: ir.Statement,
    indent: int = 0,
    *,
    assign: str = ":=",
) -> list[str]:
    """Render one statement as lines, indented by ``indent`` levels.

    ``assign`` is the spelling of an assignment, and it is a parameter because the same
    :class:`~modelica.ir.Assign` means two different pieces of syntax depending on where it
    stands. Inside an ``algorithm`` section it is ``:=``. Inside a ``when`` in an *equation*
    section it is ``=``, because there it is an equation that happens to have a designated
    left-hand side. Printing ``:=`` in the second place produces a file no Modelica tool
    will read, which is the sort of thing a round-trip test catches and a reader does not.
    """
    pad = "  " * indent
    match node:
        case ir.Assign(target=target, value=value):
            return [f"{pad}{target.name} {assign} {unparse_expr(value)};"]
        case ir.Reinit(target=target, value=value):
            return [f"{pad}reinit({target.name}, {unparse_expr(value)});"]
        case ir.Assert(condition=condition, message=message, level=level):
            arguments = [unparse_expr(condition), unparse_expr(message)]
            if level is not None:
                arguments.append(unparse_expr(level))
            return [f"{pad}assert({', '.join(arguments)});"]
        case ir.CallStatement(targets=targets, call=call):
            named = ", ".join("" if t is None else t.name for t in targets)
            return [f"{pad}({named}) := {unparse_expr(call)};"]
        case ir.When(branches=branches):
            lines: list[str] = []
            for position, branch in enumerate(branches):
                keyword = "when" if position == 0 else "elsewhen"
                lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
                for inner in branch.body:
                    lines += unparse_statement(inner, indent + 1, assign=assign)
            lines.append(f"{pad}end when;")
            return lines
        case ir.Break():
            return [f"{pad}break;"]
        case ir.Return():
            return [f"{pad}return;"]
        case ir.ForStatement(index=index, over=over, body=body):
            return [
                f"{pad}for {index} in {unparse_expr(over)} loop",
                *_statement_body(body, indent + 1),
                f"{pad}end for;",
            ]
        case ir.WhileStatement(condition=condition, body=body):
            return [
                f"{pad}while {unparse_expr(condition)} loop",
                *_statement_body(body, indent + 1),
                f"{pad}end while;",
            ]
        case ir.IfStatement():
            return _if_statement(node, indent)


def _statement_body(body: tuple[ir.Statement, ...], indent: int) -> list[str]:
    lines: list[str] = []
    for statement in body:
        lines += unparse_statement(statement, indent)
    return lines


def _if_statement(node: ir.IfStatement, indent: int) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    for position, branch in enumerate(node.branches):
        keyword = "if" if position == 0 else "elseif"
        lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
        lines += _statement_body(branch.body, indent + 1)
    if node.otherwise:
        lines.append(f"{pad}else")
        lines += _statement_body(node.otherwise, indent + 1)
    lines.append(f"{pad}end if;")
    return lines


def unparse_equation(  # noqa: C901, PLR0911 -- one arm and one return per equation form
    node: ir.Equation,
    indent: int = 0,
) -> list[str]:
    """Render one equation as a list of lines, indented by ``indent`` levels."""
    pad = "  " * indent
    match node:
        case ir.Equal(lhs=lhs, rhs=rhs):
            return [f"{pad}{unparse_expr(lhs)} = {unparse_expr(rhs)};"]
        case ir.CallEquation(targets=(), call=call):
            return [f"{pad}{unparse_expr(call)};"]
        case ir.CallEquation(targets=targets, call=call):
            named = ", ".join("" if t is None else t.name for t in targets)
            return [f"{pad}({named}) = {unparse_expr(call)};"]
        case ir.Assert(condition=condition, message=message, level=level):
            arguments = [unparse_expr(condition), unparse_expr(message)]
            if level is not None:
                arguments.append(unparse_expr(level))
            return [f"{pad}assert({', '.join(arguments)});"]
        case ir.Connect(a=a, b=b, annotation=annotation):
            ends = f"{unparse_expr(a)}, {unparse_expr(b)}"
            return [f"{pad}connect({ends}){_annotation(annotation)};"]
        case ir.When(branches=branches):
            lines: list[str] = []
            for position, branch in enumerate(branches):
                keyword = "when" if position == 0 else "elsewhen"
                lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
                for statement in branch.body:
                    lines += unparse_statement(statement, indent + 1, assign="=")
            lines.append(f"{pad}end when;")
            return lines
        case ir.ForEquation(index=index, over=over, body=body):
            return [
                f"{pad}for {index} in {unparse_expr(over)} loop",
                *_body(body, indent + 1),
                f"{pad}end for;",
            ]
        case ir.IfEquation():
            return _if_equation(node, indent)


def _body(equations: tuple[ir.Equation, ...], indent: int) -> list[str]:
    """A run of equations, each rendered at ``indent``."""
    lines: list[str] = []
    for equation in equations:
        lines += unparse_equation(equation, indent)
    return lines


def _if_equation(node: ir.IfEquation, indent: int) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    for position, branch in enumerate(node.branches):
        keyword = "if" if position == 0 else "elseif"
        lines.append(f"{pad}{keyword} {unparse_expr(branch.condition)} then")
        lines += _body(branch.body, indent + 1)
    if node.otherwise:
        lines.append(f"{pad}else")
        lines += _body(node.otherwise, indent + 1)
    lines.append(f"{pad}end if;")
    return lines


def _component(component: ir.Component) -> str:
    prefix = ""
    if component.variability in ("parameter", "constant"):
        prefix += f"{component.variability} "
    elif component.variability == "discrete":
        prefix += "discrete "
    if component.scoping != "none":
        prefix += f"{component.scoping} "
    if component.connector != "none":
        prefix += f"{component.connector} "
    if component.causality != "none":
        prefix += f"{component.causality} "
    shape = (
        f"[{', '.join(unparse_expr(d) for d in component.dimensions)}]"
        if component.dimensions
        else ""
    )
    modification = _modification(component.modification)
    condition = f" if {unparse_expr(component.condition)}" if component.condition else ""
    constraining = f" constrainedby {component.constraining}" if component.constraining else ""
    comment = _comment(component.comment)
    annotation = _annotation(component.annotation)
    head = "replaceable " if component.replaceable else ""
    return (
        f"{head}{prefix}{component.type_name} {component.name}{shape}"
        f"{modification}{condition}{constraining}{comment}{annotation};"
    )


def _is_short_definition(node: ir.Class) -> bool:
    """``type Voltage = Real(unit = "V");`` -- one extends and nothing else.

    A `replaceable` class qualifies whatever its restriction, because that is the form it
    is written in: ``replaceable package Medium = PartialMedium constrainedby ...`` is a
    short definition with a package's keyword, and printing it long would turn its default
    into an ``extends`` that means something else.
    """
    return (
        (node.restriction == "type" or node.replaceable)
        and len(node.extends) == 1
        and not node.components
        and not node.equations
        and not node.algorithms
        and not node.initial_algorithms
        and not node.classes
        and not node.imports
        and not node.redeclarations
    )


def _import(node: ir.Import) -> str:
    if node.wildcard:
        return f"import {node.path}.*;"
    if node.alias:
        return f"import {node.alias} = {node.path};"
    return f"import {node.path};"


def _members(node: ir.Class, indent: int) -> list[str]:
    """The declarations: imports, nested classes, then ``extends``, then components.

    Imports come first because they bind the names everything after them uses, which is
    both the convention and the only order that reads.
    """
    pad = "  " * indent
    lines: list[str] = [f"{pad}  {_import(i)}" for i in node.imports]
    for nested in node.classes:
        lines.append("")
        lines += unparse_class(nested, indent + 1)
    if node.classes:
        lines.append("")
    lines += [
        f"{pad}  extends {base.type_name}{_modification(base.modification)};"
        for base in node.extends
    ]
    # After the `extends` that brought them in, which is the only order that reads.
    lines += [f"{pad}  redeclare {_component(c)}" for c in node.redeclarations]
    lines += [f"{pad}  {_component(component)}" for component in node.components]
    return lines


def _sections(node: ir.Class, indent: int) -> list[str]:
    """The equation sections, initial first, then the algorithm sections in order.

    A class may interleave them in the source; the IR keeps the equations as one list and
    the algorithm sections as several, so this order is what a round trip converges to
    rather than what was necessarily typed. Nothing downstream depends on the interleaving
    -- an algorithm section is a unit and the equations are a set.
    """
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
    for section in node.initial_algorithms:
        lines.append(f"{pad}initial algorithm")
        for statement in section:
            lines += unparse_statement(statement, indent + 1)
    for section in node.algorithms:
        lines.append(f"{pad}algorithm")
        for statement in section:
            lines += unparse_statement(statement, indent + 1)
    return lines


def unparse_class(node: ir.Class, indent: int = 0) -> list[str]:
    """Render a class definition as a list of lines."""
    pad = "  " * indent
    head = "replaceable " if node.replaceable else ""
    constraining = f" constrainedby {node.constraining}" if node.constraining else ""
    if node.literals:
        body = f"enumeration({', '.join(node.literals)})"
        tail = f"{constraining}{_comment(node.comment)}{_annotation(node.annotation)}"
        return [f"{pad}{head}type {node.name} = {body}{tail};"]
    if _is_short_definition(node):
        base = node.extends[0]
        keyword = "type" if not node.replaceable else node.restriction
        if node.expandable:
            keyword = f"expandable {node.restriction}"
        modification = _modification(base.modification)
        comment = _comment(node.comment)
        annotation = _annotation(node.annotation)
        body = f"{modification}{constraining}{comment}{annotation}"
        return [f"{pad}{head}{keyword} {node.name} = {base.type_name}{body};"]

    partial = "partial " if node.partial else ""
    expandable = "expandable " if node.expandable else ""
    header = f"{pad}{head}{partial}{expandable}{node.restriction} {node.name}"
    lines = [f"{header}{_comment(node.comment)}"]
    lines += _members(node, indent)
    lines += _sections(node, indent)

    # The class's own annotation goes last, where a tool writes it: after the equations
    # and before `end`. That is also where the parser looks for it.
    if node.annotation:
        lines.append(f"{pad}  annotation{_modification(node.annotation)};")

    lines.append(f"{pad}end {node.name}{constraining};")
    return lines


def unparse(node: ir.Program | ir.Class) -> str:
    """Render a program or a single class as Modelica source, newline-terminated."""
    classes = node.classes if isinstance(node, ir.Program) else (node,)
    blocks = ["\n".join(unparse_class(c)) for c in classes]
    return "\n\n".join(blocks) + "\n"
