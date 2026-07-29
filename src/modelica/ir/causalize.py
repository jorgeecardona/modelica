"""Turning a set of equations into a computation.

A flat model says what is true. It does not say what to compute from what, and for an
acausal language that gap is the whole problem: ``r.v = r.R * r.i`` is a statement about
three numbers, not an instruction. Causalization closes the gap in four passes.

**Naming derivatives.** ``der(x)`` becomes an ordinary variable ``der_x``. After this the
model is algebraic, and every later pass gets to be ignorant of calculus.

**Alias elimination.** Connection equations generate enormous numbers of relations of the
form ``a = ±b + k`` -- three flanges bolted together are three names for one position.
Substituting them away is not an optimisation. Without it ``damper.s_rel`` and ``mass.s``
look like independent states and the model comes out one index too high.

**Index reduction** (Pantelides). Some models constrain their states: the pendulum's
``x^2 + y^2 = L^2`` says the two positions are really one. Such a system has no solution
for the highest derivatives until some equations are differentiated. Pantelides finds
which ones by asking, repeatedly, for a matching that assigns every equation a variable of
top order; where it fails it has found exactly the subset to differentiate.

**Matching and sorting.** A maximum matching of equations to unknowns decides who computes
what. The variables left over are the states -- integrate them and everything else follows.
Tarjan's algorithm on the resulting dependency graph then sorts the equations, and its
strongly connected components are precisely the algebraic loops.

Dummy derivatives need no separate pass here. A "dummy" is what the literature calls a
derivative variable that got matched as an ordinary unknown instead of being integrated,
and the matching produces them without being told to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import TYPE_CHECKING

from modelica.ir import nodes as ir
from modelica.ir.derive import (
    ZERO,
    DerivativeError,
    add,
    d_dt,
    d_dx,
    der_name,
    div,
    mul,
    neg,
    simplify,
)
from modelica.ir.flat import FlatVariable
from modelica.ir.rewrite import names, residual, solve_for, substitute
from modelica.ir.system import Block, System

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from modelica.ir.flat import FlatModel

__all__ = ["CausalizationError", "causalize"]

# How many times index reduction may differentiate before we call it a runaway. Real
# models need a handful; anything past this is a bug in the model or in this file.
_BUDGET = 500


class CausalizationError(Exception):
    """A model that cannot be arranged into a computation, and why."""


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _ref(name: str) -> ir.Ref:
    return ir.Ref(tuple(name.split(".")))


def _number(node: ir.Expr) -> float | None:
    match node:
        case ir.RealLiteral(value=value):
            return value
        case ir.IntLiteral(value=value):
            return float(value)
        case _:
            return None


def _unique(items: Iterable[str]) -> Iterator[str]:
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            yield item


def _apply_to(equation: ir.Equal, mapping: Mapping[str, ir.Expr]) -> ir.Equal:
    return ir.Equal(
        simplify(substitute(equation.lhs, mapping)), simplify(substitute(equation.rhs, mapping))
    )


# --------------------------------------------------------------------------------------
# The working state -- one mutable object threaded through the passes
# --------------------------------------------------------------------------------------


@dataclass
class _Work:
    """Everything the passes rewrite, kept in one place so they can be read side by side.

    ``derivative`` and ``integral`` are inverses, and both are needed: matching asks "does
    this variable have a derivative?" (may it be a state?) and "is this variable somebody's
    derivative?" (how high does its chain reach?), and neither question answers the other.
    """

    name: str
    equations: list[ir.Equal]
    variables: dict[str, FlatVariable] = field(default_factory=dict[str, FlatVariable])
    order: dict[str, int] = field(default_factory=dict[str, int])
    derivative: dict[str, str] = field(default_factory=dict[str, str])
    integral: dict[str, str] = field(default_factory=dict[str, str])
    aliases: dict[str, ir.Expr] = field(default_factory=dict[str, ir.Expr])
    parent: dict[int, int] = field(default_factory=dict[int, int])
    differentiated: dict[int, int] = field(default_factory=dict[int, int])
    depth: dict[int, int] = field(default_factory=dict[int, int])
    counter: int = 0

    def declare(self, variable: FlatVariable) -> None:
        self.variables[variable.name] = variable
        self.order[variable.name] = self.counter
        self.counter += 1

    def forget(self, name: str) -> None:
        self.variables.pop(name, None)
        self.order.pop(name, None)

    def is_unknown(self, name: str) -> bool:
        """A continuous variable this stage has to solve for.

        Parameters and constants are data; inputs come from outside; discrete variables are
        assigned by ``when`` clauses. All of them count as known here.
        """
        variable = self.variables.get(name)
        return (
            variable is not None
            and variable.variability == "continuous"
            and variable.causality != "input"
        )

    def constants(self) -> frozenset[str]:
        return frozenset(name for name, v in self.variables.items() if v.is_constant)

    def incidence(self, index: int) -> list[str]:
        equation = self.equations[index]
        reads = _unique([*names(equation.lhs), *names(equation.rhs)])
        return [name for name in reads if self.is_unknown(name)]

    def level(self, name: str) -> int:
        """How many times ``name`` has been differentiated: ``der_der_x`` is 2."""
        count = 0
        while name in self.integral:
            name = self.integral[name]
            count += 1
        return count


# --------------------------------------------------------------------------------------
# Pass 1 -- der(x) becomes a variable
# --------------------------------------------------------------------------------------


def _ensure_derivative(work: _Work, name: str) -> str:
    """The variable standing for ``der(name)``, declaring it the first time it is asked for.

    The answer is not always ``der_name``: alias elimination may already have discovered
    that some variable the modeller named is this derivative, and that name wins.
    """
    existing = work.derivative.get(name)
    if existing is not None:
        return existing
    base = work.variables.get(name)
    if base is None:
        raise CausalizationError(f"{work.name}: der() of undeclared {name!r}")
    if base.is_constant:
        raise CausalizationError(f"{work.name}: der() of the {base.variability} {name!r}")
    created = der_name(name)
    work.declare(
        FlatVariable(name=created, base_type=base.base_type, comment=f"derivative of {name}")
    )
    work.derivative[name] = created
    work.integral[created] = name
    return created


def _name_derivatives(node: ir.Expr, work: _Work) -> ir.Expr:
    """Rewrite every ``der(v)`` in ``node`` as a reference to the variable standing for it."""
    match node:
        case ir.Call(name="der", args=args):
            (argument,) = args
            if not isinstance(argument, ir.Ref):
                raise CausalizationError(f"{work.name}: der() needs a variable, got {argument!r}")
            return _ref(_ensure_derivative(work, argument.name))
        case ir.Binary(op=op, lhs=lhs, rhs=rhs):
            return ir.Binary(op, _name_derivatives(lhs, work), _name_derivatives(rhs, work))
        case ir.Unary(op=op, operand=operand):
            return ir.Unary(op, _name_derivatives(operand, work))
        case ir.Call(name=name, args=args):
            return ir.Call(name, tuple(_name_derivatives(a, work) for a in args))
        case ir.IfExpr(condition=condition, then=then, otherwise=otherwise):
            return ir.IfExpr(
                _name_derivatives(condition, work),
                _name_derivatives(then, work),
                _name_derivatives(otherwise, work),
            )
        case _:
            return node


def _named(equation: ir.Equal, work: _Work) -> ir.Equal:
    return ir.Equal(_name_derivatives(equation.lhs, work), _name_derivatives(equation.rhs, work))


def _start(flat: FlatModel) -> _Work:
    work = _Work(name=flat.name, equations=[])
    for variable in flat.variables:
        work.declare(variable)
    work.equations = [_named(equation, work) for equation in flat.equations]
    work.depth = dict.fromkeys(range(len(work.equations)), 0)
    return work


# --------------------------------------------------------------------------------------
# Pass 2 -- alias elimination
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Affine:
    """``sum(coefficients * unknowns) + offset = 0``, with the offset free of unknowns."""

    unknowns: tuple[str, ...]
    coefficients: tuple[ir.Expr, ...]
    offset: ir.Expr


@dataclass(frozen=True)
class _Elimination:
    victim: str
    replacement: ir.Expr
    extra: tuple[tuple[str, ir.Expr], ...] = ()
    derivative_of: tuple[str, str] | None = None

    def mapping(self) -> dict[str, ir.Expr]:
        return {self.victim: self.replacement, **dict(self.extra)}


def _affine(work: _Work, equation: ir.Equal) -> _Affine | None:
    """Read ``equation`` as an affine relation between at most two unknowns, or give up.

    The gradient with respect to the unknowns has to be free of unknowns -- that is what
    affine means, and it is why ``u1 * u2`` is rejected while ``k * u1 + u2`` is not.
    """
    expression = residual(equation)
    appearing = [name for name in _unique(names(expression)) if work.is_unknown(name)]
    if not appearing:
        return None
    unknowns: list[str] = []
    coefficients: list[ir.Expr] = []
    for name in appearing:
        try:
            coefficient = simplify(d_dx(expression, name))
        except DerivativeError:
            return None
        if any(work.is_unknown(other) for other in names(coefficient)):
            return None
        # A coefficient that folds to zero means the variable cancelled out; it is not
        # really in this equation, and pretending otherwise would reject the equation.
        if _number(coefficient) == 0:
            continue
        unknowns.append(name)
        coefficients.append(coefficient)
    if not unknowns or len(unknowns) > 2:  # noqa: PLR2004 - two is the alias case
        return None
    offset = simplify(substitute(expression, dict.fromkeys(appearing, ZERO)))
    if any(work.is_unknown(name) for name in names(offset)):
        return None
    return _Affine(tuple(unknowns), tuple(coefficients), offset)


def _keep(work: _Work, name: str) -> int:
    """How much we want to keep ``name`` rather than substitute it away.

    A variable the modeller pinned with ``fixed = true`` is a state they asked for by name,
    an output is part of the model's interface, and a ``start`` value is a hint that
    somebody cares. A derivative variable this stage invented is the natural thing to lose.
    """
    variable = work.variables[name]
    score = 0
    if variable.fixed:
        score += 8
    if variable.causality == "output":
        score += 4
    if variable.start is not None:
        score += 2
    if name in work.integral:
        score -= 1
    return score


def _rank(work: _Work, name: str) -> tuple[int, int, int]:
    """Sort key for elimination: smallest goes first. Deep names lose to shallow ones."""
    return (_keep(work, name), -name.count("."), -work.order[name])


def _permits(work: _Work, victim: str, other: str, ratio: float, shift: ir.Expr) -> bool:
    """Whether ``victim`` may be replaced by ``ratio * other + shift``."""
    if work.variables[victim].fixed:
        return False
    if victim in work.integral:
        # Replacing der(x) by another variable renames x's derivative, and a name is all
        # we have -- a scaled or shifted derivative has nowhere to live.
        return ratio == 1.0 and shift == ZERO and other not in work.integral
    derivative = work.derivative.get(victim)
    # A second derivative of the victim would be left with no first derivative to follow.
    return derivative is None or derivative not in work.derivative


def _solo(work: _Work, affine: _Affine) -> _Elimination | None:
    """One unknown against a constant: the variable simply has a value."""
    (name,) = affine.unknowns
    if name in work.integral or name in work.derivative or work.variables[name].fixed:
        return None
    return _Elimination(name, simplify(neg(div(affine.offset, affine.coefficients[0]))))


def _pair(work: _Work, affine: _Affine) -> _Elimination | None:
    """Two unknowns with unit coefficients: ``a = ±b + k``, the shape connections make."""
    options: list[tuple[tuple[int, int, int], str, str, float, ir.Expr]] = []
    for position in (0, 1):
        victim, other = affine.unknowns[position], affine.unknowns[1 - position]
        mine, theirs = (
            _number(c) for c in (affine.coefficients[position], affine.coefficients[1 - position])
        )
        if mine is None or theirs is None or abs(mine) != 1 or abs(theirs) != 1:
            return None
        ratio = -theirs / mine
        shift = simplify(neg(div(affine.offset, affine.coefficients[position])))
        if _permits(work, victim, other, ratio, shift):
            options.append((_rank(work, victim), victim, other, ratio, shift))
    if not options:
        return None
    _, victim, other, ratio, shift = min(options)

    replacement = simplify(add(mul(ir.RealLiteral(ratio), _ref(other)), shift))
    derivative_of = (work.integral[victim], other) if victim in work.integral else None
    extra: tuple[tuple[str, ir.Expr], ...] = ()
    if victim in work.derivative:
        target = _ensure_derivative(work, other)
        extra = ((work.derivative[victim], mul(ir.RealLiteral(ratio), _ref(target))),)
    return _Elimination(victim, replacement, extra, derivative_of)


def _eliminate(work: _Work, index: int, elimination: _Elimination) -> None:
    """Drop one equation and the variables it let us stop carrying."""
    mapping = elimination.mapping()
    work.equations = [
        _apply_to(equation, mapping)
        for position, equation in enumerate(work.equations)
        if position != index
    ]
    work.aliases = {
        name: simplify(substitute(value, mapping)) for name, value in work.aliases.items()
    }
    work.aliases.update(mapping)
    for name in mapping:
        work.forget(name)
        base = work.integral.pop(name, None)
        if base is not None and work.derivative.get(base) == name:
            del work.derivative[base]
        derivative = work.derivative.pop(name, None)
        if derivative is not None:
            work.integral.pop(derivative, None)
    if elimination.derivative_of is not None:
        base, replacement = elimination.derivative_of
        work.derivative[base] = replacement
        work.integral[replacement] = base


def _eliminate_aliases(work: _Work) -> None:
    """Substitute away every ``a = ±b + k`` the model contains, to a fixed point."""
    while True:
        for index, equation in enumerate(work.equations):
            affine = _affine(work, equation)
            if affine is None:
                continue
            found = _solo(work, affine) if len(affine.unknowns) == 1 else _pair(work, affine)
            if found is None:
                continue
            _eliminate(work, index, found)
            break
        else:
            return


# --------------------------------------------------------------------------------------
# Pass 3 -- index reduction
# --------------------------------------------------------------------------------------


def _differentiate(work: _Work, index: int) -> int:
    """Append ``d/dt`` of equation ``index``, or return the one already appended."""
    existing = work.differentiated.get(index)
    if existing is not None:
        return existing
    equation = work.equations[index]
    constants = work.constants()
    try:
        derived = ir.Equal(d_dt(equation.lhs, constants), d_dt(equation.rhs, constants))
    except DerivativeError as error:
        raise CausalizationError(f"{work.name}: cannot differentiate {equation!r}") from error
    work.equations.append(_named(ir.Equal(simplify(derived.lhs), simplify(derived.rhs)), work))
    position = len(work.equations) - 1
    work.differentiated[index] = position
    work.parent[position] = index
    work.depth[position] = work.depth[index] + 1
    return position


def _reach(
    work: _Work,
    equation: int,
    assign: dict[str, int],
    variables: set[str],
    equations: set[int],
) -> bool:
    """Pantelides' augmenting search, over variables of top differentiation order only.

    Colouring is the point: when the search fails, ``variables`` and ``equations`` hold
    exactly the subsystem that has more equations than top-order variables, which is
    exactly what has to be differentiated.
    """
    equations.add(equation)
    eligible = [name for name in work.incidence(equation) if name not in work.derivative]
    for name in eligible:
        if name not in assign:
            assign[name] = equation
            return True
    for name in eligible:
        if name in variables:
            continue
        variables.add(name)
        if _reach(work, assign[name], assign, variables, equations):
            assign[name] = equation
            return True
    return False


def _reduce_index(work: _Work) -> None:
    """Differentiate equations until every one of them can be assigned a top-order variable."""
    assign: dict[str, int] = {}
    budget = _BUDGET
    for start in range(len(work.equations)):
        index = start
        while True:
            variables: set[str] = set()
            equations: set[int] = set()
            if _reach(work, index, assign, variables, equations):
                break
            budget -= 1
            if budget <= 0:
                raise CausalizationError(f"{work.name}: index reduction did not terminate")
            for name in variables:
                _ensure_derivative(work, name)
            for equation in equations:
                _differentiate(work, equation)
            for name in variables:
                if name in assign:
                    assign[work.derivative[name]] = work.differentiated[assign[name]]
            index = work.differentiated[index]


# --------------------------------------------------------------------------------------
# Pass 4 -- matching, then sorting
# --------------------------------------------------------------------------------------


def _priority(work: _Work, name: str) -> tuple[int, int, int, int]:
    """Which variable an equation should try to solve for first.

    Anything with no derivative in the system *must* be solved for -- nothing integrates
    it -- so it goes first. Among the rest, higher derivatives go before lower ones, which
    is what keeps states at the bottom of each differentiation chain where they belong,
    and a variable the modeller gave a ``fixed`` start goes last of all.
    """
    return (
        1 if name in work.derivative else 0,
        -work.level(name),
        1 if work.variables[name].fixed else 0,
        work.order[name],
    )


@dataclass
class _Matching:
    """Which equation solves for which unknown.

    Any maximum matching is structurally valid, but they are not equally good: the
    unknowns left over become the states, so the choice among them *is* state selection.
    Two rules steer it. :func:`_priority` orders what an equation reaches for, and
    displacing an equation that is already settled is a last resort -- a displaced
    equation retreats to its second choice, and its second choice is, by construction,
    a variable something else wanted to leave alone.
    """

    work: _Work
    assign: dict[str, int] = field(default_factory=dict[str, int])
    matched: dict[int, str] = field(default_factory=dict[int, str])
    excluded: frozenset[str] = frozenset()
    """Unknowns no equation may claim -- the states, when a selection is being *tested*
    rather than discovered. Empty for the real run, where the leftovers decide it."""

    def candidates(self, equation: int) -> tuple[list[str], list[str]]:
        """The unknowns of ``equation`` as ``(preferred, rest)``, each best first.

        A differentiated equation gets one extra rule, and it is the rule that makes index
        reduction produce sensible states rather than merely valid ones: if the equation it
        came from solves for ``v``, this one should solve for ``der(v)``. Following the
        chain keeps a position and its own velocity as the states; jumping between chains
        leaves a selection that is structurally fine and numerically singular. That one is
        worth displacing for, so it is kept apart from the rest.
        """
        incident = [name for name in self.work.incidence(equation) if name not in self.excluded]
        parent = self.work.parent.get(equation)
        settled = self.matched.get(parent, "") if parent is not None else ""
        chain = self.work.derivative.get(settled)
        preferred = [chain] if chain in incident else []
        rest = sorted(
            (n for n in incident if n not in preferred),
            key=lambda n: _priority(self.work, n),
        )
        return preferred, rest

    def take(self, equation: int, name: str, visited: set[str]) -> bool:
        """Give ``name`` to ``equation``, moving whoever holds it somewhere else."""
        if name in visited:
            return False
        visited.add(name)
        holder = self.assign.get(name)
        if holder is not None and not self.augment(holder, visited):
            return False
        self.assign[name] = equation
        self.matched[equation] = name
        return True

    def augment(self, equation: int, visited: set[str]) -> bool:
        preferred, rest = self.candidates(equation)
        free = [name for name in rest if name not in self.assign]
        order = (*preferred, *free, *rest)
        return any(self.take(equation, name, visited) for name in order)

    def run(self) -> dict[int, str]:
        for index in range(len(self.work.equations)):
            if not self.augment(index, set()):
                raise CausalizationError(
                    f"{self.work.name}: no unknown left for equation {self.work.equations[index]!r}"
                )
        return self.matched


def _match(work: _Work) -> dict[int, str]:
    """Assign every equation the variable it computes. Whatever is left over is a state."""
    return _Matching(work).run()


@dataclass
class _Tarjan:
    """Strongly connected components, iteratively so that deep graphs cannot blow the stack.

    An edge means "needs", so a component is emitted only once everything it needs has
    been -- which makes the output order the evaluation order, for free.
    """

    edges: Mapping[int, list[int]]
    index_of: dict[int, int] = field(default_factory=dict[int, int])
    low: dict[int, int] = field(default_factory=dict[int, int])
    stack: list[int] = field(default_factory=list[int])
    on_stack: set[int] = field(default_factory=set[int])
    result: list[list[int]] = field(default_factory=list[list[int]])
    pending: list[tuple[int, Iterator[int]]] = field(
        default_factory=list[tuple[int, "Iterator[int]"]]
    )

    def push(self, node: int) -> None:
        self.index_of[node] = self.low[node] = len(self.index_of)
        self.stack.append(node)
        self.on_stack.add(node)
        self.pending.append((node, iter(self.edges[node])))

    def descend(self, node: int, remaining: Iterator[int]) -> bool:
        """Walk edges until one leads somewhere unvisited, which is pushed and reported."""
        for target in remaining:
            if target not in self.index_of:
                self.push(target)
                return True
            if target in self.on_stack:
                self.low[node] = min(self.low[node], self.index_of[target])
        return False

    def emit(self, root: int) -> None:
        """Pop the component rooted at ``root``: everything above it on the stack."""
        component: list[int] = []
        while True:
            top = self.stack.pop()
            self.on_stack.discard(top)
            component.append(top)
            if top == root:
                break
        self.result.append(component)

    def run(self, nodes: list[int]) -> list[list[int]]:
        for root in nodes:
            if root in self.index_of:
                continue
            self.push(root)
            while self.pending:
                node, remaining = self.pending[-1]
                if self.descend(node, remaining):
                    continue
                self.pending.pop()
                if self.low[node] == self.index_of[node]:
                    self.emit(node)
                if self.pending:
                    above = self.pending[-1][0]
                    self.low[above] = min(self.low[above], self.low[node])
        return self.result


def _blocks(work: _Work, matched: dict[int, str]) -> tuple[Block, ...]:
    """Sort the matched equations into block-lower-triangular order."""
    producer = {name: equation for equation, name in matched.items()}
    nodes = sorted(matched)
    edges = {
        equation: [
            producer[name]
            for name in work.incidence(equation)
            if name in producer and producer[name] != equation
        ]
        for equation in nodes
    }
    blocks: list[Block] = []
    for component in _Tarjan(edges).run(nodes):
        members = sorted(component)
        equations = tuple(work.equations[index] for index in members)
        unknowns = tuple(matched[index] for index in members)
        assignment = solve_for(residual(equations[0]), unknowns[0]) if len(members) == 1 else None
        blocks.append(Block(unknowns, equations, assignment))
    return tuple(blocks)


# --------------------------------------------------------------------------------------
# Putting it together
# --------------------------------------------------------------------------------------


def _states(work: _Work, matched: dict[int, str]) -> tuple[str, ...]:
    """The unknowns nothing solves for. They are states, or the model is unsolvable."""
    solved = set(matched.values())
    free = [name for name in work.variables if work.is_unknown(name) and name not in solved]
    stateless = [name for name in free if name not in work.derivative]
    if stateless:
        raise CausalizationError(
            f"{work.name}: nothing determines {', '.join(sorted(stateless))} -- the model is "
            f"structurally singular"
        )
    return tuple(sorted(free, key=lambda name: work.order[name]))


_SELECTIONS = 10_000
"""Candidate sets beyond which the alternatives are not enumerated.

Testing one is a matching pass, so ten thousand is a second or so. Past that,
:attr:`~modelica.ir.system.System.candidates` holds the chosen set alone -- said here rather
than left as a silent truncation, because "no alternatives" and "too many to look" are very
different answers and a caller reading the field cannot tell them apart.
"""


def _candidates(work: _Work, states: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Every admissible state set, the chosen one first.

    A set of unknowns can be the states exactly when holding them back still leaves an
    equation for every remaining unknown -- so the test is the matching itself, run once per
    candidate set with that set forbidden. The pendulum answers ``(y, vy)``, ``(x, vx)`` and
    a few structurally valid oddities; the two useful ones are the two the literature names,
    and they are two *exchanges* apart, which is why this enumerates rather than perturbing
    the chosen set.

    Structural, not numerical. A set appearing here means the equations can be arranged to
    compute everything else from it, not that the arrangement is well conditioned -- the
    pendulum's ``(y, vy)`` is on this list and still goes singular at the bottom of the
    swing. Which of them is *usable where* is a question for the runtime; this is the menu.

    This is the mode list a dynamic state selection would switch between. Nothing switches
    yet, and that is the open gap. The alternatives are recorded anyway because they are a
    property of the model rather than of the runtime.
    """
    differentiable = [
        name for name in work.variables if work.is_unknown(name) and name in work.derivative
    ]
    if comb(len(differentiable), len(states)) > _SELECTIONS:
        return (states,)
    found: list[tuple[str, ...]] = [states]
    for combination in combinations(differentiable, len(states)):
        selection = tuple(sorted(combination, key=lambda name: work.order[name]))
        if selection == states:
            continue
        try:
            _Matching(work, excluded=frozenset(selection)).run()
        except CausalizationError:
            continue
        found.append(selection)
    return tuple(found)


def _statement(statement: ir.Statement, work: _Work) -> ir.Statement:
    value = simplify(substitute(_name_derivatives(statement.value, work), work.aliases))
    return type(statement)(statement.target, value)


def _when(when: ir.When, work: _Work) -> ir.When:
    return ir.When(
        tuple(
            ir.WhenBranch(
                simplify(substitute(_name_derivatives(branch.condition, work), work.aliases)),
                tuple(_statement(s, work) for s in branch.body),
            )
            for branch in when.branches
        )
    )


def causalize(flat: FlatModel) -> System:
    """Arrange a flat model into states and a sorted sequence of blocks.

    Raises :class:`CausalizationError` when no such arrangement exists -- when an equation
    has nothing left to solve for, when a variable nothing computes has no derivative to
    integrate, or when index reduction asks for a derivative that does not exist.
    """
    work = _start(flat)
    _eliminate_aliases(work)
    _reduce_index(work)
    matched = _match(work)
    states = _states(work, matched)
    return System(
        name=flat.name,
        variables=tuple(sorted(work.variables.values(), key=lambda v: work.order[v.name])),
        states=states,
        candidates=_candidates(work, states),
        derivative={name: work.derivative[name] for name in states},
        blocks=_blocks(work, matched),
        index=1 + max(work.depth.values(), default=0),
        aliases=dict(work.aliases),
        whens=tuple(_when(when, work) for when in flat.whens),
        initial_equations=tuple(
            _apply_to(_named(equation, work), work.aliases) for equation in flat.initial_equations
        ),
    )
