# Finding where two tiers disagree

The same model runs in more than one way. The interpreter evaluates it. The emitted Python
kernel runs it. The engine runs it as a tape. Each of these is a second implementation of the same arithmetic.

When two of them give different answers, one of them is wrong. The problem is to find
**which equation** they first disagree on. A comparison of the final answers says only that
they disagree.

`modelica.sim.compare.diverges` answers the question. It evaluates one state in two tiers and
returns the first block where the values differ.

## The workflow

Build the model. Build the two tiers. Ask.

```python
from modelica.lang import parse
from modelica.sim import build, causalize, flatten
from modelica.sim.compare import diverges

path = "examples/models/RoastingMachine.mo"
system = causalize(flatten([parse(open(path).read(), path)], "RoastingMachine"))

plain = build(system)
folded = build(system, fold=True)

print(diverges(system, plain, folded))
```

Two tiers that agree give `None`:

```text
None
```

## What a disagreement looks like

This example plants one. It adds 1.0 to the block that computes `C_bean`, the heat capacity
of the charge, and builds a second kernel from the changed model.

```python
import dataclasses
from modelica.ir import nodes as ir

blocks = list(system.mode.blocks)
for position, block in enumerate(blocks):
    if block.unknowns[0] == "C_bean":
        blocks[position] = dataclasses.replace(
            block, assignment=ir.Binary("+", block.assignment, ir.RealLiteral(1.0))
        )
mode = dataclasses.replace(system.mode, blocks=tuple(blocks))
bent = dataclasses.replace(system, modes=(mode,))

print(diverges(system, build(system), build(bent)))
```

The report names the block, the variable, both values, the equation and the line:

```text
block 2 disagrees by 1, written at examples/models/RoastingMachine.mo:33:3
  C_bean: 29120.484 vs 29121.484
  C_bean = m_dry * cp_dry + m_w * cp_water;
```

## Why the first block, and not all of them

`C_bean` is read by the block that computes `der(T_bean)`, and that block runs after it. A
wrong value in an early block reaches every later block that reads it. A report of every
disagreement is a list of one cause and its consequences, with no way to see which is which.

The blocks run in a known order, so the first one to disagree is the cause. That is the one
this returns.

## Why one state, and not a run

Two runs that separate show you where they ended. They do not show you where they started to
separate, and that instant is usually much earlier.

So this function compares **one evaluation at one state**. Give it the state a run was in
when the two answers first differed, with the `t`, `y`, `p` and `w` arguments. It names the
block.

A run that follows both tiers until they separate is a larger tool. It is not built.

## Any two tiers

The two arguments are observers. An observer is anything with the signature of the emitted
`observe` function: it takes the time, the state, the parameters and the workspace, and it
writes every variable of the model into an output array.

A `Kernel` is an observer. An exported FMU becomes one with a small wrapper.

Neither side is privileged. The interpreter is a useful reference because it is the simplest
implementation, but the pairs that matter do not always contain it.

## The tolerance

Two Python tiers agree bit for bit, because they perform the same operations in the same
order.

The engine need not. A different `libm` rounds `exp` in a different last place. So
the default agreement is `1e-12`, relative and absolute. Pass `relative=0.0, absolute=0.0`
for a pair that must be exact.

## What the report can and cannot tell you

The report gives the equation as Modelica text with flat names. It gives the line where the
equation was written, when that line survived flattening.

Flattening keeps the line for an equation a modeller wrote. It has no line for an equation
the compiler made: a connection equation, or an equation that index reduction differentiated.
Over the example models, 77 of 86 block equations carry a line.
