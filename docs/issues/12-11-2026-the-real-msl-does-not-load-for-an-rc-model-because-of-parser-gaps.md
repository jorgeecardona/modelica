---
title: The real MSL does not load for an RC model, because of parser gaps
status: open
kind: story
scope: M
owner: jorgeecardona
filed: 2026-11-12
area: lang
verify: test
blocked_by: null
fixed_in: null
---

# The real MSL does not load for an RC model, because of parser gaps

Goal: `run(model, libraries=[".../Modelica"])` loads MSL 4.1.0 for a model on
`Modelica.Electrical.Analog.Basic.Resistor`, `Capacitor` and `Ground`.

`modelica.lang.library.closure` reads only the files a model's names reach. For the RC model
that is 29 files. The parse stops in one of them. A file is parsed whole, so a gap anywhere in
it blocks it, even when the model never uses the class with the gap.

## Gaps, in the order the RC model hits them

1. Fixed (c330daa): a leading dot in an expression, `.Modelica.Blocks.Types.SimpleController.PID`,
   `.sqrt(x)`.
2. Fixed (062bef3): `end` in a subscript, `a[end]` (`Blocks/Continuous.mo`, line 1147).
3. Open: an output list whose targets have subscripts and colons,
   `(eigenvalues[:, 1], eigenvalues[:, 2], info) := LAPACK.dgeev_eigenValues(A);`
   (`Math/package.mo`, line 1690). RC needs this file for `Modelica.Math.exp`.

Known from the eager load (`library.load` with `skipped`: 59 of the MSL files refuse), not yet
on the RC path:

- A component of a record array element, `v[i].re` (`ComplexMath.mo`, line 19).
- A subscript on a non-last name in a `connect` endpoint (`Electrical/Analog/Lines/OLine.mo`,
  line 81).
- A subscript on an inner component (`Electrical/Analog/Basic/M_Transformer.mo`, line 32).

## Plan

Close the gaps parse-only, in the order the RC model hits them. Stop when RC loads, or after
five gaps. Test: with the MSL cache present, the RC model simulates through `run(...,
libraries=[msl])` to `exp(-t / RC)`.
