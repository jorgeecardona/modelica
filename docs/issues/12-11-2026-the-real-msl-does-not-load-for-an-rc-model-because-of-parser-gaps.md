---
title: The real MSL does not load for an RC model, because of parser gaps
status: done
kind: story
scope: M
owner: jorgeecardona
filed: 2026-11-12
area: lang
verify: test
blocked_by: null
fixed_in: 0.1.x
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
3. Fixed (this commit): an output list whose targets have subscripts and colons,
   `(eigenvalues[:, 1], eigenvalues[:, 2], info) := LAPACK.dgeev_eigenValues(A);`
   (`Math/package.mo`, line 1690). RC needs this file for `Modelica.Math.exp`.

After these three, all 29 files the RC model reaches parse, given both roots `Modelica` and
`ModelicaServices` (`Modelica.Constants.eps` is `ModelicaServices.Machine.eps`). The next stop
is over-reading, not a need: `Blocks/package.mo` holds its examples inline, and
`Blocks.Examples.Rectifier12pulseFFT` extends
`Modelica.Electrical.Machines.Examples.Transformers.Rectifier12pulse`, whose file has a
subscript on an inner component (line 55). `closure` scans a whole file once it reads it; it
would have to follow names per class, not per file, to skip that example.

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

## Result (2026-11-12)

Done. `closure` now follows names per class: a class is scanned only when a name the model
reaches resolves to it, so `Blocks.Examples.Rectifier12pulseFFT` pulls in nothing. The RC
model on MSL 4.1.0 (`libraries=[.../Modelica, .../ModelicaServices]`) reads 24 files, nothing
under `Machines` or `Blocks`, loads in 2.95 s (most of it parsing the 11,705-line
`Math/package.mo`) and gives `c.v(1) = exp(-1)`. Test: `tests/test_library.py`, skipped
without the MSL cache. Three parser gaps were closed on the way; the others listed above stay
open for models that reach them.
