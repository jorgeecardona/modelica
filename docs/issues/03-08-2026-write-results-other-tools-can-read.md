---
title: A Result is readable only by this package, and the whole field reads one `.mat` format
status: open
area: sim
owner: null
filed: 2026-08-03
kind: story
scope: M
verify: test
---

## Measured

- `modelica.sim.result.Result` holds names, times and values, and is read through this
  package's own API. `grep -rn "savemat\|\.mat\b" src/` finds nothing: there is no way to get
  a run out of here into anything else except by writing the loop yourself.
- The Dymola/OpenModelica binary result format is what the field reads. Three maintained
  Python packages exist to do nothing but read it:
  - [ModelicaRes](https://github.com/kdavies4/ModelicaRes) — "only loads Dymola/OpenModelica-
    formatted binary and text results (`*.mat` and `*.txt`)", plus plotting and analysis;
  - DyMat — exports that format to CSV, Gnuplot, MATLAB and netCDF;
  - [BuildingsPy](https://pypi.org/project/buildingspy) — processes `*.mat` output from
    Dymola, OPTIMICA or OpenModelica as part of library regression testing.
- The format is not a secret and not large: a `MATLAB level 4` file with `Aclass`, `name`,
  `description`, `dataInfo`, `data_1` and `data_2` matrices, where `data_1` holds constants
  and `data_2` the trajectories, and `dataInfo` maps each variable to a row and a sign.

## Inferred

This is the cheapest interoperability in the whole project, and the direction matters.
*Reading* the format buys the ability to compare against other tools' runs, which the
reachability-and-agreement issue will want anyway. *Writing* it buys every plotting script,
every regression harness and every analysis notebook that already exists in this field, for
one emitter and no dependency — `scipy.io.savemat` is already available under the `sim`
extra, and the format can be written without it if that matters.

Inferred, not measured: the negative sign convention in `dataInfo` (a variable stored as the
negative of another, which is how aliases are packed) is the part most likely to be got
wrong, because nothing in a round trip through this package alone would catch it. Test
against a file a different tool wrote.

## Done

- `Result.write(path)` produces a Dymola-format `.mat` that ModelicaRes and DyMat open, and
  that reports the same values this package reports.
- `modelica.sim` reads one back into a `Result`, so a run by another tool is an ordinary
  value here — which is what makes an agreement check against OpenModelica possible.
- The round trip is tested both ways, and the reading side is tested against a `.mat`
  **produced by another tool** and committed as a fixture. A round trip through our own
  writer proves only that we are self-consistent.
- Aliases are packed the way the format expects, negative sign included, and that case has
  its own test.

## Deliberately not done

- No new required dependency. If `scipy.io` is used it is behind the `sim` extra, which is
  where the numerics already live; the core promise that reading a model's structure costs
  nothing does not change.
- No CSV, no netCDF, no plotting. DyMat converts from `.mat` to all of those already, and
  writing the one format everybody reads is what buys the rest.
