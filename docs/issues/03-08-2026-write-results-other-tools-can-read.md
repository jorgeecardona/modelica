---
title: A Result is readable only by this package, and the whole field reads one `.mat` format
status: done
area: sim
owner: null
filed: 2026-08-03
fixed_in: a09cc90
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

## What landed

`modelica.sim.matfile`, `Result.write(path)`, `modelica.sim.read(path)`.
`tests/test_matfile.py`, 5 tests. 1285 pass.

- **`Result.write` produces a file DyMat opens.** Not a claim: `test_dymat_opens_what_we_wrote`
  runs `DyMat.DyMatFile` over a written `RLCCircuit` run and checks all 32 variables and the
  abscissa. DyMat is one of the three readers this entry names and it is not a dependency --
  the test skips without it, the way `tests/test_jit.py` skips without Numba.
- **`read` brings one back**, and it is written to read what *other* tools write: the signed
  one-based `dataInfo` index and the `data_1` constants are both honoured.
- **The round trip is tested both ways**, and the reading side is tested against files this
  writer cannot produce -- built by hand with `scipy.io.savemat` and cross-read by DyMat.
- **The alias sign has its own test.** A variable stored as the negative of another is packed
  by pointing at that other one's row with a negative index. This writer never emits one, so
  the test builds the file and asserts that *our reader and DyMat agree* about what it means.
  Two readers agreeing on a sign is a stronger statement than either alone.

### Two ways to get it wrong that both produce a file that opens

Neither was visible from inside this package, and both were found by asking DyMat.

**The character matrices are not all stored the same way round.** `Aclass` is read row by row
and `name` and `description` are read *column* by column -- in the same file, by the same
reader. Writing both the natural way round gives a header that reads `Atra`, `ject`, `ory1`,
and DyMat answers "File structure not supported!".

**`scipy.io.savemat` writes an array's buffer, not its logical order.** A transposed *view*
goes to disk in the original order under a header claiming the new shape. That file opens,
reports the right number of variables, and names them `Te  v  .`, `i  c   i`, `m  .  l`. The
fix is one `np.ascontiguousarray`, and there is no way to notice it without a second reader:

    view .T          ['Te  v  .', 'i  c   i', 'm  .  l']
    contiguous .T    ['Time', 'c.v', 'l.i']

That is the entry's own argument -- "A round trip through our own writer proves only that we
are self-consistent" -- landing on something specific.

## Deliberately not done, found while doing it

- **Everything goes into `data_2`; `data_1` holds only the time range.** Sorting constants
  into `data_1` is what a Modelica tool does and it saves space, but it decides *from this
  run's samples* that a flat variable is a parameter. `read` handles `data_1` because other
  tools write it; `write` does not produce it.
- **No new dependency.** `scipy.io` is behind the `sim` extra, where the numerics already are.
  DyMat is test-only and optional.
- **No third-party *fixture* is committed**, and the entry asked for one. Neither Dymola nor
  OpenModelica is on this machine, so no file written by a Modelica tool exists here to
  commit. What replaced it is stronger in one way and weaker in another: DyMat is a real
  third-party *reader* and it validates every variable of every write, but nothing here has
  read a file that a Modelica tool actually produced. The first person with OpenModelica
  should drop one in and point the reader tests at it.
- **No CSV, no netCDF, no plotting**, as planned. DyMat converts from here.
