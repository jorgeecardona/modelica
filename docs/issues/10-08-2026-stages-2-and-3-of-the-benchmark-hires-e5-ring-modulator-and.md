---
title: "Stages 2 and 3 of the benchmark: HIRES, E5, Ring Modulator, and ScalableTestSuite"
status: open
area: sim
owner: null
filed: 2026-08-10
scope: L
verify: probe
kind: story
---

## Measured

`scripts/bench.py` exists and `moon run :bench` runs it. Stage 1 of
`docs/ideas/benchmarks.md:84` is measured and published in
[`work-precision-harness`](03-08-2026-work-precision-harness.md): `Robertson` and `VanDerPol`,
three methods, an error axis, on a named machine.

Stages 2 and 3 are not, and neither is blocked on the harness. What each needs, measured:

- **Stage 2 -- `HIRES`, `E5`, `Ring Modulator`.** None is in `examples/models/`. `HIRES` is 8
  states, `E5` is 4 with a very wide range, `Ring Modulator` is 15 and is the one where the
  linear solve starts to dominate. Each has to be written as a `.mo`, written again as its
  Python twin, and given a `.toml` with reference points -- the corpus rule -- before it can
  be timed. Adding a problem to `PROBLEMS` in the harness is one line; getting the model right
  is not.
- **Stage 3 -- `ScalableTestSuite`.** An external Modelica library, not vendored here. It is
  the only stage where compile time is the thing under test rather than the integrator, and
  the only one with a public baseline: OpenModelica publishes per-branch results against it.

## Inferred

Inferred from stage 1 rather than from having run these: the reference is the hard part, not
the timing. On `VanDerPol` two solvers at rtol = atol = 1e-12 agree only to 1.6e-8 relative,
so a self-computed reference supports an error axis to 1e-6 and no deeper. `E5` spans a much
wider dynamic range than either stage-1 problem, so the same limit will bite sooner and harder.

Which makes the Bari published solutions a prerequisite for stage 2 in a way they were only an
improvement for stage 1. That is inference from one problem's behaviour, not a measurement of
`E5`.

## Done

- The three stage-2 problems are corpus models, each written twice with a `.toml`, and each
  appears in the harness's `PROBLEMS`.
- Their reference solutions come from the Bari test set, not from this harness. Where a
  published value is used, the `.toml` says which one.
- The work-precision tables are published with the four things
  `docs/ideas/benchmarks.md:94` demands: the problem, the tolerance sweep, the reference, and
  the machine.
- Stage 3 is a separate decision recorded either way: either `ScalableTestSuite` is vendored
  or fetched and the compile-time numbers are published against OpenModelica's, or it is
  written down here why not.

## Deliberately not done

- Not a CI gate, for the reason `work-precision-harness` already gives.
- No new harness. `scripts/bench.py` takes a problem, a horizon and a tolerance list; these
  are entries in a table.

## HIRES landed 2026-08-13 — one of the three, and a new kind of reference point

`examples/models/HIRES.{mo,py,toml}`, and `"HIRES": (321.8122, (1e-4, 1e-6, 1e-8))` in
`scripts/bench.py`'s `PROBLEMS`.

Eight states, the equations exactly as the Bari report prints them, ``y0 = (1, 0, 0, 0, 0, 0,
0, 0.0057)``, ``0 <= t <= 321.8122``. The horizon is not derived from anything and the report
says so: "The end point of the integration interval, 321.8122, was chosen arbitrarily".

### The reference is somebody else's

This entry said "the reference is the hard part, not the timing", and named the Bari published
solutions as a prerequisite. Taken literally rather than approximated: the `.toml` point is
Table II.1.1 of report `hires.pdf` from *Test Set for Initial Value Problem Solvers*, release
2.4 (Mazzia and Magherini, University of Bari and INdAM, February 2008), computed there by
RADAU5 on a Cray C90 with ``uround = 1.01e-19``.

Agreement, measured:

| tolerance | worst relative difference from the published values |
| --- | --- |
| 1e-8 | 8.6e-8 |
| 1e-10 | 7.5e-10 |
| 1e-12 | **5.1e-11** |

`y7 + y8 = 0.0057` to machine precision throughout, which is conservation of the enzyme and
follows from `der(y7) + der(y8) = 0` by inspection.

**`tests/test_examples.py::SOURCES` gained a fourth kind, `published`.** Its own docstring
already admitted the gap -- ``analytic`` and ``reference`` are "both of them things this
project could in principle have got wrong in the same way twice" -- and every `source =
"reference"` in the corpus until now meant an independent integration *by this project*. A
published number cannot be wrong in the same way twice because nothing about it came from
here. It is required to carry a `citation` naming the work **and the table**, since a test set
prints run characteristics beside its solutions and only one of those is the answer.

That requirement is a small change with a wide blast radius, and it is the part of this work
that outlives HIRES: the corpus can now hold somebody else's arithmetic and say whose.

### The harness runs it

    HIRES, to t = 321.812, 8 states
      method       rtol    seconds   rel error   steps
      Radau       1e-04    0.0390s    4.04e-03      81
      Radau       1e-06    0.0452s    1.59e-04     131
      Radau       1e-08    0.0990s    1.57e-06     290
      BDF         1e-04    0.0222s    2.04e-01     130
      BDF         1e-06    0.0409s    1.68e-03     214
      BDF         1e-08    0.0764s    1.68e-05     358
      LSODA       1e-04    0.0057s    7.09e-02     160
      LSODA       1e-06    0.0116s    1.18e-03     309
      LSODA       1e-08    0.0227s    6.76e-05     575

Which is the comparison this problem was chosen for: LSODA is four to seven times faster than
Radau here and is not on Robertson, because HIRES is *mildly* stiff -- about three decades of
eigenvalue spread against Robertson's nine -- so a method that switches has somewhere to
switch to. Not published as a table yet; that is the next paragraph.

### What is left on this entry

- **`E5` and `Ring Modulator`.** Nothing done. `E5` is the one this entry predicts will be
  hardest, and the prediction stands: the published reference matters more there than here
  precisely because its dynamic range is wider.
- **The work-precision tables.** The numbers above were run on this machine and pasted here.
  `docs/ideas/benchmarks.md:94` wants the problem, the sweep, the reference and the machine
  published together, and pasting into a register entry is not that.
- **Stage 3, `ScalableTestSuite`.** Untouched, and still needing a decision recorded either
  way.

## E5 landed 2026-08-13 — two of three, and the entry's prediction was right

`examples/models/E5.{mo,py,toml}`. Four species, `0 <= t <= 1e13`, from the same Bari release
2.4, report `e5.pdf`, Table II.11.3.

This entry predicted: "`E5` spans a much wider dynamic range than either stage-1 problem, so
the same limit will bite sooner and harder." It bit in three places, and none of them was the
one predicted.

### 1. One tolerance is not enough

`y1` starts at 1.76e-3 and is 1e-291 at the horizon; `y2` and `y3` never exceed 4e-10. The
source says so directly, quoting Shampine 1981: "a scalar absolute error control is unsuitable
and a componentwise scaled absolute error control would be recommendable for this problem".

`simulate` already had `absolute`; nothing that reads a spec passed it. **Six runners** --
`test_examples` (three call sites), `test_verify`, `test_radau`, `test_fold`,
`test_solution_set`, `scripts/make-gallery` -- integrated from a `.toml` while ignoring half of
what it said. With `atol` defaulting to `tolerance`, E5 fails as
``E5: Required step size is less than spacing between numbers`` before `t` is anywhere. A
runner that integrates a model differently from the way its spec says to is testing a
different model, so `[simulate] absolute` is now honoured everywhere.

### 2. The rearrangement is the answer, not a nicety

The source recommends writing the third row as `y3' = y2' - y4'` "in order to avoid eventual
cancellation of digits". Both forms measured at rtol = 1e-8, atol = 1.7e-24:

| | invariant `y2 - y3 - y4` | `y2` against the published value |
| --- | --- | --- |
| f(y) as printed | 9.6e-21 | **wrong by 100 %** |
| with the substitution | 1.3e-25 | 1.1e-3 |

Four orders on the invariant and three on the answer. `y2` and `y3` are both about 8.9e-23 at
the horizon and their difference is 1.3e-25, so the cancellation is not a subtlety: without
the rearrangement the difference *is* the round-off. The corpus model is written in the
recommended form and the `.toml` carries both numbers.

### 3. Double precision cannot reach the published point, and the spec says so

Agreement is **1.1e-3 relative on `y2` and `y3`**, and tightening does not help -- 1e-8, 1e-10
and 1e-12 all land within a factor of two of each other. The reason is in the report's own run
characteristics: the reference was computed at ``rtol = h0 = 1.1e-18`` with ``uround =
1.01e-19``, on an Alphaserver. Both are below double precision's eps.

So the point is asserted at `rtol = 5e-3`, and that is the finding rather than a compromise.
Every other point in this corpus is asserted near machine precision; this one is asserted at
the accuracy double arithmetic can actually reach.

`y1` is deliberately not asserted at all. Its published value is 1.15e-291 and the run's
absolute tolerance is 1.7e-24: claiming agreement there would be claiming to have integrated
267 decades below the error control.

### Still left

- **`Ring Modulator`.** Untouched. 15 states, and this entry says it is where the linear solve
  starts to dominate.
- **The work-precision tables**, still pasted into this register rather than published in the
  form `docs/ideas/benchmarks.md:94` asks for. E5 is *not* in `scripts/bench.py`'s `PROBLEMS`:
  the harness grades against a self-computed reference from two methods agreeing, and on a
  problem where neither method is accurate to better than 1e-3 that reference does not exist.
  Adding the row would produce an error axis measured against nothing.
- **Stage 3, `ScalableTestSuite`.** Untouched.

## Ring Modulator, 2026-08-13 — written, verified, and deliberately not landed

The third stage-2 problem, from the same Bari release 2.4, report `ringmod.pdf`. Fifteen
states, four diodes with an `exp` nonlinearity, a 1 kHz signal mixed into a 10 kHz carrier,
`0 <= t <= 1e-3`. Written out and run:

    15 states, index 1, 23 blocks
    rtol = atol = 1e-6
    73.8 s, 119,723 steps
    worst relative difference from Table II.3.2: 6.35e-5, on y14

**It works, and it agrees.** Two things worth recording beyond that.

`numpy` reports `overflow encountered in exp` twice during the run and the integration
survives it. That is the condition the test set's own Fortran guards with `IERR = -1`, and
Table II.3.1 lists RADAU (m = 0..26), RADAU5 (m = 0..9) and VODE as **failed runs** for exactly
that reason -- "solver cannot handle IERR=-1". Here the overflow produces an infinity, the step
is rejected, and the solver backs off. Not a claim of superiority: SciPy's Radau is not the
Fortran RADAU5, and the comparison is between what the two harnesses do with an infinity rather
than between the methods.

**It is not landed, and the reason is cost, not correctness.** Seven test call sites simulate
every corpus model, so this one would add about nine minutes to an eight-minute suite. That is
filed as
[`the-corpus-has-no-way-to-hold-a-model-that-costs-a-minute`](13-08-2026-the-corpus-has-no-way-to-hold-a-model-that-costs-a.md),
and landing Ring Modulator is on *that* entry's Done list rather than being forced through
here.

The model is kept here verbatim so the next person does not rewrite it. It is the report's
f(y) with the auxiliary voltages and diode currents given names, which is what makes it
readable as a circuit rather than as fifteen rows.

```modelica
model RingModulator "A ring modulator mixing a 1 kHz signal into a 10 kHz carrier -- fifteen states and four diodes"
  parameter Real pi = 3.141592653589793 "Half a turn, since there is no standard library here";
  parameter Real C = 1.6e-8 "The two output capacitors";
  parameter Real Cs = 2e-12 "Diode junction capacitance -- what makes this stiff rather than index-2";
  parameter Real Cp = 1e-8;
  parameter Real Lh = 4.45 "The two transformer inductances";
  parameter Real Ls1 = 0.002;
  parameter Real Ls2 = 5e-4;
  parameter Real Ls3 = 5e-4;
  parameter Real gamma = 40.67286402e-9 "Diode saturation current";
  parameter Real delta = 17.7493332 "Diode exponent, per volt";
  parameter Real R = 25000;
  parameter Real Rp = 50;
  parameter Real Rg1 = 36.3;
  parameter Real Rg2 = 17.3;
  parameter Real Rg3 = 17.3;
  parameter Real Ri = 50;
  parameter Real Rc = 600;
  Real y1(start = 0, fixed = true) "U1";
  Real y2(start = 0, fixed = true) "U2";
  Real y3(start = 0, fixed = true) "U3";
  Real y4(start = 0, fixed = true) "U4";
  Real y5(start = 0, fixed = true) "U5";
  Real y6(start = 0, fixed = true) "U6";
  Real y7(start = 0, fixed = true) "U7";
  Real y8(start = 0, fixed = true) "I1";
  Real y9(start = 0, fixed = true) "I2";
  Real y10(start = 0, fixed = true) "I3";
  Real y11(start = 0, fixed = true) "I4";
  Real y12(start = 0, fixed = true) "I5";
  Real y13(start = 0, fixed = true) "I6";
  Real y14(start = 0, fixed = true) "I7";
  Real y15(start = 0, fixed = true) "I8";
  Real Uin1 "The low-frequency signal, 1 kHz";
  Real Uin2 "The carrier, 10 kHz";
  Real UD1 "Voltage across diode 1";
  Real UD2;
  Real UD3;
  Real UD4;
  Real qD1 "Current through diode 1";
  Real qD2;
  Real qD3;
  Real qD4;
equation
  Uin1 = 0.5 * sin(2000 * pi * time);
  Uin2 = 2 * sin(20000 * pi * time);
  UD1 = y3 - y5 - y7 - Uin2;
  UD2 = -y4 + y6 - y7 - Uin2;
  UD3 = y4 + y5 + y7 + Uin2;
  UD4 = -y3 - y6 + y7 + Uin2;
  qD1 = gamma * (exp(delta * UD1) - 1);
  qD2 = gamma * (exp(delta * UD2) - 1);
  qD3 = gamma * (exp(delta * UD3) - 1);
  qD4 = gamma * (exp(delta * UD4) - 1);
  der(y1) = (y8 - 0.5 * y10 + 0.5 * y11 + y14 - y1 / R) / C;
  der(y2) = (y9 - 0.5 * y12 + 0.5 * y13 + y15 - y2 / R) / C;
  der(y3) = (y10 - qD1 + qD4) / Cs;
  der(y4) = (-y11 + qD2 - qD3) / Cs;
  der(y5) = (y12 + qD1 - qD3) / Cs;
  der(y6) = (-y13 - qD2 + qD4) / Cs;
  der(y7) = (-y7 / Rp + qD1 + qD2 - qD3 - qD4) / Cp;
  der(y8) = -y1 / Lh;
  der(y9) = -y2 / Lh;
  der(y10) = (0.5 * y1 - y3 - Rg2 * y10) / Ls2;
  der(y11) = (-0.5 * y1 + y4 - Rg3 * y11) / Ls3;
  der(y12) = (0.5 * y2 - y5 - Rg2 * y12) / Ls2;
  der(y13) = (-0.5 * y2 + y6 - Rg3 * y13) / Ls3;
  der(y14) = (-y1 + Uin1 - (Ri + Rg1) * y14) / Ls1;
  der(y15) = (-y2 - (Rc + Rg1) * y15) / Ls1;
end RingModulator;
```

The reference point, Table II.3.2, computed there by PSIDE at `atol = rtol = 1e-13`:

    y1  = -0.2339057358486745e-1     y9  = -0.2840029933642329e-7
    y2  = -0.7367485485540825e-2     y10 =  0.7267198267264553e-3
    y3  =  0.2582956709291169        y11 =  0.7929487196960840e-3
    y4  = -0.4064465721283450        y12 = -0.7255283495698965e-3
    y5  = -0.4039455665149794        y13 = -0.7941401968526521e-3
    y6  =  0.2607966765422943        y14 =  0.7088945416976114e-4
    y7  =  0.1106761861269975        y15 =  0.2390059075236570e-4
    y8  =  0.2939904342435596e-6
