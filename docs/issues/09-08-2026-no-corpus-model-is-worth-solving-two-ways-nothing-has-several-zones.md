---
title: No corpus model is worth solving two ways — nothing has several zones under one schedule
status: open
area: sim
owner: null
filed: 2026-08-09
blocked_by:
  - a-model-can-be-integrated-and-cannot-be-asked-where-it-settles
kind: idea
scope: M
verify: test
---

## Measured

- `examples/library/Thermal.mo` holds three components and no more: `HeatCapacitor` (one state,
  `T`), `ThermalConductor` (`Q = G*(T_a - T_b)`) and `FixedTemperature`
  (`examples/library/Thermal.mo:11,20,29`).
- The thermal models in the corpus are `NewtonCooling` (one state, closed form), `TwoBodyHeat`
  (two capacitors and one conductor — it equilibrates and stops, nothing drives it) and `HeatBar`
  (the method of lines: one bar, discretised in space). **None has a schedule, and none has more
  than one room-sized zone sharing a source.**
- `RoastingMachine` is the corpus's only model with a prescribed schedule in `time`
  (`logs/2026-08-01.md`), and it is a single thermal loop, not a network.
- **The public example this is modelled on**, a JuliaHub post about Dyad
  (<https://www.linkedin.com/posts/hvac-share-7491182510975107074-Kp5Q/>): one acausal model of a
  three-zone commercial building supporting both steady-state HVAC sizing and 24-hour dynamic
  analysis, *"the same equations, parameters, and thermal network"* across both.

## Inferred

- Inferred: three zones under a shared supply is the smallest model where the two analyses ask
  genuinely different questions of the same equations — sizing wants the steady load that holds
  setpoint, the transient wants what the temperature does overnight — and where a model that gets
  one right and the other wrong would be caught, because both are stated.
- Inferred: it exercises what the corpus does not have anywhere. Three zones coupled to each
  other and to a supply give an algebraic loop across the network, which is what
  `tearing-is-priced-in-by-three-documents-and-implemented-by-none` would bite on; a daily
  schedule gives time events; a thermostat gives state events. `TwoBodyHeat` has none of the
  three, so nothing in the corpus tests them together.
- Inferred, and the reason this is an `idea` rather than a `story`: the physics is arbitrary until
  someone picks the numbers. It becomes a story when the zones, the conductances and the schedule
  are written down with a source for each — the way `RoastingMachine`'s parameters are a real
  Loring S15 and not invented.

## Done

- A corpus model: three thermal zones, conduction between them and to ambient, a shared supply on
  a daily schedule, a thermostat on at least one zone. Written twice, in `.mo` and in `.py`, as
  every example here is.
- `Thermal.mo` gains what it lacks for that — at minimum a heat-flow source, and an ambient whose
  temperature varies with `time`. The Python twin gains the same.
- A `.toml` with reference points from an independent integration, and a `[steady]` section for
  the sizing answer.
- **The invariant that makes it worth two analyses:** with the schedule held at a constant, the
  transient arrives at the `[steady]` answer. Same flat model, both ways, checked in CI rather
  than claimed in prose.

## Deliberately not done

- No air flow, no humidity, no comfort model, no solar gain. Three lumped capacities and the
  conductances between them. A building-physics library is a different project and the MSL
  already has one.
- No sizing *optimisation*: choosing a supply capacity to hit a target is a solve over parameters,
  which belongs to the calibration issue and not here.
- Not a replacement for `TwoBodyHeat`. That model earns its place by being small enough to check
  by hand; this one earns a different one.
