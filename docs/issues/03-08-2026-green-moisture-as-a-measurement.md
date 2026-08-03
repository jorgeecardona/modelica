---
title: RoastingMachine's moisture is assumed, and Cropster has it measured per lot
status: open
area: sim
owner: null
filed: 2026-08-03
kind: idea
scope: S
verify: probe
---

## Measured

- `examples/models/RoastingMachine.mo` carries `moisture` as a parameter, added on 2026-07-30
  ("lets add humidity of the beans please as a parameter as well", `logs/2026-07-30.md`). Its
  value is assumed, not measured.
- The mass balance closes only with a second assumption: about 3.5 % organic loss on top of the
  water. Also assumed.
- The measured end of it, read off a screen recording of the roaster's Cropster session by
  `mighty-asimo-ddcb2`: charge **13.8 kg** (not the rated 15), end weights 11.73-11.76 kg, so
  **14.78-15.00 % lost**, four batches of the same blend at the same charge. Caveat recorded at
  the time: the end weights are three distinct values 0.01-0.03 kg apart on ~11.75 kg, which is
  at the resolution of the scale.
- `logs/2026-07-30.md` names the opportunity: "Green moisture is askable — Cropster has a
  per-lot moisture column John doesn't display. That would turn `moisture` and the assumed 3.5 %
  organic loss into one measurement and one residual."

## Inferred

With green moisture measured, the two assumptions separate. Today one number (total loss ~15 %)
is explained by two unknowns (water + organic), so any split fits and the model's `moisture` is
unidentifiable from the weighings alone. Given measured moisture, water loss is known and the
organic fraction becomes a **residual** — a number the model predicts and the scale checks,
rather than a number chosen to make the scale agree.

This is inference from the structure of the mass balance, not a claim that the current fit is
wrong. The fit is good on temperature (rms 1.03-1.60 K across five batches,
`logs/2026-07-30.md`); it is the *decomposition* of the mass loss that is unconstrained.

## Done

- The per-lot green moisture for the batches in the 2026-07-30 capture, obtained from Cropster.
- `examples/models/RoastingMachine.toml` records it as a measured input with its source, the way
  the other measured quantities in that spec are recorded.
- The organic loss becomes a stated residual with a number: predicted against 14.78-15.00 %
  measured, with the scale's own resolution quoted beside it.
- If the residual does not close, that is the finding, and it is written down rather than
  absorbed into a parameter.

## Deliberately not done

- No first-crack exotherm term. It was reached for on 2026-07-30 to explain the mass gap and the
  conclusion was **withdrawn** the same day: the seven machine parameters had been fitted against
  a model whose latent sink was 4x too small, so `UA`, `G_loss` and `Q_max` had already absorbed
  the missing heat. Refitting jointly reached both the curve and the mass. Adding an exotherm on
  top of that fit cannot be justified by this data.
- Blocked on nothing in this repository. It is one question to one person.
