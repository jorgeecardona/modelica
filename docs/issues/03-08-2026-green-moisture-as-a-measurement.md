---
title: RoastingMachine's moisture is assumed, and Cropster has it measured per lot
status: blocked
area: sim
owner: jorge
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

## What is measured so far, 2026-08-09

**Blocked, and on the one thing this repository cannot produce**: the per-lot green moisture.
`verify: probe`. Everything below was run; the claim was not.

The rest of the issue was done anyway, because none of it depends on the answer. When the
number arrives, the organic residual is a lookup.

### The exchange rate between the two unknowns

Sweeping `moisture` from 10 % to 16 % and simulating each, with every other parameter left
where the joint fit put it:

| green moisture | water lost | organic residual | probe rms |
| --- | --- | --- | --- |
| | % of charge | % of charge | K, 5 measured points |
| 10.0 % | 8.92 | 5.86 to 6.08 | 6.098 |
| 11.0 % | 9.77 | 5.01 to 5.23 | 4.171 |
| 12.0 % | 10.60 | 4.18 to 4.40 | 2.356 |
| 13.0 % | 11.43 | 3.35 to 3.57 | 1.138 |
| 14.0 % | 12.25 | 2.53 to 2.75 | 2.075 |
| 15.0 % | 13.05 | 1.73 to 1.95 | 3.775 |
| 16.0 % | 13.85 | 0.93 to 1.15 | 5.552 |

The residual column is the measured total loss, 14.78 to 15.00 %, less the water the model
removes.

It is a straight line to within 0.028 points over that whole range:

    water lost (% of charge) = 0.821 * green moisture (%) + 0.739

so **one point of green moisture is 0.82 points of charge**, and the organic residual moves
0.82 points the other way. The slope is not 1.00 because the water is also a heat sink: more
of it means cooler beans and a smaller *fraction* driven off — 88.8 % at 11 % moisture, 87.9 %
at 13 %, 87.0 % at 15 %.

### What the current numbers are, and why they prove nothing

At the model's `moisture = 0.13` the residual is 3.35 to 3.57 %, which brackets the assumed
3.5 %. That is not agreement. The 11.37 % water target in
`examples/models/RoastingMachine.toml` was *derived* as 14.89 % less an assumed 3.5 %, so the
model reproduces the assumption it was given. The circle is closed and empty, which is exactly
what the entry says.

### The probe curve is not indifferent to moisture, and that does not help either

The right-hand column has a minimum near 13.1 % (1.129 K) and costs 4.2 K by 10 %. It is
tempting to read that as the curve determining the moisture on its own.

It does not, and the reason is already written in that `.toml`: the seven machine parameters
were fitted *jointly with* `moisture = 0.13`. Freezing them and sweeping the eighth asks where
the eighth belongs given the other seven, and the other seven were chosen to suit it. The same
trap caught the exotherm question on 2026-07-30. Reading a determination off that column would
repeat it.

So the sweep gives an exchange rate, not an answer. The rate is the useful part.

## The one question

For the blend roasted at Sincerely Coffee on 2026-07-30, five batches at a 13.8 kg charge:
**what does Cropster's per-lot moisture column say?** One number, one lot.

With it:

- read the organic residual off the line above;
- if it lands in the low single figures, the model's decomposition is confirmed by something
  other than itself, and `moisture` becomes a measured input in the `.toml` with a source;
- if it does not, that is the finding, and the entry says to write it down rather than absorb
  it into a parameter.

## Deliberately not done, found while doing it

- **No refit.** Sweeping one parameter with seven co-fitted ones frozen is what the table does,
  and its limits are stated above. A joint refit at each moisture would give a better rms curve
  and a worse argument: it would fit the assumption harder, not test it.
- **The `.toml` note now carries the exchange rate.** It already said a moisture meter "would
  turn the softest two numbers on this page into one measurement and one residual"; it now says
  what the conversion is, so the next reader does not have to re-run the sweep.
- **The scale resolution has not been folded in.** The 14.78 to 15.00 % band comes from end
  weights 0.01 to 0.03 kg apart on about 11.75 kg, which the original capture flagged as being
  at the resolution of the scale. The band is carried through as a band and no narrower claim is
  made from it.
