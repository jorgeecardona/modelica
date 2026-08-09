---
title: The return wire is routed through the component it comes back to
status: open
area: diagram
owner: null
filed: 2026-08-09
kind: bug
scope: M
verify: inspect
---

## Measured

Rendering `RLCCircuit` today (`python -m modelica.diagram examples/models/RLCCircuit.mo
examples/library/Electrical.mo --depth 2`) produces a real schematic — resistor zigzag,
inductor coils, capacitor plates, a `+-` source, a ground symbol, orthogonal wires and a
junction dot. Icons work. One route does not.

The components sit on the row `y = 62`, spanning:

    r    27.8 .. 96.2
    l   127.0 .. 195.4
    c   226.2 .. 294.6

and the wire closing the loop is emitted as

    <path class="link" d="M 127.0 190.0 L 114.8 190.0 L 106.0 190.0 L 106.0 62.0 L 27.8 62.0" />

That last segment runs from `x = 106` to `x = 27.8` **along `y = 62`** — the components' own
row. It therefore passes straight through the resistor's icon (27.8..96.2) and over the
`r`-to-`l` wire. On screen the resistor has a line drawn through it.

## Inferred

Inferred from the geometry rather than from the code: the router takes the *pin* as the
target and goes to it by the shortest orthogonal path, without treating a component's own
extent as something to route around. `ff68f8f` fixed the neighbouring case — "a wire does not
cross the symbol it just left" — so the rule exists for the departing end and not for the
arriving one.

Inferred consequence, and the reason this is a bug rather than a nicety: a line through a
resistor is not a cosmetic flaw in a schematic, it is a *different circuit*. A reader cannot
tell it from a short across the component. `docs/architecture.md:217` is the standing argument
that a picture which misrepresents the model is worse than no picture.

## Done

- The wire that closes `RLCCircuit`'s loop reaches `r.p` without crossing `r`, and the same
  holds for `RCCircuit` and `ProbeRC`.
- A route that must pass a component goes around its extent, on a row or column no component
  occupies.
- `verify: inspect` — the artifact is the evidence. The docs site is where it is looked at,
  and `docs/notebooks/drawing.ipynb` is the page that shows it.

## Deliberately not done

- No general orthogonal router with obstacle avoidance. Every component in the corpus sits on
  one of a few rows; leaving the row is enough, and a full router is a much larger thing to
  own and to test.
