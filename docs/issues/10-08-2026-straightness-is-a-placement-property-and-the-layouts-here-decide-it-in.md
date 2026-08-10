---
title: Straightness is a placement property, and the layouts here decide it in the router
status: open
area: diagram
owner: null
filed: 2026-08-10
scope: L
verify: probe
kind: story
---

## Measured

Four routing rules landed over 2026-08-09 and 10 and each is held by a test in
`tests/test_diagram_layout.py`: no wire crosses a component it does not touch, wires and
labels keep 6 px clearance from every box, an attachment slides along its side to meet the
other end, and no wire keeps a corner it does not turn at.

They took the corpus from 11 % of wires straight to 50 %. The fifth change was not a routing
change at all -- an alignment term in `modelica.diagram.layout._force`, pulling a connected
pair into line -- and it took 50 % to **75 %**, corners 30 to 13 at depth 1 and 45 to 17 at
depth 2. Swept:

    weight   straight (depth 1)   corners
     0.0            50 %             30
     1.0            75 %             13
     2.5            53 %             29, and a clearance test fails

The routing rules had reached a floor that the placement set.

Read after the fact rather than before, and recorded in `docs/ideas/layout.md`: [ELK
Layered](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html) carries
the same idea as a documented option -- how important it is "to keep an edge straight, i.e.
aligned with one of the two axes, which is **evaluated during node placement**". The standard
algorithm for that step is Brandes-Köpf, *Fast and Simple Horizontal Coordinate Assignment*
(GD 2001), linear time, with a [2020 erratum](https://arxiv.org/abs/2008.01252) naming two
flaws, one needing "a non-trivial adaptation".

## Inferred

Inferred from the sweep's shape rather than from theory: `ALIGN` is a crude version of the
right idea and it is near its ceiling. It is non-monotonic -- past about 1.5 the alignment
overpowers the springs, the arrangement drops into a different basin, and `_separate` pushes
boxes apart along the axis just squeezed. A term inside an iterative force loop cannot do
better than that, because it is fighting the other terms rather than solving for coordinates.

Also inferred: the corpus already splits the way the literature does. `_signal` places causal
block diagrams, which is the layered/Sugiyama case; `_loop` places circuits, which are
undirected and sparse, which is the topology-shape-metrics case. Neither is currently anything
of the sort -- `_signal` is a longest-path ranking with a barycentre order and no coordinate
assignment step, and `_loop` walks a cycle onto a rectangle.

## Done

- `_signal` assigns coordinates with a straightness-maximising pass over its existing layers,
  rather than centring each column. Brandes-Köpf is the named candidate; anything that beats
  the current numbers on the measurements below is acceptable.
- The measurements move and are recorded: percentage of wires straight and total corners, at
  both depths, against the numbers above.
- The four routing rules still hold. They are tests, so this is checked rather than hoped.
- Whether `ALIGN` survives is decided by measurement, not left in place beside a better
  mechanism.

## Deliberately not done

- **No dependency on ELK.** `elkjs` is JavaScript and ELK proper is Java; `docs/ideas/landscape.md`
  records why the core takes no dependency lightly. What is borrowed is the design, and the
  reading list is `docs/ideas/layout.md`.
- **Nothing for `_loop`.** Circuits are the orthogonal/TSM case and that is a separate entry
  with separate evidence.
- **The force layout stays** as the fallback for a model that is neither a loop nor causal.
  Every measurement says it is adequate at that.
