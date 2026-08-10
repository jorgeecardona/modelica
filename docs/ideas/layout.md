# Drawing a graph so a person can read it

This project draws its own diagrams, and every complaint about them so far has been the same
complaint: a wire bends where it did not need to, a wire runs through something, a label sits
on a box. Those were fixed one rule at a time, and the fixes worked — but the last one, the
alignment term in `modelica.diagram.layout`, was invented here without checking whether
anybody had done it before.

They had. This page is what August 2026 found, and it is a reading list rather than a plan.

!!! info "Short version"
    Graph drawing is a subfield with named algorithms, and the one this corpus wants is
    **Sugiyama layered placement with port constraints and orthogonal routing** — which
    [ELK][elk] implements and calls `ELK Layered`. ELK already has the exact knob added here
    by hand, and puts it where the evidence says it belongs: **in node placement, not in
    routing**. Worth knowing before writing more of this by feel: MathWorks does not ship a
    comprehensive automatic layout for Simulink either, and the community tool for it
    delegates to Graphviz.

## The two families

**Layered (Sugiyama).** Assign nodes to layers, order within each layer to reduce crossings,
then assign coordinates. It suits a graph with a *direction* — signal flow, data flow, a
control loop. [ELK Layered][elk-layered] is the reference implementation: it "arranges as many
edges as possible into one direction by placing nodes into subsequent layers", supports
straight, orthogonal and spline routing, and with orthogonal routing "arbitrary port
constraints are respected, thus enabling the layout of block diagrams such as actor-oriented
models or **circuit schematics**". That last phrase is this corpus.

**Topology-shape-metrics (orthogonal).** Planarise to minimise crossings, choose the *shape* —
where the bends go — then choose coordinates and compact. [yFiles' orthogonal layout][yfiles]
describes the three phases and says it is "well suited for small and medium-sized sparse
graphs" and "produces compact drawings with no overlaps, few crossings, and few bends". A
circuit is undirected and sparse, which is exactly that description.

So the corpus wants both, and it already splits along that line without knowing it:
`_signal` handles the causal block diagrams and `_loop` handles the circuits.

## The thing this project reinvented

`layout.ALIGN` pulls a connected pair into line and took the corpus from 50 % of wires
straight to 75 %. ELK has the same idea as a documented option: one "defines how important it
is to keep an edge straight, i.e. aligned with one of the two axes, which is **evaluated
during node placement**".

Evaluated during node placement. That is the part worth taking: straightness is a property of
where the boxes are, and the routing can only make the best of what placement left it. Every
fix here before the alignment term was a routing fix, and each one hit a floor that the
placement had set.

The standard algorithm for that step is **Brandes–Köpf**, ["Fast and Simple Horizontal
Coordinate Assignment"][bk] (GD 2001): linear time, takes the ordering within each layer as
given, and maximises the number of straight edges by aligning each node with a chosen
neighbour and resolving conflicts. It is, in the authors' successors' words, "implemented and
taught widely". Read the [2020 erratum][bk-erratum] first — it documents two flaws, one of
which "requires a non-trivial adaptation".

## That this is hard is not a local failing

The question that started this page was "why are we not able to fix this". A fair answer is
that the tool with the largest block-diagram user base in engineering does not ship a fix
either: as of the survey behind the [Simulink Auto Layout Tool][autolayout], "there does not
exist a comprehensive commercial automatic layout tool for Simulink models", and that
community tool offers three backends of which the first is Graphviz.

## What this suggests, and what it does not

Suggested, and not decided here:

* **Placement is where the remaining wins are.** The routing rules are done and measured —
  no crossing, 6 px clearance for wires and labels, no false corner. The bends that remain
  exist because two connected boxes are not in line, which is a placement fact.
* **The two layouts already split the right way**, so a layered algorithm for `_signal` and
  an orthogonal one for `_loop` is a refinement of the existing shape rather than a rewrite.
* **A dependency is a real option and is not obviously the right one.** [elkjs][elkjs] is
  JavaScript; ELK proper is Java. Neither is a Python dependency this project would take
  lightly, and `docs/ideas/landscape.md` records why the core stays free of them.

Not suggested: replacing the force layout wholesale. It is the fallback for models that are
neither a loop nor causal, and every measurement here says it is adequate at that.

[elk]: https://eclipse.dev/elk/
[elk-layered]: https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html
[elkjs]: https://github.com/kieler/elkjs
[yfiles]: https://docs.yworks.com/yfiles-html/dguide/layout/orthogonal_layout.html
[bk]: https://link.springer.com/chapter/10.1007/3-540-45848-4_3
[bk-erratum]: https://arxiv.org/abs/2008.01252
[autolayout]: https://github.com/McSCert/Auto-Layout
