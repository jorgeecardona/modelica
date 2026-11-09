---
title: PhaseChange reports two more events on SciPy than on the engine at one instant
status: open
kind: bug
scope: S
owner: jorgeecardona
filed: 2026-11-09
area: sim
verify: test
blocked_by: null
fixed_in: null
---

PhaseChange over its `stop` gives 14 events on `backend="python"` and 12 on `backend="engine"`.
Every event time agrees, and so does every reference point.

The difference is the event iteration at shared instants:

```text
python  117.916215: 4, 5, 4, 4   ...   709.749069: 4, 5
engine  117.916215: 4, 5         ...   709.749069: 5, 4
```

SciPy fires indicator 4 twice more at t = 117.916 after indicator 5, and the two backends order
4 and 5 differently at t = 709.749. One of the two event loops re-fires an indicator that is
already settled, or the other misses a real re-fire.

To do: find which one is right against the model's `when` semantics, make both agree, and add a
test that compares the event lists for PhaseChange on both backends.
