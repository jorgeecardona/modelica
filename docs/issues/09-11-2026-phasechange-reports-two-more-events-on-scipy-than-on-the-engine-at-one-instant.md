---
title: PhaseChange reports two more events on SciPy than on the engine at one instant
status: done
kind: bug
scope: S
owner: jorgeecardona
filed: 2026-11-09
area: sim
verify: test
blocked_by: null
fixed_in: 0.0.266
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

## Fixed

The count already agreed (12 and 12). The order at t = 709.749 differed: `T > Ms` and `T < Ms`
cross at one instant, and the engine took the root one ULP earlier. Both backends now treat
roots within 8 ULPs as one instant and fire the lower index first.
`tests/test_vm.py::test_phase_change_fires_the_same_events_in_the_same_order`.
