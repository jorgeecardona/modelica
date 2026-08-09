model ProbeLC "The same probe with an inductor redeclared into the slot: an LC oscillation"
  Electrical.Probe probe(redeclare Electrical.Inductor element(L = 0.5, i(start = 0, fixed = true)));
end ProbeLC;
