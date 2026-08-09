model TwoFalls "Two bodies dropped from different heights, sharing one gravity"
  inner Translational.Environment environment(g = 9.81);
  Translational.FallingBody high(s(start = 100, fixed = true), v(start = 0, fixed = true));
  Translational.FallingBody low(s(start = 20, fixed = true), v(start = 0, fixed = true));
  Real gap(unit = "m") "How far apart they are -- constant, because gravity ignores mass";
equation
  gap = high.s - low.s;
end TwoFalls;
