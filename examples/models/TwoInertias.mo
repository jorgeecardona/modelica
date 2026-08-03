model TwoInertias "Two inertias on one rigid shaft, braked: a constraint between states, so index 3"
  Rotational.Inertia big(J = 2, phi(start = 0, fixed = true), w(start = 10, fixed = true));
  Rotational.Inertia small(J = 0.5);
  Rotational.Damper brake(d = 0.25);
equation
  connect(big.flange_b, small.flange_a);
  connect(small.flange_b, brake.flange);
end TwoInertias;
