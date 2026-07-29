model Pendulum "A point mass on a rigid rod, in Cartesian coordinates -- an index-3 DAE"
  parameter Real m(unit = "kg") = 1 "Point mass";
  parameter Real L(unit = "m") = 1 "Rod length";
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  Real x(unit = "m", start = 1, fixed = true) "Horizontal position";
  Real y(unit = "m", start = 0, fixed = true) "Vertical position, downward negative";
  Real vx(unit = "m/s", start = 0, fixed = true);
  Real vy(unit = "m/s", start = 0, fixed = true);
  Real F(unit = "N") "Rod force -- the constraint multiplier";
equation
  der(x) = vx;
  der(y) = vy;
  m * der(vx) = -F * x / L;
  m * der(vy) = -F * y / L - m * g;
  x ^ 2 + y ^ 2 = L ^ 2;
end Pendulum;
