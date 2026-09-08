model KeplerOrbit "The Kepler two-body orbit: an eccentric ellipse conserving energy and angular momentum"
  parameter Real mu(unit = "m3/s2") = 1 "Gravitational parameter G*M of the central body";
  Real x(unit = "m", start = 0.4, fixed = true) "Position, x component";
  Real y(unit = "m", start = 0, fixed = true) "Position, y component";
  Real vx(unit = "m/s", start = 0, fixed = true) "Velocity, x component";
  Real vy(unit = "m/s", start = 2, fixed = true) "Velocity, y component";
equation
  der(x) = vx;
  der(y) = vy;
  der(vx) = -mu * x / (x ^ 2 + y ^ 2) ^ 1.5;
  der(vy) = -mu * y / (x ^ 2 + y ^ 2) ^ 1.5;
end KeplerOrbit;
