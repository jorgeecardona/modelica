model SchwarzschildOrbit "A test mass orbiting a Schwarzschild black hole: a precessing relativistic orbit"
  parameter Real bigm = 1 "Black-hole mass in geometrized units G = c = 1 (a length)";
  parameter Real ell = 4 "Specific angular momentum, conserved; > sqrt(12)*M for a bound orbit";
  Real r(start = 20, fixed = true) "Areal radius, in units of the mass";
  Real phi(unit = "rad", start = 0, fixed = true) "Azimuthal angle in the equatorial plane";
  Real vr(start = 0, fixed = true) "Radial rate dr/dtau, per unit proper time";
equation
  der(r) = vr;
  der(phi) = ell / r ^ 2;
  der(vr) = -bigm / r ^ 2 + ell ^ 2 / r ^ 3 - 3 * bigm * ell ^ 2 / r ^ 4;
end SchwarzschildOrbit;
