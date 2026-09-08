model HenonHeiles "The Henon-Heiles system: a 2-DOF Hamiltonian, regular at low energy and mixing to chaos"
  Real x(start = 0, fixed = true) "Position, first coordinate";
  Real y(start = 0, fixed = true) "Position, second coordinate";
  Real px(start = 0.45, fixed = true) "Momentum conjugate to x (unit mass, so a velocity)";
  Real py(start = 0.15, fixed = true) "Momentum conjugate to y (unit mass, so a velocity)";
equation
  der(x) = px;
  der(y) = py;
  der(px) = -x - 2 * x * y;
  der(py) = -y - x ^ 2 + y ^ 2;
end HenonHeiles;
