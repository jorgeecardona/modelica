model CoulombOscillator "A mass on a spring damped by dry friction, its amplitude falling linearly not exponentially"
  parameter Real m(unit = "kg") = 1 "Mass";
  parameter Real k(unit = "N/m") = 1 "Spring stiffness";
  parameter Real fc(unit = "N") = 0.1 "Coulomb friction force magnitude";
  parameter Real v_eps(unit = "m/s") = 0.05 "Velocity scale that smooths the friction sign";
  Real x(unit = "m", start = 1.5, fixed = true) "Position";
  Real v(unit = "m/s", start = 0, fixed = true) "Velocity";
equation
  der(x) = v;
  m * der(v) = -k * x - fc * tanh(v / v_eps);
end CoulombOscillator;
