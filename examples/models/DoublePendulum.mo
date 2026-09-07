model DoublePendulum "Two rods, two masses, released from horizontal: the textbook chaotic machine, in angles"
  parameter Real m1(unit = "kg") = 1 "Upper mass";
  parameter Real m2(unit = "kg") = 1 "Lower mass";
  parameter Real L1(unit = "m") = 1 "Upper rod";
  parameter Real L2(unit = "m") = 1 "Lower rod";
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  Real th1(start = 1.5707963267948966, fixed = true, unit = "rad") "Upper angle from the vertical, down";
  Real th2(start = 1.5707963267948966, fixed = true, unit = "rad") "Lower angle from the vertical, down";
  Real w1(start = 0, fixed = true, unit = "rad/s") "Upper angular velocity";
  Real w2(start = 0, fixed = true, unit = "rad/s") "Lower angular velocity";
  Real d(unit = "rad") "The angle between the rods";
  Real den "The common denominator of both accelerations; never zero for positive masses";
  Real E(unit = "J") "Total energy, kinetic plus potential: zero at the start and conserved";
equation
  d = th1 - th2;
  den = 2 * m1 + m2 - m2 * cos(2 * d);
  der(th1) = w1;
  der(th2) = w2;
  der(w1) = (-g * (2 * m1 + m2) * sin(th1) - m2 * g * sin(th1 - 2 * th2) - 2 * sin(d) * m2 * (w2 ^ 2 * L2 + w1 ^ 2 * L1 * cos(d))) / (L1 * den);
  der(w2) = 2 * sin(d) * (w1 ^ 2 * L1 * (m1 + m2) + g * (m1 + m2) * cos(th1) + w2 ^ 2 * L2 * m2 * cos(d)) / (L2 * den);
  E = (m1 + m2) / 2 * L1 ^ 2 * w1 ^ 2 + m2 / 2 * L2 ^ 2 * w2 ^ 2 + m2 * L1 * L2 * w1 * w2 * cos(d) - (m1 + m2) * g * L1 * cos(th1) - m2 * g * L2 * cos(th2);
end DoublePendulum;
