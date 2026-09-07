model MagLev "A steel ball levitated under an electromagnet: an unstable plant held at a fixed gap by PD feedback, the magnetic force pulling up as the inverse square of the gap"
  parameter Real m(unit = "kg") = 0.05 "Ball mass";
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  parameter Real R(unit = "Ohm") = 1 "Coil resistance";
  parameter Real L(unit = "H") = 0.01 "Coil inductance";
  parameter Real km(unit = "N.m2/A2") = 0.001 "Magnetic force constant";
  parameter Real z_ref(unit = "m") = 0.01 "Gap the controller holds";
  parameter Real Kp(unit = "V/m") = 200 "Proportional gain";
  parameter Real Kd(unit = "V.s/m") = 5 "Derivative gain";
  Real z(unit = "m", start = 0.012, fixed = true) "Gap between the ball and the magnet";
  Real v(unit = "m/s", start = 0, fixed = true) "Rate the gap changes";
  Real i(unit = "A", start = 0.22147234590350104, fixed = true) "Coil current";
  Real i_ref(unit = "A") "Current that balances gravity at the target gap";
  Real V(unit = "V") "Coil voltage the controller commands";
  Real F(unit = "N") "Magnetic force lifting the ball";
equation
  i_ref = z_ref * sqrt(m * g / km);
  V = R * i_ref + Kp * (z - z_ref) + Kd * v;
  F = km * (i / z) ^ 2;
  der(z) = v;
  m * der(v) = m * g - F;
  L * der(i) = V - R * i;
end MagLev;
