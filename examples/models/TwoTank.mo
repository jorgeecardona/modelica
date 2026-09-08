model TwoTank "Two interacting tanks draining by gravity, their levels coupled through a valve"
  parameter Real area1(unit = "m2") = 1.5 "Cross-sectional area of tank 1";
  parameter Real area2(unit = "m2") = 1.0 "Cross-sectional area of tank 2";
  parameter Real q_in(unit = "m3/s") = 0.5 "Constant inflow into tank 1";
  parameter Real k1(unit = "m2.5/s") = 0.6 "Coupling-valve coefficient, tank 1 to tank 2";
  parameter Real k2(unit = "m2.5/s") = 0.4 "Outlet-valve coefficient, tank 2 to atmosphere";
  Real h1(unit = "m", start = 0.5, fixed = true) "Liquid level in tank 1";
  Real h2(unit = "m", start = 0.1, fixed = true) "Liquid level in tank 2";
  Real q1(unit = "m3/s") "Flow from tank 1 into tank 2 through the coupling valve";
  Real q2(unit = "m3/s") "Outflow from tank 2 to atmosphere";
equation
  q1 = k1 * sqrt(h1 - h2);
  q2 = k2 * sqrt(h2);
  area1 * der(h1) = q_in - q1;
  area2 * der(h2) = q1 - q2;
end TwoTank;
