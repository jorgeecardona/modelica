model E5 "Chemical pyrolysis -- four species, thirteen decades of time, and badly scaled"
  parameter Real A = 7.89e-10 "k1, the slow decay of A1";
  parameter Real B = 11000000.0 "k3";
  parameter Real C = 1130.0 "k4";
  parameter Real M = 1000000.0 "k2 / k4, the ratio that carries the stiffness";
  Real y1(start = 0.00176, fixed = true) "Concentration of A1";
  Real y2(start = 0, fixed = true) "Concentration of A2";
  Real y3(start = 0, fixed = true) "Concentration of A3";
  Real y4(start = 0, fixed = true) "Concentration of A4";
equation
  der(y1) = -A * y1 - B * y1 * y3;
  der(y2) = A * y1 - M * C * y2 * y3;
  der(y3) = der(y2) - der(y4);
  der(y4) = B * y1 * y3 - C * y4;
end E5;
