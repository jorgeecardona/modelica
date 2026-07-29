model Robertson "Robertson's autocatalytic reaction -- the standard stiff benchmark"
  Real y1(start = 1, fixed = true) "Concentration of A";
  Real y2(start = 0, fixed = true) "Concentration of B";
  Real y3(start = 0, fixed = true) "Concentration of C";
equation
  der(y1) = -0.04 * y1 + 1e4 * y2 * y3;
  der(y2) = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 ^ 2;
  der(y3) = 3e7 * y2 ^ 2;
end Robertson;
