model Brusselator "Two chemical concentrations that feed each other into a steady oscillation -- a limit cycle"
  parameter Real A = 1 "Feed of the first species, held constant";
  parameter Real B = 3 "Feed of the second; the mixture oscillates once B exceeds 1 + A^2";
  Real x(start = 1, fixed = true) "Concentration of the autocatalytic species";
  Real y(start = 1, fixed = true) "Concentration of the species it consumes";
equation
  der(x) = A + x * x * y - (B + 1) * x;
  der(y) = B * x - x * x * y;
end Brusselator;
