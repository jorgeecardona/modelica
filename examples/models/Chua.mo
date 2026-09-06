model Chua "Chua's circuit: two capacitors, an inductor and a kinked resistor -- the double scroll"
  parameter Real alpha = 15.6 "Ratio of the two capacitances, scaled";
  parameter Real beta = 28 "The inductor, scaled";
  parameter Real m0 = -(8 / 7) "Slope of the resistor's middle segment";
  parameter Real m1 = -(5 / 7) "Slope of its outer segments";
  Real x(start = 0.7, fixed = true) "Voltage across the first capacitor";
  Real y(start = 0, fixed = true) "Voltage across the second capacitor";
  Real z(start = 0, fixed = true) "Current through the inductor";
  Real h "The resistor's characteristic: three straight lines joined at x = -1 and x = 1";
equation
  h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1));
  der(x) = alpha * (y - x - h);
  der(y) = x - y + z;
  der(z) = -beta * y;
end Chua;
