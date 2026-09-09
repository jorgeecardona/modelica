model MixingRing "A closed ring of stirred tanks pumped one way: a tracer circulates and mixes to uniform, total mass conserved"
  parameter Integer n = 4 "Number of tanks around the ring";
  parameter Real k(unit = "1/s") = 1 "Pumped flow rate over the tank volume";
  parameter Real slack(unit = "mol/m3") = 1e-6 "Round-off the positivity guard allows below 0";
  Real c[n](unit = "mol/m3", start = {4, 0, 0, 0}, fixed = true) "Tracer concentration in each tank";
  Real total(unit = "mol/m3") "Total tracer over the ring, conserved";
equation
  for i in 1:n loop
    der(c[i]) = k * (c[mod(i - 2, n) + 1] - c[i]);
  end for;
  total = sum(c);
  assert(min(c) > -slack, "a tank concentration went negative");
end MixingRing;
