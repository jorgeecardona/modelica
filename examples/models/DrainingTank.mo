model DrainingTank "Torricelli outflow through an orifice -- the tank empties, then stays empty"
  parameter Real A(unit = "m2") = 1 "Tank cross-section";
  parameter Real a(unit = "m2") = 0.02 "Orifice area";
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  Real level(unit = "m", start = 1, fixed = true) "Water level";
  Real outflow(unit = "m3/s") "Volumetric outflow";
equation
  outflow = if level > 0 then a * sqrt(2 * g * level) else 0;
  A * der(level) = -outflow;
end DrainingTank;
