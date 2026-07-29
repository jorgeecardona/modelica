model NewtonCooling "A hot object cooling toward ambient -- the smallest useful model"
  parameter Real T_amb(unit = "degC") = 25 "Ambient temperature";
  parameter Real k(unit = "1/s") = 0.5 "Cooling coefficient";
  Real T(unit = "degC", start = 90, fixed = true) "Object temperature";
equation
  der(T) = -k * (T - T_amb);
end NewtonCooling;
