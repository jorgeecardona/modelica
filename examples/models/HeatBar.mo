model HeatBar "A rod discretised into n cells: the method of lines, written by hand"
  parameter Integer n = 8 "Number of cells";
  parameter Real a(unit = "1/s") = 0.4 "Diffusivity over the square of the cell width";
  parameter Real T_left(unit = "degC") = 100 "The hot end, held";
  parameter Real T_0(unit = "degC") = 20 "What the rod starts at, everywhere";
  Real T[n](start = T_0, fixed = true, unit = "degC") "Cell temperatures, left to right";
equation
  der(T[1]) = a * (T_left - 2 * T[1] + T[2]);
  for i in 2:n - 1 loop
    der(T[i]) = a * (T[i - 1] - 2 * T[i] + T[i + 1]);
  end for;
  der(T[n]) = a * (T[n - 1] - T[n]);
end HeatBar;
