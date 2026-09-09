model WaveString "The wave equation on a string with fixed ends: a standing wave by the method of lines"
  parameter Integer n = 8 "Interior grid points";
  parameter Real c(unit = "m/s") = 1 "Wave speed along the string";
  parameter Real h(unit = "m") = 1 / (n + 1) "Grid spacing; the string has length 1";
  parameter Real k(unit = "1/s2") = c ^ 2 / h ^ 2 "Discrete-Laplacian stiffness";
  constant Real pi = 3.141592653589793 "The ratio of a circle's circumference to diameter";
  Real u[n](unit = "m") "Displacement of each interior point";
  Real v[n](unit = "m/s", start = 0, fixed = true) "Velocity of each interior point";
initial equation
  for i in 1:n loop
    u[i] = sin(pi * i / (n + 1));
  end for;
equation
  der(u) = v;
  der(v[1]) = k * (-2 * u[1] + u[2]);
  for i in 2:n - 1 loop
    der(v[i]) = k * (u[i - 1] - 2 * u[i] + u[i + 1]);
  end for;
  der(v[n]) = k * (u[n - 1] - 2 * u[n]);
end WaveString;
