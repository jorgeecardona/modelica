model FisherKPP "The Fisher-KPP equation: a logistic-growth invasion front by the method of lines"
  parameter Integer n = 16 "Interior grid points";
  parameter Real d(unit = "m2/s") = 1 "Diffusion coefficient";
  parameter Real r(unit = "1/s") = 1 "Logistic growth rate";
  parameter Real length(unit = "m") = 10 "Domain length";
  parameter Real h(unit = "m") = length / (n + 1) "Grid spacing";
  parameter Real a(unit = "1/s") = d / h ^ 2 "Discrete-Laplacian stiffness";
  parameter Real u_left = 1 "Population density held at the left boundary (invaded)";
  Real u[n](start = 0, fixed = true) "Population density at each interior point";
equation
  der(u[1]) = a * (u_left - 2 * u[1] + u[2]) + r * u[1] * (1 - u[1]);
  for i in 2:n - 1 loop
    der(u[i]) = a * (u[i - 1] - 2 * u[i] + u[i + 1]) + r * u[i] * (1 - u[i]);
  end for;
  der(u[n]) = a * (u[n - 1] - 2 * u[n]) + r * u[n] * (1 - u[n]);
end FisherKPP;
