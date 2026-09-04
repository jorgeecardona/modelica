model Lorenz "The Lorenz system: three coupled equations, deterministic and chaotic -- the butterfly"
  parameter Real sigma = 10 "Prandtl number";
  parameter Real rho = 28 "Rayleigh number, past the onset of chaos at 24.74";
  parameter Real beta = 8 / 3 "Geometric factor";
  Real x(start = 1, fixed = true);
  Real y(start = 1, fixed = true);
  Real z(start = 1, fixed = true);
equation
  der(x) = sigma * (y - x);
  der(y) = x * (rho - z) - y;
  der(z) = x * y - beta * z;
end Lorenz;
