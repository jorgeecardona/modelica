model Rossler "The Rössler system: a spiral that folds, one nonlinear term, chaotic -- and gentler than Lorenz"
  parameter Real a = 0.2 "Spiral growth in the x-y plane";
  parameter Real b = 0.2 "Lift of the fold";
  parameter Real c = 5.7 "Where the fold begins; chaotic past about 5.7";
  Real x(start = 1, fixed = true);
  Real y(start = 1, fixed = true);
  Real z(start = 1, fixed = true);
equation
  der(x) = -y - z;
  der(y) = x + a * y;
  der(z) = b + z * (x - c);
end Rossler;
