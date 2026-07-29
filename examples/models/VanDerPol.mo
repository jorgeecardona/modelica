model VanDerPol "The van der Pol oscillator -- a relaxation limit cycle; large mu makes it stiff"
  parameter Real mu = 1000 "Nonlinearity, and the stiffness ratio";
  Real x(start = 2, fixed = true);
  Real y(start = 0, fixed = true);
equation
  der(x) = y;
  der(y) = mu * ((1 - x ^ 2) * y - x);
end VanDerPol;
