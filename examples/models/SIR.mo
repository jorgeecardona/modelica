model SIR "The Kermack-McKendrick SIR epidemic: susceptible, infected and recovered fractions"
  parameter Real beta(unit = "1/s") = 0.4 "Infection rate per susceptible-infected contact";
  parameter Real gamma(unit = "1/s") = 0.1 "Recovery rate; 1/gamma is the mean infectious time";
  Real s(start = 0.99, fixed = true) "Susceptible fraction of the population";
  Real i(start = 0.01, fixed = true) "Infected fraction of the population";
  Real r(start = 0, fixed = true) "Recovered (or removed) fraction of the population";
equation
  der(s) = -beta * s * i;
  der(i) = beta * s * i - gamma * i;
  der(r) = gamma * i;
end SIR;
