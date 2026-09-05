model LotkaVolterra "Prey and predators: two populations that chase each other around a closed orbit, forever"
  parameter Real alpha = 1.1 "Prey birth rate, per unit time";
  parameter Real beta = 0.4 "How fast predators eat prey, per predator";
  parameter Real delta = 0.1 "How much of what is eaten becomes predators";
  parameter Real gamma = 0.4 "Predator death rate, per unit time";
  Real x(start = 10, fixed = true) "Prey";
  Real y(start = 10, fixed = true) "Predators";
equation
  der(x) = alpha * x - beta * x * y;
  der(y) = delta * x * y - gamma * y;
end LotkaVolterra;
