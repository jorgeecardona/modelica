model PurePursuit "A pursuer that always steers toward a target: pure-pursuit guidance with atan2"
  parameter Real xt(unit = "m") = 10 "Target x position";
  parameter Real yt(unit = "m") = 0 "Target y position";
  parameter Real V(unit = "m/s") = 1 "Pursuer speed, constant";
  parameter Real k(unit = "1/s") = 3 "Heading turn-rate gain";
  Real x(unit = "m", start = 0, fixed = true) "Pursuer x position";
  Real y(unit = "m", start = 8, fixed = true) "Pursuer y position";
  Real theta(unit = "rad", start = 0, fixed = true) "Pursuer heading";
  Real thd(unit = "rad") "Desired heading, the bearing to the target";
equation
  thd = atan2(yt - y, xt - x);
  der(theta) = k * (thd - theta);
  der(x) = V * cos(theta);
  der(y) = V * sin(theta);
end PurePursuit;
