model Duffing "A beam between two magnets, shaken: one degree of freedom, driven, and chaotic"
  parameter Real delta = 0.3 "Damping";
  parameter Real alpha = -1 "Linear stiffness; negative, so the centre is a hilltop between two wells";
  parameter Real beta = 1 "Cubic stiffness, which makes the wells";
  parameter Real gamma = 0.5 "Amplitude of the drive";
  parameter Real omega = 1.2 "Angular frequency of the drive";
  Real x(start = 1, fixed = true) "Displacement";
  Real v(start = 0, fixed = true) "Velocity";
equation
  der(x) = v;
  der(v) = gamma * cos(omega * time) - delta * v - alpha * x - beta * x ^ 3;
end Duffing;
