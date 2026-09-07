model ServoDrive "A DC motor under PI speed control: the loop drives the shaft to a speed setpoint"
  parameter Real w_ref(unit = "rad/s") = 100 "Speed setpoint";
  parameter Real Kp(unit = "V.s/rad") = 0.1 "Proportional gain";
  parameter Real Ki(unit = "V/rad") = 1 "Integral gain";
  parameter Real R(unit = "Ohm") = 0.5 "Armature resistance";
  parameter Real L(unit = "H") = 0.05 "Armature inductance";
  parameter Real km(unit = "N.m/A") = 0.1 "Motor constant, torque and back-EMF";
  parameter Real J(unit = "kg.m2") = 0.001 "Rotor and load inertia";
  parameter Real b(unit = "N.m.s/rad") = 0.0002 "Load damping";
  Real i(unit = "A", start = 0, fixed = true) "Armature current";
  Real w(unit = "rad/s", start = 0, fixed = true) "Shaft speed";
  Real x(unit = "rad", start = 0, fixed = true) "Integral of the speed error";
  Real e(unit = "rad/s") "Speed error, setpoint minus measurement";
  Real v(unit = "V") "Armature voltage the controller commands";
  Real tau(unit = "N.m") "Electromagnetic torque";
equation
  e = w_ref - w;
  v = Kp * e + Ki * x;
  tau = km * i;
  der(x) = e;
  L * der(i) = v - R * i - km * w;
  J * der(w) = tau - b * w;
end ServoDrive;
