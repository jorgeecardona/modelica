model Genset "A governed generator set: a PI speed governor holds the shaft at its setpoint, and when an electrical load is switched on the speed dips and the governor brings it back"
  parameter Real w_ref(unit = "rad/s") = 100 "Speed setpoint";
  parameter Real k(unit = "N.m/A") = 0.5 "Generator constant, torque and back-EMF";
  parameter Real R_gen(unit = "Ohm") = 0.5 "Generator internal resistance";
  parameter Real R_open(unit = "Ohm") = 10000 "Load resistance before the switch";
  parameter Real R_closed(unit = "Ohm") = 2 "Load resistance after the switch";
  parameter Real t_load(unit = "s") = 1 "Time the load is switched on";
  parameter Real b(unit = "N.m.s/rad") = 0.01 "Bearing friction";
  parameter Real J(unit = "kg.m2") = 0.05 "Shaft inertia";
  parameter Real Kp(unit = "N.m.s/rad") = 1 "Governor proportional gain";
  parameter Real Ki(unit = "N.m/rad") = 10 "Governor integral gain";
  Real w(unit = "rad/s", start = 0, fixed = true) "Shaft speed";
  Real x(unit = "rad", start = 0, fixed = true) "Integral of the speed error";
  Real R_load(unit = "Ohm") "Load resistance";
  Real v_gen(unit = "V") "Generator voltage";
  Real i(unit = "A") "Load current";
  Real tau_gen(unit = "N.m") "Braking torque from the generator";
  Real tau_eng(unit = "N.m") "Engine torque the governor commands";
equation
  R_load = if time < t_load then R_open else R_closed;
  v_gen = k * w;
  i = v_gen / (R_gen + R_load);
  tau_gen = k * i;
  tau_eng = Kp * (w_ref - w) + Ki * x;
  der(x) = w_ref - w;
  J * der(w) = tau_eng - b * w - tau_gen;
end Genset;
