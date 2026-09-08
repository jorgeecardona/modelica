model PumpTank "A DC motor drives a pump that fills a tank against the head it builds"
  parameter Real V(unit = "V") = 12 "Supply voltage";
  parameter Real R(unit = "Ohm") = 0.5 "Armature resistance";
  parameter Real L(unit = "H") = 0.01 "Armature inductance";
  parameter Real k(unit = "N.m/A") = 0.1 "Motor constant, torque and back-EMF";
  parameter Real J(unit = "kg.m2") = 0.005 "Rotor and pump inertia";
  parameter Real b(unit = "N.m.s/rad") = 0.001 "Bearing damping";
  parameter Real kq(unit = "m3/rad") = 0.0004 "Pump displacement, flow per shaft angle";
  parameter Real c_load(unit = "N.m/m") = 0.675 "Head-to-torque coefficient";
  parameter Real A(unit = "m2") = 1 "Tank cross-section";
  parameter Real Rv(unit = "s/m2") = 100 "Outlet valve resistance, Q_out = h/Rv";
  Real i(unit = "A", start = 0, fixed = true) "Armature current";
  Real w(unit = "rad/s", start = 0, fixed = true) "Shaft speed";
  Real h(unit = "m", start = 0, fixed = true) "Tank head";
  Real Q(unit = "m3/s") "Pump flow into the tank";
  Real tau_pump(unit = "N.m") "Load torque the head imposes on the shaft";
equation
  Q = kq * w;
  tau_pump = c_load * h;
  L * der(i) = V - R * i - k * w;
  J * der(w) = k * i - b * w - tau_pump;
  A * der(h) = Q - h / Rv;
end PumpTank;
