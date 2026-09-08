model SaturatingServo "A position servo whose actuator voltage saturates: a step drives it to the limit first"
  parameter Real xref(unit = "m") = 1 "Position setpoint";
  parameter Real Kp(unit = "V/m") = 50 "Proportional gain on position error";
  parameter Real Kd(unit = "V.s/m") = 5 "Derivative gain on speed";
  parameter Real Vmax(unit = "V") = 12 "Actuator voltage limit";
  parameter Real R(unit = "Ohm") = 1 "Coil resistance";
  parameter Real L(unit = "H") = 0.05 "Coil inductance";
  parameter Real km(unit = "N/A") = 0.5 "Motor constant, force and back-EMF";
  parameter Real J(unit = "kg") = 0.02 "Moving mass";
  parameter Real b(unit = "N.s/m") = 0.1 "Viscous damping";
  Real x(unit = "m", start = 0, fixed = true) "Position";
  Real w(unit = "m/s", start = 0, fixed = true) "Speed";
  Real i(unit = "A", start = 0, fixed = true) "Coil current";
  Real e(unit = "m") "Position error, setpoint minus position";
  Real u(unit = "V") "Unsaturated voltage the PD law asks for";
  Real v(unit = "V") "Actuator voltage after the plus/minus Vmax limit";
equation
  e = xref - x;
  u = Kp * e - Kd * w;
  v = max(-Vmax, min(Vmax, u));
  der(x) = w;
  J * der(w) = km * i - b * w;
  L * der(i) = v - R * i - km * w;
end SaturatingServo;
