model SyncGenerator "A synchronous generator on an infinite bus: the swing equation damps to its angle"
  parameter Real M(unit = "s2") = 0.02 "Inertia, 2H over the synchronous speed";
  parameter Real D(unit = "s") = 0.1 "Damping coefficient";
  parameter Real Pm(unit = "1") = 0.8 "Mechanical power in, per unit";
  parameter Real Pmax(unit = "1") = 1.5 "Peak transferable electrical power, E*V/X";
  parameter Real delta_max(unit = "rad") = 3.141592653589793 "Synchronism limit, one pole slip";
  Real delta(unit = "rad", start = 0.2, fixed = true) "Rotor angle relative to the infinite bus";
  Real w(unit = "rad/s", start = 0, fixed = true) "Speed deviation from synchronous";
  Real Pe(unit = "1") "Electrical power delivered to the bus, per unit";
equation
  Pe = Pmax * sin(delta);
  der(delta) = w;
  M * der(w) = Pm - Pe - D * w;
  assert(delta < delta_max, "loss of synchronism: rotor angle slipped a pole");
end SyncGenerator;
