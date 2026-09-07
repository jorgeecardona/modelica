model DCMotorThermal "A loaded DC motor that warms as it runs: the winding's heat raises its own resistance"
  Electrical.ConstantVoltage supply(V = 12);
  Electrical.Ground ground;
  Electrical.HeatingResistor winding(R_ref = 0.5, alpha = 0.0039, T_ref = 293.15);
  Electrical.Inductor coil(L = 0.05, i(start = 0, fixed = true));
  Machines.EMF converter(k = 0.5);
  Rotational.Inertia rotor(J = 0.02, phi(start = 0, fixed = true), w(start = 0, fixed = true));
  Rotational.Damper friction(d = 0.05);
  Thermal.HeatCapacitor winding_mass(C = 20, T(start = 293.15, fixed = true));
  Thermal.ThermalConductor to_ambient(G = 0.5);
  Thermal.FixedTemperature ambient(T = 293.15);
equation
  connect(supply.p, winding.p);
  connect(winding.n, coil.p);
  connect(coil.n, converter.p);
  connect(converter.n, supply.n);
  connect(supply.n, ground.p);
  connect(converter.flange, rotor.flange_a);
  connect(rotor.flange_b, friction.flange);
  connect(winding.heat_port, winding_mass.port);
  connect(winding_mass.port, to_ambient.port_a);
  connect(to_ambient.port_b, ambient.port);
end DCMotorThermal;
