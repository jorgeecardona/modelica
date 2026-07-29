model TwoBodyHeat "Two thermal masses equilibrating through a conductor -- energy is conserved"
  Thermal.HeatCapacitor hot(C = 100, T(start = 400, fixed = true));
  Thermal.HeatCapacitor cold(C = 200, T(start = 300, fixed = true));
  Thermal.ThermalConductor conductor(G = 5);
equation
  connect(hot.port, conductor.port_a);
  connect(conductor.port_b, cold.port);
end TwoBodyHeat;
