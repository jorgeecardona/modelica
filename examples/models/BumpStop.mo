model BumpStop "A mass on a spring meeting an elastic stop -- the contact law is a function"
  Translational.Fixed wall(s0 = 0);
  Translational.Spring spring(c = 4, s_rel0 = 0);
  Translational.Stop stop(k = 100, gap = 0.5);
  Translational.Mass mass(m = 1, s(start = -1, fixed = true), v(start = 0, fixed = true));
  Real E(unit = "J") "Total energy -- constant, because nothing here dissipates";
equation
  connect(wall.flange, spring.flange_a);
  connect(wall.flange, stop.flange_a);
  connect(spring.flange_b, mass.flange_a);
  connect(stop.flange_b, mass.flange_a);
  E = 0.5 * mass.m * mass.v ^ 2 + 0.5 * spring.c * spring.s_rel ^ 2 + Translational.contactEnergy(stop.s_rel, stop.gap, stop.k);
end BumpStop;
