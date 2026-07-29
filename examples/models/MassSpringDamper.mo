model MassSpringDamper "A mass on a spring and damper, released from 1 m -- the RLC circuit's twin"
  Translational.Fixed wall(s0 = 0);
  Translational.Spring spring(c = 4, s_rel0 = 0);
  Translational.Damper damper(d = 0.4);
  Translational.Mass mass(m = 1, s(start = 1, fixed = true), v(start = 0, fixed = true));
equation
  connect(wall.flange, spring.flange_a);
  connect(wall.flange, damper.flange_a);
  connect(spring.flange_b, mass.flange_a);
  connect(damper.flange_b, mass.flange_a);
end MassSpringDamper;
