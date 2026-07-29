model RCCircuit "A capacitor charging through a resistor from a 1 V source"
  Electrical.ConstantVoltage source(V = 1);
  Electrical.Resistor r(R = 100);
  Electrical.Capacitor c(C = 0.01, v(start = 0, fixed = true));
  Electrical.Ground ground;
equation
  connect(source.p, r.p);
  connect(r.n, c.p);
  connect(c.n, source.n);
  connect(source.n, ground.p);
end RCCircuit;
