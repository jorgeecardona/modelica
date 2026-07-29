model RLCCircuit "A series RLC circuit stepped with 1 V -- damped oscillation, two states"
  Electrical.ConstantVoltage source(V = 1);
  Electrical.Resistor r(R = 1);
  Electrical.Inductor l(L = 1, i(start = 0, fixed = true));
  Electrical.Capacitor c(C = 1, v(start = 0, fixed = true));
  Electrical.Ground ground;
equation
  connect(source.p, r.p);
  connect(r.n, l.p);
  connect(l.n, c.p);
  connect(c.n, source.n);
  connect(source.n, ground.p);
end RLCCircuit;
