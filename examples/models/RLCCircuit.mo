model RLCCircuit "A series RLC circuit stepped with 1 V -- damped oscillation, two states"
  import E = Electrical;
  E.ConstantVoltage source(V = 1) annotation(Placement(transformation(extent = {{-18, -64}, {18, -28}})));
  E.Resistor r(R = 1) annotation(Placement(transformation(extent = {{-80, 16}, {-44, 52}})));
  E.Inductor l(L = 1, i(start = 0, fixed = true)) annotation(Placement(transformation(extent = {{-18, 16}, {18, 52}})));
  E.Capacitor c(C = 1, v(start = 0, fixed = true)) annotation(Placement(transformation(extent = {{44, 16}, {80, 52}})));
  E.Ground ground annotation(Placement(transformation(extent = {{3, -114}, {39, -78}})));
equation
  connect(source.p, r.p);
  connect(r.n, l.p);
  connect(l.n, c.p);
  connect(c.n, source.n);
  connect(source.n, ground.p);
end RLCCircuit;
