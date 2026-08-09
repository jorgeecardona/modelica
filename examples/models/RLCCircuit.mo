model RLCCircuit "A series RLC circuit stepped with 1 V -- damped oscillation, two states"
  import E = Electrical;
  E.ConstantVoltage source(V = 1) annotation(Placement(transformation(extent = {{-92, -18}, {-56, 18}}, rotation = -90)));
  E.Resistor r(R = 1) annotation(Placement(transformation(extent = {{-38, 42}, {-2, 78}})));
  E.Inductor l(L = 1, i(start = 0, fixed = true)) annotation(Placement(transformation(extent = {{16, 42}, {52, 78}})));
  E.Capacitor c(C = 1, v(start = 0, fixed = true)) annotation(Placement(transformation(extent = {{56, -18}, {92, 18}}, rotation = -90)));
  E.Ground ground annotation(Placement(transformation(extent = {{-18, -96}, {18, -60}})));
equation
  connect(source.p, r.p);
  connect(r.n, l.p);
  connect(l.n, c.p);
  connect(c.n, source.n);
  connect(source.n, ground.p);
end RLCCircuit;
