model RC "A charged capacitor discharging through a resistor, on MSL names"
  Modelica.Electrical.Analog.Basic.Resistor r(R = 1000);
  Modelica.Electrical.Analog.Basic.Capacitor c(C = 1e-3, v(start = 1));
  Modelica.Electrical.Analog.Basic.Ground g;
equation
  connect(r.p, c.p);
  connect(r.n, c.n);
  connect(c.n, g.p);
end RC;
