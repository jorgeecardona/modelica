package Electrical "Minimal analog electrical components, after Modelica.Electrical.Analog"

  type Voltage = Real(unit = "V");
  type Current = Real(unit = "A");
  type Resistance = Real(unit = "Ohm");
  type Capacitance = Real(unit = "F");
  type Inductance = Real(unit = "H");

  connector Pin "Electrical pin: voltage is shared, current sums to zero"
    Voltage v "Potential at the pin";
    flow Current i "Current flowing into the pin";
  end Pin;

  partial model OnePort "A component with two pins carrying the same current"
    Pin p "Positive pin" annotation(Placement(transformation(extent = {{-94.0, -4.0}, {-86.0, 4.0}})));
    Pin n "Negative pin" annotation(Placement(transformation(extent = {{86.0, -4.0}, {94.0, 4.0}})));
    Voltage v "Voltage drop from p to n";
    Current i "Current flowing from p to n";
  equation
    v = p.v - n.v;
    i = p.i;
    p.i + n.i = 0;
  end OnePort;

  model Resistor "Ohm's law: v = R*i"
    extends OnePort;
    parameter Resistance R = 1 "Resistance";
  equation
    v = R * i;
    annotation(Icon(graphics = {Line(points = {{-90, 0}, {-52, 0}}), Line(points = {{-52, 0}, {-42, 22}, {-22, -22}, {-2, 22}, {18, -22}, {38, 22}, {48, 0}, {52, 0}}), Line(points = {{52, 0}, {90, 0}}), Ellipse(extent = {{-94.0, 4.0}, {-86.0, -4.0}}, fillPattern = FillPattern.Solid), Ellipse(extent = {{86, 4}, {94, -4}}, fillPattern = FillPattern.Solid), Text(extent = {{-100.0, 62.0}, {100.0, 40.0}}, textString = "%name")}));
  end Resistor;

  model Capacitor "i = C*der(v) -- introduces one state, v"
    extends OnePort;
    parameter Capacitance C = 1 "Capacitance";
  equation
    i = C * der(v);
    annotation(Icon(graphics = {Line(points = {{-90, 0}, {-12, 0}}), Line(points = {{12, 0}, {90, 0}}), Line(points = {{-12, 34}, {-12, -34}}), Line(points = {{12, 34}, {12, -34}}), Ellipse(extent = {{-94.0, 4.0}, {-86.0, -4.0}}, fillPattern = FillPattern.Solid), Ellipse(extent = {{86, 4}, {94, -4}}, fillPattern = FillPattern.Solid), Text(extent = {{-100.0, 62.0}, {100.0, 40.0}}, textString = "%name")}));
  end Capacitor;

  model Inductor "v = L*der(i) -- introduces one state, i"
    extends OnePort;
    parameter Inductance L = 1 "Inductance";
  equation
    v = L * der(i);
    annotation(Icon(graphics = {Line(points = {{-90, 0}, {-60, 0}}), Line(points = {{-60, 0}, {-60, 14}, {-45, 22}, {-30, 14}, {-30, 0}}), Line(points = {{-30, 0}, {-30, 14}, {-15, 22}, {0, 14}, {0, 0}}), Line(points = {{0, 0}, {0, 14}, {15, 22}, {30, 14}, {30, 0}}), Line(points = {{30, 0}, {30, 14}, {45, 22}, {60, 14}, {60, 0}}), Line(points = {{60, 0}, {90, 0}}), Ellipse(extent = {{-94.0, 4.0}, {-86.0, -4.0}}, fillPattern = FillPattern.Solid), Ellipse(extent = {{86, 4}, {94, -4}}, fillPattern = FillPattern.Solid), Text(extent = {{-100.0, 62.0}, {100.0, 40.0}}, textString = "%name")}));
  end Inductor;

  model ConstantVoltage "Ideal source holding a fixed voltage"
    extends OnePort;
    parameter Voltage V = 1 "Source voltage";
  equation
    v = V;
    annotation(Icon(graphics = {Line(points = {{-90, 0}, {-30, 0}}), Line(points = {{30, 0}, {90, 0}}), Ellipse(extent = {{-30, 30}, {30, -30}}), Line(points = {{-16, 0}, {-4, 0}}), Line(points = {{-10, -6}, {-10, 6}}), Line(points = {{4, 0}, {16, 0}}), Ellipse(extent = {{-94.0, 4.0}, {-86.0, -4.0}}, fillPattern = FillPattern.Solid), Ellipse(extent = {{86, 4}, {94, -4}}, fillPattern = FillPattern.Solid), Text(extent = {{-100.0, 62.0}, {100.0, 40.0}}, textString = "%name")}));
  end ConstantVoltage;

  model Ground "Reference potential; every circuit needs exactly one"
    Pin p annotation(Placement(transformation(extent = {{-4.0, 86.0}, {4.0, 94.0}})));
  equation
    p.v = 0;
    annotation(Icon(graphics = {Line(points = {{0, 90}, {0, 20}}), Line(points = {{-46, 20}, {46, 20}}), Line(points = {{-28, 2}, {28, 2}}), Line(points = {{-11, -16}, {11, -16}}), Ellipse(extent = {{-4, 94}, {4, 86}}, fillPattern = FillPattern.Solid), Text(extent = {{-100, -30}, {100, -52}}, textString = "%name")}));
  end Ground;

end Electrical;
