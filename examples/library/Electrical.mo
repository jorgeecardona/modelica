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
    Pin p "Positive pin";
    Pin n "Negative pin";
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
  end Resistor;

  model Capacitor "i = C*der(v) -- introduces one state, v"
    extends OnePort;
    parameter Capacitance C = 1 "Capacitance";
  equation
    i = C * der(v);
  end Capacitor;

  model Inductor "v = L*der(i) -- introduces one state, i"
    extends OnePort;
    parameter Inductance L = 1 "Inductance";
  equation
    v = L * der(i);
  end Inductor;

  model ConstantVoltage "Ideal source holding a fixed voltage"
    extends OnePort;
    parameter Voltage V = 1 "Source voltage";
  equation
    v = V;
  end ConstantVoltage;

  model Ground "Reference potential; every circuit needs exactly one"
    Pin p;
  equation
    p.v = 0;
  end Ground;

end Electrical;
