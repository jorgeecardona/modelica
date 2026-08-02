package Machines "Where the electrical domain meets the rotational one"

  model EMF "Ideal converter between current and torque, and between speed and voltage"
    parameter Real k(unit = "N.m/A") = 1 "Torque constant, equal to the back-EMF constant";
    Electrical.Pin p "Positive terminal";
    Electrical.Pin n "Negative terminal";
    Rotational.Flange flange "Shaft it drives";
    Electrical.Voltage v "Voltage across the terminals";
    Electrical.Current i "Current through the machine";
    Rotational.AngularVelocity w "Speed of the shaft";
  equation
    v = p.v - n.v;
    i = p.i;
    p.i + n.i = 0;
    w = der(flange.phi);
    v = k * w;
    flange.tau = -k * i;
  end EMF;

  model Motor "A DC motor as one component: winding, converter and rotor inside"
    parameter Real Rw(unit = "Ohm") = 0.5 "Winding resistance";
    parameter Real Lw(unit = "H") = 0.05 "Winding inductance";
    parameter Real k(unit = "N.m/A") = 0.5 "Torque constant";
    parameter Real J(unit = "kg.m2") = 0.02 "Moment of inertia of the rotor";
    Electrical.Pin p "Positive terminal";
    Electrical.Pin n "Negative terminal";
    Rotational.Flange flange "Output shaft";
    Electrical.Resistor winding(R = Rw);
    Electrical.Inductor coil(L = Lw);
    EMF converter(k = k);
    Rotational.Inertia rotor(J = J);
  equation
    connect(p, winding.p);
    connect(winding.n, coil.p);
    connect(coil.n, converter.p);
    connect(converter.n, n);
    connect(converter.flange, rotor.flange_a);
    connect(rotor.flange_b, flange);
  end Motor;

end Machines;
