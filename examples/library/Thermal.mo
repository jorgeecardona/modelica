package Thermal "Minimal lumped thermal components, after Modelica.Thermal.HeatTransfer"

  type Temperature = Real(unit = "K");
  type HeatFlowRate = Real(unit = "W");

  connector HeatPort "Thermal port: temperature is shared, heat flow sums to zero"
    Temperature T "Port temperature";
    flow HeatFlowRate Q_flow "Heat flowing into the port";
  end HeatPort;

  model HeatCapacitor "A lumped thermal mass -- introduces one state, T"
    parameter Real C(unit = "J/K") = 1 "Heat capacity";
    Temperature T "Temperature of the mass";
    HeatPort port;
  equation
    port.T = T;
    C * der(T) = port.Q_flow;
  end HeatCapacitor;

  model ThermalConductor "Fourier conduction: Q = G*(T_a - T_b)"
    parameter Real G(unit = "W/K") = 1 "Thermal conductance";
    HeatPort port_a;
    HeatPort port_b;
  equation
    port_a.Q_flow = G * (port_a.T - port_b.T);
    port_a.Q_flow + port_b.Q_flow = 0;
  end ThermalConductor;

  model FixedTemperature "Ideal reservoir holding a fixed temperature"
    parameter Temperature T = 293.15;
    HeatPort port;
  equation
    port.T = T;
  end FixedTemperature;

end Thermal;
