model Thermoelectric "A thermoelectric generator: a hot junction between a source and a sink drives a Seebeck current into a load, and the current carries heat back the way a Peltier element does"
  parameter Real T_src(unit = "K") = 500 "Hot-source temperature";
  parameter Real T_c(unit = "K") = 300 "Cold-sink temperature";
  parameter Real G_h(unit = "W/K") = 2 "Conductance from the source to the hot junction";
  parameter Real K(unit = "W/K") = 0.5 "Conductance leaking through the module to the sink";
  parameter Real S(unit = "V/K") = 0.05 "Seebeck coefficient";
  parameter Real R_int(unit = "Ohm") = 1 "Module internal resistance";
  parameter Real R_load(unit = "Ohm") = 1 "Load resistance";
  parameter Real C(unit = "J/K") = 10 "Heat capacity of the hot junction";
  Real T_h(unit = "K", start = 300, fixed = true) "Hot-junction temperature";
  Real V_oc(unit = "V") "Open-circuit Seebeck voltage";
  Real i(unit = "A") "Current through the load";
  Real P_load(unit = "W") "Electrical power delivered to the load";
  Real Q_peltier(unit = "W") "Heat the current carries off the hot junction";
equation
  V_oc = S * (T_h - T_c);
  i = V_oc / (R_int + R_load);
  P_load = i ^ 2 * R_load;
  Q_peltier = S * T_h * i;
  C * der(T_h) = G_h * (T_src - T_h) - K * (T_h - T_c) - Q_peltier + 0.5 * R_int * i ^ 2;
end Thermoelectric;
