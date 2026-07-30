model RoastingMachine "A drum coffee roaster over one batch -- the gas schedule makes time events"
  parameter Real T_amb(unit = "K") = 293.15 "Ambient temperature";
  parameter Real T_boil(unit = "K") = 373.15 "Boiling point of water";
  parameter Real C_drum(unit = "J/K") = 40000 "Drum and internals";
  parameter Real C_bean(unit = "J/K") = 20400 "12 kg of green beans";
  parameter Real G_loss(unit = "W/K") = 60 "Drum to ambient, mostly the exhaust";
  parameter Real G_bean(unit = "W/K") = 200 "Drum air to the bean pile";
  parameter Real L(unit = "J/kg") = 2.26e6 "Latent heat of vaporisation";
  parameter Real k_dry(unit = "1/(K.s)") = 1e-4 "Drying rate above boiling";
  parameter Real t_cut(unit = "s") = 480 "Gas cut to low, at first crack";
  parameter Real t_off(unit = "s") = 640 "Flame out, the batch is dropped";
  parameter Real Q_dry(unit = "W") = 28000 "Burner through drying and ramp";
  parameter Real Q_dev(unit = "W") = 10000 "Burner through development";
  Real T_drum(unit = "K", start = 473.15, fixed = true) "Drum temperature";
  Real T_bean(unit = "K", start = 293.15, fixed = true) "Bean temperature";
  Real m_w(unit = "kg", start = 1.44, fixed = true) "Water left in the beans";
  Real Q_burner(unit = "W") "Burner heat into the drum";
  Real evap(unit = "kg/s") "Evaporation rate";
equation
  Q_burner = if time < t_cut then Q_dry else (if time < t_off then Q_dev else 0);
  C_drum * der(T_drum) = Q_burner - G_loss * (T_drum - T_amb) - G_bean * (T_drum - T_bean);
  evap = if T_bean > T_boil then k_dry * m_w * (T_bean - T_boil) else 0;
  C_bean * der(T_bean) = G_bean * (T_drum - T_bean) - L * evap;
  der(m_w) = -evap;
end RoastingMachine;
