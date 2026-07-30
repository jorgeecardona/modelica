model RoastingMachine "A Loring S15 coffee roaster over one batch -- the gas schedule makes time events"
  parameter Real T_amb(unit = "K") = 293.15 "The roastery, 20 degC";
  parameter Real C_mach(unit = "J/K") = 600000.0 "Machine hot mass: steel, lining, loop air";
  parameter Real C_bean(unit = "J/K") = 30000.0 "15 kg of green coffee, drying folded in";
  parameter Real UA(unit = "W/K") = 222 "Air-to-bean coupling over the whole bed";
  parameter Real G_loss(unit = "W/K") = 164 "Loop to ambient: the bleed and the shell";
  parameter Real Q_max(unit = "W") = 125000.0 "Burner at 100 percent";
  Real T_mach(unit = "K", start = 424.45, fixed = true) "Machine temperature";
  Real T_bean(unit = "K", start = 293.15, fixed = true) "Bean temperature";
  Real bean(unit = "degC") "Bean temperature in degC -- the number the roaster reads";
  Real burner "Gas modulation, per cent -- the machine's own signal";
  Real Q_burner(unit = "W") "Burner heat into the loop";
equation
  burner = if time < 66.7 then 20 elseif time < 93.0 then 27 elseif time < 526.9 then 85 elseif time < 618.0 then 79 elseif time < 688.5 then 70 elseif time < 714.6 then 61 elseif time < 754.4 then 50 elseif time < 780.8 then 37 else 20;
  Q_burner = burner / 100 * Q_max;
  C_mach * der(T_mach) = Q_burner - UA * (T_mach - T_bean) - G_loss * (T_mach - T_amb);
  C_bean * der(T_bean) = UA * (T_mach - T_bean);
  bean = T_bean - 273.15;
end RoastingMachine;
