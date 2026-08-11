model Thermostat "A heater with hysteresis: the regime is remembered, not re-decided"
  parameter Real T_low(unit = "degC") = 19 "Switch the heater on at or below this";
  parameter Real T_high(unit = "degC") = 21 "Switch it off at or above this";
  parameter Real T_amb(unit = "degC") = 5 "Outside, which the room leaks toward";
  parameter Real k(unit = "1/s") = 0.15 "How fast the room leaks";
  parameter Real q(unit = "K/s") = 3 "What the heater adds while it is on";
  Real T(start = 20, fixed = true, unit = "degC") "Room temperature";
  discrete Real heating(start = 0, fixed = true) "1 while the heater is on, 0 while it is off";
  discrete Real cycles(start = 0, fixed = true) "How many times the heater has come on";
equation
  der(T) = k * (T_amb - T) + q * heating;
  when T <= T_low then
    heating = 1;
    cycles = pre(cycles) + 1;
  end when;
  when T >= T_high then
    heating = 0;
  end when;
end Thermostat;
