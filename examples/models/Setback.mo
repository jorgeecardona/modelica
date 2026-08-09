model Setback "A room on a night setback: an enumeration says which setpoint is in force"
  parameter Real k(unit = "1/h") = 0.5 "How fast the room follows its setpoint";
  parameter Real T_day(unit = "degC") = 21 "Occupied setpoint";
  parameter Real T_night(unit = "degC") = 16 "Setback setpoint";
  parameter Real switch_at(unit = "h") = 4.0 "When the setback starts";
  Thermal.Schedule schedule "Which setpoint the clock has selected";
  Real T_set(unit = "degC") "The setpoint in force";
  Real T(unit = "degC", start = 21, fixed = true) "Room temperature";
equation
  schedule = if time < switch_at then Thermal.Schedule.day else Thermal.Schedule.night;
  T_set = if schedule == Thermal.Schedule.day then T_day else T_night;
  der(T) = k * (T_set - T);
end Setback;
