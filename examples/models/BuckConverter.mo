model BuckConverter "An averaged buck converter under PI voltage control: step 24 V down to a 12 V setpoint"
  parameter Real Vin(unit = "V") = 24 "Input voltage";
  parameter Real L(unit = "H") = 0.001 "Filter inductance";
  parameter Real C(unit = "F") = 0.001 "Filter capacitance";
  parameter Real Rload(unit = "Ohm") = 6 "Load resistance";
  parameter Real vref(unit = "V") = 12 "Output voltage setpoint";
  parameter Real Kp(unit = "1/V") = 0.05 "Proportional gain";
  parameter Real Ki(unit = "1/(V.s)") = 5 "Integral gain";
  Real i_ind(unit = "A", start = 0, fixed = true) "Inductor current";
  Real v_out(unit = "V", start = 0, fixed = true) "Output voltage across the capacitor";
  Real x(unit = "V.s", start = 0, fixed = true) "Integral of the voltage error";
  Real e(unit = "V") "Voltage error, setpoint minus output";
  Real D(unit = "1") "Duty cycle the controller commands";
equation
  e = vref - v_out;
  D = Kp * e + Ki * x;
  der(x) = e;
  L * der(i_ind) = D * Vin - v_out;
  C * der(v_out) = i_ind - v_out / Rload;
  assert(D >= 0, "duty cycle went negative: the buck cannot invert its output");
  assert(D <= 1, "duty cycle saturated: the buck cannot boost above the input");
end BuckConverter;
