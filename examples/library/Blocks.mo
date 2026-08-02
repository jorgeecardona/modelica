package Blocks "Minimal causal signal blocks, after Modelica.Blocks"

  connector InPort "A signal arriving at a block"
    input Real signal "Value carried on the wire";
  end InPort;

  connector OutPort "A signal leaving a block"
    output Real signal "Value carried on the wire";
  end OutPort;

  partial block SISO "One signal in, one signal out"
    InPort u "Input signal";
    OutPort y "Output signal";
  end SISO;

  block Constant "Holds one value for the whole run"
    parameter Real k = 1 "The value";
    OutPort y "Output signal";
  equation
    y.signal = k;
  end Constant;

  block Gain "Multiplies the input by a constant"
    extends SISO;
    parameter Real k = 1 "Gain";
  equation
    y.signal = k * u.signal;
  end Gain;

  block Feedback "Subtracts the measurement from the setpoint"
    InPort u1 "Setpoint";
    InPort u2 "Measurement";
    OutPort y "Error";
  equation
    y.signal = u1.signal - u2.signal;
  end Feedback;

  block PI "Proportional and integral controller -- introduces one state, x"
    extends SISO;
    parameter Real k = 1 "Proportional gain";
    parameter Real T(unit = "s") = 1 "Integral time constant";
    Real x "The integrator state, in units of the input";
  equation
    der(x) = u.signal / T;
    y.signal = k * (x + u.signal);
  end PI;

  block FirstOrder "A plant that lags its input -- introduces one state, y.signal"
    extends SISO;
    parameter Real k = 1 "Steady-state gain";
    parameter Real T(unit = "s") = 1 "Time constant";
  equation
    T * der(y.signal) = k * u.signal - y.signal;
  end FirstOrder;

  block Limiter "Holds the signal between two values"
    extends SISO;
    parameter Real minimum = -1 "Lower limit";
    parameter Real maximum = 1 "Upper limit";
  equation
    y.signal = if u.signal < minimum then minimum elseif u.signal > maximum then maximum else u.signal;
  end Limiter;

end Blocks;
