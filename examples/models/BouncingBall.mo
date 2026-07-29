model BouncingBall "A ball dropped onto the floor -- one state event per bounce"
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  parameter Real e = 0.8 "Coefficient of restitution (fraction of speed kept)";
  Real h(unit = "m", start = 1, fixed = true) "Height above the floor";
  Real v(unit = "m/s", start = 0, fixed = true) "Vertical velocity";
equation
  der(h) = v;
  der(v) = -g;
  when h <= 0 then
    reinit(v, -e * pre(v));
  end when;
end BouncingBall;
