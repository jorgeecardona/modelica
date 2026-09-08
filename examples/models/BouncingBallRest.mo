model BouncingBallRest "A bouncing ball that comes to rest -- the well-posed companion to BouncingBall"
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  parameter Real e = 0.8 "Coefficient of restitution (fraction of speed kept)";
  parameter Real v_min(unit = "m/s") = 0.1 "Stick to the floor once the rebound speed would fall below this";
  Real h(unit = "m", start = 1, fixed = true) "Height above the floor";
  Real v(unit = "m/s", start = 0, fixed = true) "Vertical velocity";
  discrete Real resting(start = 0, fixed = true) "1 once the ball is at rest on the floor, 0 while it moves";
equation
  der(h) = (1 - resting) * v;
  der(v) = -(1 - resting) * g;
  when h <= 0 then
    resting = if -e * pre(v) < v_min then 1 else 0;
    reinit(v, if -e * pre(v) < v_min then 0 else -e * pre(v));
    reinit(h, 0);
  end when;
end BouncingBallRest;
