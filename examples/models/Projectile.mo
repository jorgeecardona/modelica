model Projectile "A ball thrown straight up, its rise cut short and its fall made slow by air drag"
  parameter Real m(unit = "kg") = 0.145 "Mass of the ball";
  parameter Real g(unit = "m/s2") = 9.81 "Gravitational acceleration";
  parameter Real cd(unit = "kg/m") = 0.0013 "Quadratic-drag coefficient, 0.5*rho*Cd*A";
  parameter Boolean drag = true "Whether air drag acts; false is the vacuum shot";
  Real y(unit = "m", start = 0, fixed = true) "Height above the launch point";
  Real vy(unit = "m/s", start = 30, fixed = true) "Vertical velocity";
equation
  der(y) = vy;
  if drag then
    m * der(vy) = -m * g - cd * vy * abs(vy);
  else
    m * der(vy) = -m * g;
  end if;
end Projectile;
