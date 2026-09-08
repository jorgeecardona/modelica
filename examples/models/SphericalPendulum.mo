model SphericalPendulum "A pendulum free to swing in 3D: a bob on a rigid rod tracing a path on a sphere"
  parameter Real g(unit = "m/s2") = 9.81 "Gravity";
  parameter Real rod(unit = "m") = 1 "Rod length, the radius of the sphere the bob moves on";
  Real r(unit = "m") "Radial coordinate of the bob, held at the rod length";
  Real theta(unit = "rad", start = 1.0, fixed = true) "Polar angle from the downward vertical";
  Real phi(unit = "rad", start = 0, fixed = true) "Azimuth about the vertical";
  Real w_theta(unit = "rad/s", start = 0, fixed = true) "Rate of the polar angle";
  Real w_phi(unit = "rad/s", start = 2, fixed = true) "Rate of the azimuth";
equation
  r = rod;
  der(theta) = w_theta;
  der(phi) = w_phi;
  der(w_theta) = sin(theta) * cos(theta) * w_phi ^ 2 - g / rod * sin(theta);
  der(w_phi) = -2 * cos(theta) / sin(theta) * w_theta * w_phi;
end SphericalPendulum;
