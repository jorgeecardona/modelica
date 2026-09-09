model RigidBodyFlip "A torque-free rigid body about its principal axes: Euler's equations and the tennis-racket flip"
  parameter Real inertia1(unit = "kg.m2") = 1 "Moment of inertia about principal axis 1 (small)";
  parameter Real inertia2(unit = "kg.m2") = 2 "Moment of inertia about principal axis 2 (intermediate)";
  parameter Real inertia3(unit = "kg.m2") = 3 "Moment of inertia about principal axis 3 (large)";
  Real w1(unit = "rad/s", start = 0.05, fixed = true) "Angular velocity about axis 1";
  Real w2(unit = "rad/s", start = 1, fixed = true) "Angular velocity about axis 2";
  Real w3(unit = "rad/s", start = 0, fixed = true) "Angular velocity about axis 3";
equation
  inertia1 * der(w1) = (inertia2 - inertia3) * w2 * w3;
  inertia2 * der(w2) = (inertia3 - inertia1) * w3 * w1;
  inertia3 * der(w3) = (inertia1 - inertia2) * w1 * w2;
end RigidBodyFlip;
