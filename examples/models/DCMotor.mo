model DCMotor "A DC motor started from rest on 12 V, running up against viscous friction"
  Electrical.ConstantVoltage supply(V = 12);
  Electrical.Ground ground;
  Machines.Motor motor(Rw = 0.5, Lw = 0.05, k = 0.5, J = 0.02, coil(i(start = 0, fixed = true)), rotor(phi(start = 0, fixed = true), w(start = 0, fixed = true)));
  Rotational.Damper friction(d = 0.002);
equation
  connect(supply.p, motor.p);
  connect(motor.n, supply.n);
  connect(supply.n, ground.p);
  connect(motor.flange, friction.flange);
end DCMotor;
