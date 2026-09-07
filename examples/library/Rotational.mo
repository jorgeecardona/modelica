package Rotational "Minimal 1-D rotational mechanics, after Modelica.Mechanics.Rotational"

  type Angle = Real(unit = "rad");
  type AngularVelocity = Real(unit = "rad/s");
  type AngularAcceleration = Real(unit = "rad/s2");
  type Torque = Real(unit = "N.m");

  connector Flange "Rotational flange: angle is shared, torque sums to zero"
    Angle phi "Absolute angle";
    flow Torque tau "Torque exerted on the flange";
  end Flange;

  model Inertia "Newton's second law for rotation -- introduces two states, phi and w"
    parameter Real J(unit = "kg.m2") = 1 "Moment of inertia";
    Flange flange_a "Left flange";
    Flange flange_b "Right flange";
    Angle phi "Absolute angle of the body";
    AngularVelocity w;
    AngularAcceleration a;
  equation
    flange_a.phi = phi;
    flange_b.phi = phi;
    w = der(phi);
    a = der(w);
    J * a = flange_a.tau + flange_b.tau;
  end Inertia;

  model Damper "Viscous friction against a fixed housing"
    parameter Real d(unit = "N.m.s/rad") = 0.1 "Damping constant";
    Flange flange "Flange the friction acts on";
  equation
    flange.tau = d * der(flange.phi);
  end Damper;

  model Fixed "Immovable mounting point"
    parameter Angle phi0 = 0 "Angle it is held at";
    Flange flange;
  equation
    flange.phi = phi0;
  end Fixed;

  model SpeedSensor "Reports a flange's speed as a signal, drawing no torque: the sensor a speed loop reads"
    Flange flange "Flange whose speed is measured";
    Blocks.OutPort w "Measured angular velocity";
  equation
    w.signal = der(flange.phi);
    flange.tau = 0;
  end SpeedSensor;

end Rotational;
