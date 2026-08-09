package Translational "Minimal 1-D translational mechanics, after Modelica.Mechanics.Translational"

  type Position = Real(unit = "m");
  type Velocity = Real(unit = "m/s");
  type Acceleration = Real(unit = "m/s2");
  type Force = Real(unit = "N");

  connector Flange "Mechanical flange: position is shared, force sums to zero"
    Position s "Absolute position";
    flow Force f "Force exerted on the flange";
  end Flange;

  partial model Compliant "Two flanges with a force between them (spring, damper, ...)"
    Flange flange_a;
    Flange flange_b;
    Position s_rel "flange_b.s - flange_a.s";
    Force f "Force from flange_a to flange_b";
  equation
    s_rel = flange_b.s - flange_a.s;
    flange_b.f = f;
    flange_a.f = -f;
  end Compliant;

  model Mass "Newton's second law -- introduces two states, s and v"
    parameter Real m(unit = "kg") = 1 "Mass";
    Flange flange_a;
    Flange flange_b;
    Position s "Absolute position of the centre";
    Velocity v;
    Acceleration a;
  equation
    flange_a.s = s;
    flange_b.s = s;
    v = der(s);
    a = der(v);
    m * a = flange_a.f + flange_b.f;
  end Mass;

  model Spring "Linear spring: f = c*(s_rel - s_rel0)"
    extends Compliant;
    parameter Real c(unit = "N/m") = 1 "Stiffness";
    parameter Position s_rel0 = 0 "Unstretched length";
  equation
    f = c * (s_rel - s_rel0);
  end Spring;

  model Damper "Linear viscous damper: f = d*der(s_rel)"
    extends Compliant;
    parameter Real d(unit = "N.s/m") = 1 "Damping constant";
  equation
    f = d * der(s_rel);
  end Damper;

  function contactForce "Force from a one-sided elastic stop, zero until s_rel passes gap"
    input Real s_rel(unit = "m");
    input Real gap(unit = "m");
    input Real k(unit = "N/m");
    output Real f(unit = "N");
  algorithm
    f := 0.0;
    if s_rel > gap then
      f := k * (s_rel - gap);
    end if;
  end contactForce;

  function contactEnergy "Strain energy stored in that stop -- the integral of contactForce"
    input Real s_rel(unit = "m");
    input Real gap(unit = "m");
    input Real k(unit = "N/m");
    output Real u(unit = "J");
  algorithm
    u := 0.0;
    if s_rel > gap then
      u := 0.5 * k * (s_rel - gap) ^ 2;
    end if;
  end contactEnergy;

  model Stop "One-sided elastic contact, after Modelica.Mechanics.Translational.ElastoGap"
    extends Compliant;
    parameter Real k(unit = "N/m") = 100 "Contact stiffness";
    parameter Position gap = 0.5 "Free travel before the stop is met";
  equation
    f = contactForce(s_rel, gap, k);
  end Stop;

  model Environment "One gravity, shared by every body in the scene"
    parameter Acceleration g = 9.81 "Gravity, positive downward";
  end Environment;

  model FallingBody "A point mass in free fall, asking the scene how hard gravity pulls"
    outer Environment environment;
    Position s "Height above the datum";
    Velocity v;
  equation
    v = der(s);
    der(v) = -environment.g;
  end FallingBody;

  model Fixed "Immovable ground point"
    parameter Position s0 = 0 "Position it is fixed at";
    Flange flange;
  equation
    flange.s = s0;
  end Fixed;

end Translational;
