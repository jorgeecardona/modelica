package Modelica "A curated slice of the Modelica Standard Library, under its real names.

  Not the whole MSL -- pymodelica does not bundle that -- but the components the panel's own
  examples are built from, re-homed under the real Modelica.* paths so an ordinary textbook
  model resolves and simulates without configuration. Every class here flattens and integrates;
  the domains are the physical ones, whose connectors carry a potential and a flow:
  Electrical.Analog, Mechanics.Translational and Rotational, and Thermal.HeatTransfer.
  References are fully qualified throughout, since `import` is not in the language front end yet,
  and Blocks is left out until the causal signal connector (`RealInput = input Real`) is.

  A model that reaches for a part of the MSL not here fails by name (X has no nested class Y),
  which says what is missing. The slice grows as the examples need it."

  package Units "Modelica.Units.SI: the quantity types, as unit-tagged Reals."
    package SI
      type Time = Real(unit = "s");
      type Voltage = Real(unit = "V");
      type Current = Real(unit = "A");
      type Resistance = Real(unit = "Ohm");
      type Capacitance = Real(unit = "F");
      type Inductance = Real(unit = "H");
      type Angle = Real(unit = "rad");
      type AngularVelocity = Real(unit = "rad/s");
      type Torque = Real(unit = "N.m");
      type Inertia = Real(unit = "kg.m2");
      type Position = Real(unit = "m");
      type Velocity = Real(unit = "m/s");
      type Force = Real(unit = "N");
      type Mass = Real(unit = "kg");
      type TranslationalSpringConstant = Real(unit = "N/m");
      type TranslationalDampingConstant = Real(unit = "N.s/m");
      type RotationalSpringConstant = Real(unit = "N.m/rad");
      type RotationalDampingConstant = Real(unit = "N.m.s/rad");
      type Temperature = Real(unit = "K");
      type HeatFlowRate = Real(unit = "W");
      type ThermalConductance = Real(unit = "W/K");
      type HeatCapacity = Real(unit = "J/K");
    end SI;
  end Units;

  package Electrical
    package Analog
      package Interfaces
        connector Pin "Pin of an electrical component"
          Modelica.Units.SI.Voltage v "Potential at the pin";
          flow Modelica.Units.SI.Current i "Current flowing into the pin";
        end Pin;

        connector PositivePin "Positive electrical pin"
          Modelica.Units.SI.Voltage v;
          flow Modelica.Units.SI.Current i;
        end PositivePin;

        connector NegativePin "Negative electrical pin"
          Modelica.Units.SI.Voltage v;
          flow Modelica.Units.SI.Current i;
        end NegativePin;

        partial model OnePort "Component with two pins p and n and a shared current"
          Modelica.Units.SI.Voltage v "Voltage drop between p and n";
          Modelica.Units.SI.Current i "Current from p to n";
          Modelica.Electrical.Analog.Interfaces.PositivePin p;
          Modelica.Electrical.Analog.Interfaces.NegativePin n;
        equation
          v = p.v - n.v;
          i = p.i;
          p.i + n.i = 0;
        end OnePort;
      end Interfaces;

      package Basic
        model Ground "Ground node, the reference potential"
          Modelica.Electrical.Analog.Interfaces.Pin p;
        equation
          p.v = 0;
        end Ground;

        model Resistor "Ideal linear electrical resistor"
          extends Modelica.Electrical.Analog.Interfaces.OnePort;
          parameter Modelica.Units.SI.Resistance R = 1 "Resistance";
        equation
          v = R * i;
        end Resistor;

        model Capacitor "Ideal linear electrical capacitor"
          extends Modelica.Electrical.Analog.Interfaces.OnePort;
          parameter Modelica.Units.SI.Capacitance C = 1 "Capacitance";
        equation
          i = C * der(v);
        end Capacitor;

        model Inductor "Ideal linear electrical inductor"
          extends Modelica.Electrical.Analog.Interfaces.OnePort;
          parameter Modelica.Units.SI.Inductance L = 1 "Inductance";
        equation
          v = L * der(i);
        end Inductor;
      end Basic;

      package Sources
        model ConstantVoltage "Source for constant voltage"
          extends Modelica.Electrical.Analog.Interfaces.OnePort;
          parameter Modelica.Units.SI.Voltage V = 1 "Value of constant voltage";
        equation
          v = V;
        end ConstantVoltage;

        model SineVoltage "Sine voltage source"
          extends Modelica.Electrical.Analog.Interfaces.OnePort;
          parameter Modelica.Units.SI.Voltage V = 1 "Amplitude";
          parameter Real freqHz(unit = "Hz") = 1 "Frequency";
          parameter Modelica.Units.SI.Voltage offset = 0 "Voltage offset";
        equation
          v = offset + V * sin(6.283185307179586 * freqHz * time);
        end SineVoltage;
      end Sources;
    end Analog;
  end Electrical;

  package Mechanics
    package Translational
      package Interfaces
        connector Flange "One-dimensional translational flange"
          Modelica.Units.SI.Position s "Position of the flange";
          flow Modelica.Units.SI.Force f "Force on the flange";
        end Flange;

        partial model PartialCompliant "Two flanges, an elongation and its force"
          Modelica.Units.SI.Position s_rel "Elongation, flange_b minus flange_a";
          Modelica.Units.SI.Force f "Force through the component";
          Modelica.Mechanics.Translational.Interfaces.Flange flange_a;
          Modelica.Mechanics.Translational.Interfaces.Flange flange_b;
        equation
          s_rel = flange_b.s - flange_a.s;
          flange_b.f = f;
          flange_a.f = -f;
        end PartialCompliant;
      end Interfaces;

      package Components
        model Fixed "A flange fixed in place"
          parameter Modelica.Units.SI.Position s0 = 0 "Fixed position";
          Modelica.Mechanics.Translational.Interfaces.Flange flange;
        equation
          flange.s = s0;
        end Fixed;

        model Mass "Sliding mass with inertia"
          parameter Modelica.Units.SI.Mass m = 1 "Mass";
          Modelica.Units.SI.Position s "Position";
          Modelica.Units.SI.Velocity v "Velocity";
          Modelica.Mechanics.Translational.Interfaces.Flange flange_a;
          Modelica.Mechanics.Translational.Interfaces.Flange flange_b;
        equation
          flange_a.s = s;
          flange_b.s = s;
          v = der(s);
          m * der(v) = flange_a.f + flange_b.f;
        end Mass;

        model Spring "Linear 1D translational spring"
          extends Modelica.Mechanics.Translational.Interfaces.PartialCompliant;
          parameter Modelica.Units.SI.TranslationalSpringConstant c = 1 "Spring constant";
          parameter Modelica.Units.SI.Position s_rel0 = 0 "Unstretched length";
        equation
          f = c * (s_rel - s_rel0);
        end Spring;

        model Damper "Linear 1D translational damper"
          extends Modelica.Mechanics.Translational.Interfaces.PartialCompliant;
          parameter Modelica.Units.SI.TranslationalDampingConstant d = 1 "Damping constant";
        equation
          f = d * der(s_rel);
        end Damper;

        model SpringDamper "Linear 1D translational spring and damper in parallel"
          extends Modelica.Mechanics.Translational.Interfaces.PartialCompliant;
          parameter Modelica.Units.SI.TranslationalSpringConstant c = 1 "Spring constant";
          parameter Modelica.Units.SI.TranslationalDampingConstant d = 1 "Damping constant";
          parameter Modelica.Units.SI.Position s_rel0 = 0 "Unstretched length";
        equation
          f = c * (s_rel - s_rel0) + d * der(s_rel);
        end SpringDamper;
      end Components;
    end Translational;

    package Rotational
      package Interfaces
        connector Flange "One-dimensional rotational flange"
          Modelica.Units.SI.Angle phi "Angle of the flange";
          flow Modelica.Units.SI.Torque tau "Torque on the flange";
        end Flange;
      end Interfaces;

      package Components
        model Fixed "A flange fixed at an angle"
          parameter Modelica.Units.SI.Angle phi0 = 0 "Fixed angle";
          Modelica.Mechanics.Rotational.Interfaces.Flange flange;
        equation
          flange.phi = phi0;
        end Fixed;

        model Inertia "Rotating body with moment of inertia"
          parameter Modelica.Units.SI.Inertia J = 1 "Moment of inertia";
          Modelica.Units.SI.Angle phi "Angle";
          Modelica.Units.SI.AngularVelocity w "Angular velocity";
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_a;
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_b;
        equation
          flange_a.phi = phi;
          flange_b.phi = phi;
          w = der(phi);
          J * der(w) = flange_a.tau + flange_b.tau;
        end Inertia;

        model Spring "Linear 1D rotational spring"
          parameter Modelica.Units.SI.RotationalSpringConstant c = 1 "Spring constant";
          parameter Modelica.Units.SI.Angle phi_rel0 = 0 "Unstretched angle";
          Modelica.Units.SI.Angle phi_rel "Relative angle";
          Modelica.Units.SI.Torque tau "Torque through the spring";
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_a;
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_b;
        equation
          phi_rel = flange_b.phi - flange_a.phi;
          tau = c * (phi_rel - phi_rel0);
          flange_b.tau = tau;
          flange_a.tau = -tau;
        end Spring;

        model Damper "Linear 1D rotational damper"
          parameter Modelica.Units.SI.RotationalDampingConstant d = 1 "Damping constant";
          Modelica.Units.SI.Angle phi_rel "Relative angle";
          Modelica.Units.SI.Torque tau "Torque through the damper";
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_a;
          Modelica.Mechanics.Rotational.Interfaces.Flange flange_b;
        equation
          phi_rel = flange_b.phi - flange_a.phi;
          tau = d * der(phi_rel);
          flange_b.tau = tau;
          flange_a.tau = -tau;
        end Damper;
      end Components;
    end Rotational;
  end Mechanics;

  package Thermal
    package HeatTransfer
      package Interfaces
        connector HeatPort "Thermal port: temperature shared, heat flow sums to zero"
          Modelica.Units.SI.Temperature T "Port temperature";
          flow Modelica.Units.SI.HeatFlowRate Q_flow "Heat flow into the component";
        end HeatPort;
      end Interfaces;

      package Components
        model HeatCapacitor "Lumped thermal mass storing heat"
          parameter Modelica.Units.SI.HeatCapacity C = 1 "Heat capacity";
          Modelica.Units.SI.Temperature T(start = 293.15) "Temperature";
          Modelica.Thermal.HeatTransfer.Interfaces.HeatPort port;
        equation
          T = port.T;
          C * der(T) = port.Q_flow;
        end HeatCapacitor;

        model ThermalConductor "Heat flow proportional to a temperature difference"
          parameter Modelica.Units.SI.ThermalConductance G = 1 "Conductance";
          Modelica.Units.SI.Temperature dT "Temperature drop, a minus b";
          Modelica.Units.SI.HeatFlowRate Q_flow "Heat from a to b";
          Modelica.Thermal.HeatTransfer.Interfaces.HeatPort port_a;
          Modelica.Thermal.HeatTransfer.Interfaces.HeatPort port_b;
        equation
          dT = port_a.T - port_b.T;
          Q_flow = G * dT;
          port_a.Q_flow = Q_flow;
          port_b.Q_flow = -Q_flow;
        end ThermalConductor;
      end Components;

      package Sources
        model FixedTemperature "A boundary held at a fixed temperature"
          parameter Modelica.Units.SI.Temperature T = 293.15 "Fixed temperature";
          Modelica.Thermal.HeatTransfer.Interfaces.HeatPort port;
        equation
          port.T = T;
        end FixedTemperature;
      end Sources;
    end HeatTransfer;
  end Thermal;

  package Blocks "Modelica.Blocks: causal signal blocks wired by real-valued connectors."
    package Interfaces
      connector RealInput = input Real "A real signal arriving at a block";
      connector RealOutput = output Real "A real signal leaving a block";

      partial block SISO "One real input, one real output"
        Modelica.Blocks.Interfaces.RealInput u "Input signal";
        Modelica.Blocks.Interfaces.RealOutput y "Output signal";
      end SISO;
    end Interfaces;

    package Sources
      block Constant "Holds a constant output value"
        parameter Real k = 1 "Constant output value";
        Modelica.Blocks.Interfaces.RealOutput y "Output signal";
      equation
        y = k;
      end Constant;
    end Sources;

    package Math
      block Gain "Output the input scaled by a constant"
        extends Modelica.Blocks.Interfaces.SISO;
        parameter Real k = 1 "Gain";
      equation
        y = k * u;
      end Gain;

      block Feedback "Output the difference of two inputs, u1 minus u2"
        Modelica.Blocks.Interfaces.RealInput u1 "Setpoint";
        Modelica.Blocks.Interfaces.RealInput u2 "Measurement";
        Modelica.Blocks.Interfaces.RealOutput y "Error, u1 minus u2";
      equation
        y = u1 - u2;
      end Feedback;
    end Math;

    package Continuous
      block Integrator "Output the time integral of the input"
        extends Modelica.Blocks.Interfaces.SISO(y(start = 0, fixed = true));
        parameter Real k = 1 "Integrator gain";
      equation
        der(y) = k * u;
      end Integrator;

      block FirstOrder "Output lags the input with a first-order time constant"
        extends Modelica.Blocks.Interfaces.SISO(y(start = 0, fixed = true));
        parameter Real k = 1 "Steady-state gain";
        parameter Modelica.Units.SI.Time T = 1 "Time constant";
      equation
        der(y) = (k * u - y) / T;
      end FirstOrder;
    end Continuous;
  end Blocks;

end Modelica;
