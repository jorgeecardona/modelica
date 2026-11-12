"""The bundled standard-library slice resolves and simulates under its real Modelica.* names.

`editors/vscode/stdlib/Modelica.mo` is a curated slice of the Modelica Standard Library shipped
with the VS Code extension, so a model that references the real names -- `Modelica.Electrical.
Analog.Basic.Resistor`, `Modelica.Mechanics.Translational.Components.Spring` -- resolves without
configuration. It is not the whole MSL and never will be here; these tests hold that what it does
ship flattens and integrates to the analytic answer, so a broken component is caught before it
reaches a user. Blocks is here too now that its causal `input Real` connector flattens: the
signal domain wires and simulates like the physical ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten, simulate

STDLIB = Path(__file__).resolve().parent.parent / "editors" / "vscode" / "stdlib" / "Modelica.mo"


def _simulate(model_source: str, name: str, stop: float, points: int = 400):  # noqa: ANN202 - Result
    sources = [parse(STDLIB.read_text(encoding="utf-8")), parse(model_source)]
    system = causalize(flatten(sources, name))
    return simulate(system, stop=stop, times=np.linspace(0.0, stop, points))


def test_electrical_rc_charges_like_one_minus_exp() -> None:
    """A resistor and a capacitor from Electrical.Analog: v_C(t) = V(1 - e^{-t/RC})."""
    result = _simulate(
        """
        model RC
          Modelica.Electrical.Analog.Basic.Resistor r(R = 100);
          Modelica.Electrical.Analog.Basic.Capacitor c(C = 0.01, v(start = 0, fixed = true));
          Modelica.Electrical.Analog.Sources.ConstantVoltage src(V = 1);
          Modelica.Electrical.Analog.Basic.Ground g;
        equation
          connect(src.p, r.p);
          connect(r.n, c.p);
          connect(c.n, src.n);
          connect(src.n, g.p);
        end RC;
        """,
        "RC",
        stop=5.0,
    )
    time = np.asarray(result.time)
    exact = 1.0 - np.exp(-time / 1.0)  # RC = 100 * 0.01 = 1 s
    assert np.max(np.abs(np.asarray(result["c.v"]) - exact)) < 1e-4


def test_translational_mass_spring_is_a_cosine() -> None:
    """A mass on a spring to a fixed wall: s(t) = s0 cos(sqrt(c/m) t)."""
    result = _simulate(
        """
        model MSD
          Modelica.Mechanics.Translational.Components.Fixed wall(s0 = 0);
          Modelica.Mechanics.Translational.Components.Spring spring(c = 4);
          Modelica.Mechanics.Translational.Components.Mass mass(m = 1, s(start = 1, fixed = true), v(start = 0, fixed = true));
        equation
          connect(wall.flange, spring.flange_a);
          connect(spring.flange_b, mass.flange_a);
        end MSD;
        """,
        "MSD",
        stop=3.0,
    )
    time = np.asarray(result.time)
    assert np.max(np.abs(np.asarray(result["mass.s"]) - np.cos(2.0 * time))) < 1e-3


def test_thermal_capacitor_cools_exponentially() -> None:
    """A hot heat capacitor through a conductor to a fixed boundary: T -> T_inf as e^{-t/(C/G)}."""
    result = _simulate(
        """
        model Th
          Modelica.Thermal.HeatTransfer.Components.HeatCapacitor body(C = 100, T(start = 400, fixed = true));
          Modelica.Thermal.HeatTransfer.Components.ThermalConductor wall(G = 5);
          Modelica.Thermal.HeatTransfer.Sources.FixedTemperature air(T = 300);
        equation
          connect(body.port, wall.port_a);
          connect(wall.port_b, air.port);
        end Th;
        """,
        "Th",
        stop=40.0,
    )
    time = np.asarray(result.time)
    exact = 300.0 + 100.0 * np.exp(-time / 20.0)  # C/G = 20 s
    assert np.max(np.abs(np.asarray(result["body.T"]) - exact)) < 1e-2


def test_rotational_inertia_spring_is_a_cosine() -> None:
    """An inertia on a torsional spring to a fixed flange: phi(t) = phi0 cos(sqrt(c/J) t)."""
    result = _simulate(
        """
        model Rot
          Modelica.Mechanics.Rotational.Components.Fixed wall(phi0 = 0);
          Modelica.Mechanics.Rotational.Components.Spring spring(c = 9);
          Modelica.Mechanics.Rotational.Components.Inertia flywheel(J = 1, phi(start = 0.5, fixed = true), w(start = 0, fixed = true));
        equation
          connect(wall.flange, spring.flange_a);
          connect(spring.flange_b, flywheel.flange_a);
        end Rot;
        """,
        "Rot",
        stop=2.0,
    )
    time = np.asarray(result.time)
    assert np.max(np.abs(np.asarray(result["flywheel.phi"]) - 0.5 * np.cos(3.0 * time))) < 1e-3


def test_rotational_inertia_and_damper_spins_down_exponentially() -> None:
    """An inertia braked by a damper to a fixed flange: J w' = -d w, so w = w0 e^{-(d/J) t}."""
    result = _simulate(
        """
        model RotDamp
          Modelica.Mechanics.Rotational.Components.Fixed wall(phi0 = 0);
          Modelica.Mechanics.Rotational.Components.Damper brake(d = 2);
          Modelica.Mechanics.Rotational.Components.Inertia flywheel(J = 1, phi(start = 0, fixed = true), w(start = 5, fixed = true));
        equation
          connect(wall.flange, brake.flange_a);
          connect(brake.flange_b, flywheel.flange_a);
        end RotDamp;
        """,
        "RotDamp",
        stop=3.0,
    )
    time = np.asarray(result.time)
    assert np.max(np.abs(np.asarray(result["flywheel.w"]) - 5.0 * np.exp(-2.0 * time))) < 1e-3


def test_electrical_lc_tank_oscillates_at_one_over_sqrt_lc() -> None:
    """An L-C tank with the capacitor charged: v_C(t) = v0 cos(t), omega = 1/sqrt(L C) = 1."""
    result = _simulate(
        """
        model LC
          Modelica.Electrical.Analog.Basic.Capacitor cap(C = 1, v(start = 1, fixed = true));
          Modelica.Electrical.Analog.Basic.Inductor ind(L = 1, i(start = 0, fixed = true));
          Modelica.Electrical.Analog.Basic.Ground g;
        equation
          connect(cap.p, ind.p);
          connect(cap.n, ind.n);
          connect(cap.n, g.p);
        end LC;
        """,
        "LC",
        stop=6.0,
    )
    time = np.asarray(result.time)
    assert np.max(np.abs(np.asarray(result["cap.v"]) - np.cos(time))) < 1e-3


def test_translational_spring_damper_settles() -> None:
    """A mass on a spring-damper starts displaced and, being damped, settles to rest at zero."""
    result = _simulate(
        """
        model MSD
          Modelica.Mechanics.Translational.Components.Fixed wall(s0 = 0);
          Modelica.Mechanics.Translational.Components.SpringDamper sd(c = 100, d = 4, s_rel0 = 0);
          Modelica.Mechanics.Translational.Components.Mass mass(m = 1, s(start = 1, fixed = true), v(start = 0, fixed = true));
        equation
          connect(wall.flange, sd.flange_a);
          connect(sd.flange_b, mass.flange_a);
        end MSD;
        """,
        "MSD",
        stop=6.0,
    )
    s = np.asarray(result["mass.s"])
    assert np.max(np.abs(s)) < 1.001  # the spring pulls it back, never past the start amplitude
    assert abs(s[-1]) < 0.02  # damped, so it has settled at rest by the end


def test_blocks_first_order_follows_a_step_like_one_minus_exp() -> None:
    """A constant into a first-order block, wired by causal connectors: y follows 1 - e^{-t/T}."""
    result = _simulate(
        """
        model Step
          Modelica.Blocks.Sources.Constant setpoint(k = 1);
          Modelica.Blocks.Continuous.FirstOrder plant(k = 1, T = 2);
        equation
          connect(setpoint.y, plant.u);
        end Step;
        """,
        "Step",
        stop=6.0,
    )
    t = np.asarray(result.time)
    y = np.asarray(result["plant.y"])
    assert np.max(np.abs(y - (1.0 - np.exp(-t / 2.0)))) < 1e-3


def test_blocks_proportional_loop_reaches_its_offset_steady_state() -> None:
    """A closed loop (setpoint, feedback, gain, plant) settles to k/(1+k), the P-control offset."""
    result = _simulate(
        """
        model Loop
          Modelica.Blocks.Sources.Constant setpoint(k = 1);
          Modelica.Blocks.Math.Feedback error;
          Modelica.Blocks.Math.Gain controller(k = 5);
          Modelica.Blocks.Continuous.FirstOrder plant(k = 1, T = 1);
        equation
          connect(setpoint.y, error.u1);
          connect(plant.y, error.u2);
          connect(error.y, controller.u);
          connect(controller.y, plant.u);
        end Loop;
        """,
        "Loop",
        stop=5.0,
    )
    assert abs(float(np.asarray(result["plant.y"])[-1]) - 5.0 / 6.0) < 1e-4


def test_every_shipped_domain_instantiates() -> None:
    """One class from each domain flattens, so the slice's five packages all resolve."""
    probes = {
        "Electrical": "Modelica.Electrical.Analog.Basic.Inductor e;",
        "Translational": "Modelica.Mechanics.Translational.Components.SpringDamper t;",
        "Rotational": "Modelica.Mechanics.Rotational.Components.Damper r;",
        "Thermal": "Modelica.Thermal.HeatTransfer.Components.ThermalConductor h;",
        "Blocks": "Modelica.Blocks.Math.Gain b;",
    }
    lib = parse(STDLIB.read_text(encoding="utf-8"))
    for domain, decl in probes.items():
        flat = flatten([lib, parse(f"model P {decl} equation end P;")], "P")
        assert flat.variables, f"{domain} instantiated nothing"


BUNDLED = Path(__file__).resolve().parent.parent / "src" / "modelica" / "stdlib" / "Modelica.mo"

RC = """
model RC
  Modelica.Electrical.Analog.Basic.Resistor r(R=1000);
  Modelica.Electrical.Analog.Basic.Capacitor c(C=1e-3, v(start=1));
  Modelica.Electrical.Analog.Basic.Ground g;
equation
  connect(r.p, c.p);
  connect(r.n, c.n);
  connect(c.n, g.p);
end RC;
"""


def test_a_model_on_msl_names_flattens_with_no_library_given() -> None:
    """`flatten` adds the bundled ``Modelica`` package: the RC discharge is ``exp(-t / RC)``."""
    system = causalize(flatten([parse(RC, "RC.mo")], "RC"))
    result = simulate(system, stop=1.0, times=np.linspace(0.0, 1.0, 11))
    np.testing.assert_allclose(result["c.v"], np.exp(-np.linspace(0.0, 1.0, 11)), rtol=1e-4)


def test_a_users_own_modelica_package_wins_over_the_bundled_one() -> None:
    """A program that defines ``Modelica`` resolves names in it, not in the bundled slice."""
    own = """
package Modelica
  package Electrical
    package Analog
      package Basic
        model Source
          Real v(start = 2);
        equation
          der(v) = -v;
        end Source;
      end Basic;
    end Analog;
  end Electrical;
end Modelica;
model Use
  Modelica.Electrical.Analog.Basic.Source s;
end Use;
"""
    flat = flatten([parse(own, "Own.mo")], "Use")
    assert [v.name for v in flat.variables if not v.name.startswith("der")] == ["s.v"]


def test_the_wheel_and_the_editor_ship_the_same_slice() -> None:
    """``editors/vscode/stdlib`` is a copy of the packaged file; the two must not drift."""
    assert BUNDLED.read_text(encoding="utf-8") == STDLIB.read_text(encoding="utf-8")
