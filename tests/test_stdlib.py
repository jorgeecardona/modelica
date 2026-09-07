"""The bundled standard-library slice resolves and simulates under its real Modelica.* names.

`editors/vscode/stdlib/Modelica.mo` is a curated slice of the Modelica Standard Library shipped
with the VS Code extension, so a model that references the real names -- `Modelica.Electrical.
Analog.Basic.Resistor`, `Modelica.Mechanics.Translational.Components.Spring` -- resolves without
configuration. It is not the whole MSL and never will be here; these tests hold that what it does
ship flattens and integrates to the analytic answer, so a broken component is caught before it
reaches a user. Blocks is deliberately absent (its causal `input Real` connector is a front-end
gap), which is asserted too, so the slice cannot quietly regrow a domain that does not work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

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


def test_blocks_is_absent_until_the_causal_connector_lands() -> None:
    """Blocks is left out on purpose; a model that reaches for it fails by name, not silently."""
    from modelica.ir.flatten import FlattenError  # noqa: PLC0415 - the error this test is about

    source = "model B Modelica.Blocks.Sources.Constant c(k = 1); equation end B;"
    with pytest.raises(FlattenError, match="Blocks"):
        causalize(flatten([parse(STDLIB.read_text(encoding="utf-8")), parse(source)], "B"))


def test_every_shipped_domain_instantiates() -> None:
    """One class from each domain flattens, so the slice's four packages all resolve."""
    probes = {
        "Electrical": "Modelica.Electrical.Analog.Basic.Inductor e;",
        "Translational": "Modelica.Mechanics.Translational.Components.SpringDamper t;",
        "Rotational": "Modelica.Mechanics.Rotational.Components.Damper r;",
        "Thermal": "Modelica.Thermal.HeatTransfer.Components.ThermalConductor h;",
    }
    lib = parse(STDLIB.read_text(encoding="utf-8"))
    for domain, decl in probes.items():
        flat = flatten([lib, parse(f"model P {decl} equation end P;")], "P")
        assert flat.variables, f"{domain} instantiated nothing"
