"""The two bridges between the signal domain and the physical ones.

`Electrical.SignalVoltage` is an actuator: a voltage source whose voltage is a signal, so a
controller can move a circuit. `Rotational.SpeedSensor` is the matching sensor: a flange that
reports its speed as a signal and draws no torque, so a loop can be closed. These flatten and
simulate here without a corpus `.mo`, because a component model wiring a controller to a motor
trips the schematic router at depth (a signal wire past an opened `Machines.Motor`) -- a
diagram-layout gap, not a modelling one -- so the bridges earn their place by behaviour instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelica.ir.causalize import causalize
from modelica.lang import parse
from modelica.sim import flatten, simulate

LIBRARY = Path(__file__).resolve().parent.parent / "examples" / "library"


def _simulate(model_source: str, name: str, libraries: list[str], stop: float, points: int = 400):  # noqa: ANN202 - Result
    sources = [parse((LIBRARY / f"{lib}.mo").read_text(encoding="utf-8")) for lib in libraries]
    sources.append(parse(model_source))
    system = causalize(flatten(sources, name))
    return simulate(system, stop=stop, times=np.linspace(0.0, stop, points))


def test_signal_voltage_puts_its_commanded_voltage_on_an_rl_circuit() -> None:
    """A 5 V signal into a SignalVoltage across an R-L branch: the current charges to V/R."""
    result = _simulate(
        """
        model Rig
          Blocks.Constant command(k = 5);
          Electrical.SignalVoltage drive;
          Electrical.Inductor coil(L = 2, i(start = 0, fixed = true));
          Electrical.Resistor load(R = 10);
          Electrical.Ground ground;
        equation
          connect(command.y, drive.v_signal);
          connect(drive.p, coil.p);
          connect(coil.n, load.p);
          connect(load.n, drive.n);
          connect(drive.n, ground.p);
        end Rig;
        """,
        "Rig",
        ["Blocks", "Electrical"],
        stop=1.0,
    )
    time = np.asarray(result.time)
    exact = 0.5 * (1.0 - np.exp(-5.0 * time))  # V/R = 0.5, R/L = 5
    assert np.max(np.abs(np.asarray(result["coil.i"]) - exact)) < 1e-3


def test_speed_sensor_reports_the_flange_speed_and_draws_no_torque() -> None:
    """An inertia braked by a damper spins down, and the sensor reports its speed, drawing no torque."""
    result = _simulate(
        """
        model Spin
          Rotational.Inertia flywheel(J = 1, phi(start = 0, fixed = true), w(start = 10, fixed = true));
          Rotational.Damper brake(d = 2);
          Rotational.SpeedSensor tacho;
        equation
          connect(flywheel.flange_b, brake.flange);
          connect(flywheel.flange_a, tacho.flange);
        end Spin;
        """,
        "Spin",
        ["Blocks", "Rotational"],
        stop=2.0,
    )
    time = np.asarray(result.time)
    w = np.asarray(result["flywheel.w"])
    assert np.max(np.abs(w - 10.0 * np.exp(-2.0 * time))) < 1e-3
    # The sensor reports the shaft speed exactly -- that is the whole job of the component.
    assert np.max(np.abs(np.asarray(result["tacho.w.signal"]) - w)) < 1e-12


def test_a_motor_rig_spins_up_to_its_no_load_speed_through_both_bridges() -> None:
    """A 10 V command through the actuator spins a DC motor to its no-load speed V/k, its speed read back by the sensor: both bridges around a real electromechanical plant."""
    result = _simulate(
        """
        model Rig
          Blocks.Constant command(k = 10);
          Electrical.SignalVoltage drive;
          Electrical.Ground ground;
          Machines.Motor motor(Rw = 0.5, Lw = 0.05, k = 0.1, J = 0.001, coil(i(start = 0, fixed = true)), rotor(phi(start = 0, fixed = true), w(start = 0, fixed = true)));
          Rotational.SpeedSensor tacho;
        equation
          connect(command.y, drive.v_signal);
          connect(drive.p, motor.p);
          connect(motor.n, drive.n);
          connect(drive.n, ground.p);
          connect(motor.flange, tacho.flange);
        end Rig;
        """,
        "Rig",
        ["Blocks", "Electrical", "Rotational", "Machines"],
        stop=3.0,
    )
    w = np.asarray(result["motor.rotor.w"])
    assert abs(float(w[-1]) - 100.0) < 0.5  # no-load speed V/k = 10/0.1
    assert float(np.asarray(result["motor.coil.i"])[-1]) < 0.05  # current falls to ~0 with no load
    # The sensor tracks the shaft through the whole run.
    assert np.max(np.abs(np.asarray(result["tacho.w.signal"]) - w)) < 1e-12
