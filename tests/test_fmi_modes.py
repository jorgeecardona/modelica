"""The FMI 3.0 calling sequence: the table, the guard that reads it, and the machine it is.

Two kinds of test. The first kind pins facts from the specification -- where
``fmi3ExitInitializationMode`` lands, what is legal in Continuous-Time Mode -- so that
editing the table has to be deliberate. The second kind asks the machine about itself:
every mode reachable, no dead ends, and an instance always disposable. That second kind is
the reason the table exists before the FMI layer does, and it has already earned it (Event
Mode was in the Scheduled Execution table, where the standard has Clock Activation Mode).
"""

from __future__ import annotations

import pytest

from modelica.fmi import Interface, Lifecycle, Mode, ModeError, table

INTERFACES = list(Interface)


@pytest.mark.parametrize("interface", INTERFACES, ids=[i.name for i in INTERFACES])
def test_the_machine_has_no_dead_modes_and_no_traps(interface: Interface) -> None:
    check_modes = pytest.importorskip("modelica.verify").check_modes
    report = check_modes(interface)
    assert report.ok, str(report)


@pytest.mark.parametrize("interface", INTERFACES, ids=[i.name for i in INTERFACES])
def test_an_instance_can_always_be_freed(interface: Interface) -> None:
    """Every mode offers ``fmi3FreeInstance``, because the super-state says every mode does."""
    for mode, calls in table(interface).items():
        if mode is Mode.FREED:
            continue
        assert calls["fmi3FreeInstance"] is Mode.FREED, f"{mode} strands the instance"


def test_exiting_initialization_lands_where_the_interface_says() -> None:
    """The one real fork between the three interfaces."""
    assert (
        table(Interface.MODEL_EXCHANGE)[Mode.INITIALIZATION]["fmi3ExitInitializationMode"]
        is Mode.EVENT
    )
    assert (
        table(Interface.CO_SIMULATION)[Mode.INITIALIZATION]["fmi3ExitInitializationMode"]
        is Mode.STEP
    )
    assert (
        table(Interface.SCHEDULED_EXECUTION)[Mode.INITIALIZATION]["fmi3ExitInitializationMode"]
        is Mode.CLOCK_ACTIVATION
    )


def test_scheduled_execution_has_no_event_mode() -> None:
    """It has Clock Activation Mode instead -- the transcription slip `check_modes` found."""
    assert Mode.EVENT not in table(Interface.SCHEDULED_EXECUTION)
    assert Mode.CLOCK_ACTIVATION in table(Interface.SCHEDULED_EXECUTION)
    assert Mode.EVENT in table(Interface.MODEL_EXCHANGE)
    assert Mode.CLOCK_ACTIVATION not in table(Interface.MODEL_EXCHANGE)


def test_model_exchange_walks_to_continuous_time() -> None:
    life = Lifecycle(Interface.MODEL_EXCHANGE)
    assert life.mode is Mode.INSTANTIATED
    assert life.call("fmi3EnterInitializationMode") is Mode.INITIALIZATION
    assert life.call("fmi3ExitInitializationMode") is Mode.EVENT
    assert life.call("fmi3UpdateDiscreteStates") is Mode.EVENT
    assert life.call("fmi3EnterContinuousTimeMode") is Mode.CONTINUOUS_TIME
    assert life.call("fmi3SetTime") is Mode.CONTINUOUS_TIME
    assert life.call("fmi3EnterEventMode") is Mode.EVENT
    assert life.call("fmi3Terminate") is Mode.TERMINATED
    assert life.call("fmi3FreeInstance") is Mode.FREED


def test_a_call_in_the_wrong_mode_is_refused_with_the_sequence_that_led_there() -> None:
    """What an exported FMU turns into ``fmi3Error`` -- and the message has to be useful.

    By the time a sequence error is noticed the mistake is usually several calls back, so
    the refusal carries the history rather than only the offending call.
    """
    life = Lifecycle(Interface.MODEL_EXCHANGE)
    life.call("fmi3EnterInitializationMode")
    life.call("fmi3ExitInitializationMode")
    life.call("fmi3EnterContinuousTimeMode")
    with pytest.raises(ModeError) as raised:
        life.call("fmi3UpdateDiscreteStates")
    message = str(raised.value)
    assert "Continuous-Time Mode" in message
    assert "fmi3ExitInitializationMode" in message, "the history should be in the message"
    assert "fmi3EnterEventMode" in message, "and what is allowed instead"


def test_a_do_step_is_not_available_to_model_exchange() -> None:
    """The interfaces are different machines, not one machine with optional calls."""
    life = Lifecycle(Interface.MODEL_EXCHANGE)
    life.call("fmi3EnterInitializationMode")
    life.call("fmi3ExitInitializationMode")
    assert "fmi3DoStep" not in life.allowed
    other = Lifecycle(Interface.CO_SIMULATION)
    other.call("fmi3EnterInitializationMode")
    other.call("fmi3ExitInitializationMode")
    assert other.mode is Mode.STEP
    assert "fmi3DoStep" in other.allowed


def test_reset_returns_to_instantiated_from_anywhere() -> None:
    life = Lifecycle(Interface.CO_SIMULATION)
    life.call("fmi3EnterInitializationMode")
    life.call("fmi3ExitInitializationMode")
    life.call("fmi3DoStep")
    assert life.call("fmi3Reset") is Mode.INSTANTIATED
