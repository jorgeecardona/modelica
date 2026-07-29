"""The FMI 3.0 calling sequence, as one table.

FMI does not describe its interface as a list of functions. It describes a **state machine**
-- Instantiated, Configuration Mode, Initialization Mode, Event Mode, Continuous-Time Mode,
Step Mode, Terminated -- and says which ``fmi3*`` calls are legal in each of them and where
each one leaves you. Most of what goes wrong between two tools that both "support FMI" is a
sequence error: every individual call is fine, and the order is not.

That is a bad class of bug to find by testing, because a test only exercises the orders
somebody thought to write down. So the machine is written here, once, before the code that
obeys it, and read three times:

1. an FMU this project exports checks its own entry points against it and returns
   ``fmi3Error`` instead of corrupting itself;
2. an importer drives it, so it cannot emit a sequence no FMU is required to accept;
3. :func:`modelica.verify.check_modes` enumerates every sequence it allows and proves the
   machine has no dead ends, that every mode is reachable, and that an instance can always
   still be freed.

One table, three readers, nothing to keep in sync.

Modelled from the FMI 3.0 specification's own state tables (``2_3_common_states.adoc``,
``3_2_model_exchange_api.adoc``, ``4_2_co-simulation_api.adoc``). Deliberately *not*
modelled: the ``fmi3Get``/``fmi3Set`` distinction between variable causalities and
variabilities, which is a per-variable rule rather than a per-mode one, and clocked
variables beyond the Clock Activation Mode itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Interface", "Lifecycle", "Mode", "ModeError", "table"]


class Mode(StrEnum):
    """A state of an FMU instance, spelled as the specification spells it."""

    INSTANTIATED = "Instantiated"
    CONFIGURATION = "Configuration Mode"
    RECONFIGURATION = "Reconfiguration Mode"
    INITIALIZATION = "Initialization Mode"
    EVENT = "Event Mode"
    CONTINUOUS_TIME = "Continuous-Time Mode"
    STEP = "Step Mode"
    CLOCK_ACTIVATION = "Clock Activation Mode"
    CLOCK_UPDATE = "Clock Update Mode"
    INTERMEDIATE_UPDATE = "Intermediate Update Mode"
    TERMINATED = "Terminated"
    FREED = "freed"
    """Not a state of the specification's machine -- the end of one.

    ``fmi3FreeInstance`` disposes of the instance, so what follows it is not a mode an
    instance can be in. It is named anyway because the machine needs somewhere for the
    instance to *go*: without it, Terminated looks like a dead end and every check about
    dead ends becomes a check about nothing.
    """


class Interface(StrEnum):
    """Which of the three FMI interfaces an instance was created for.

    The difference is small and it is entirely about where you land: exiting Initialization
    Mode goes to Event Mode for Model Exchange, Step Mode for Co-Simulation and Clock
    Activation Mode for Scheduled Execution.
    """

    MODEL_EXCHANGE = "Model Exchange"
    CO_SIMULATION = "Co-Simulation"
    SCHEDULED_EXECUTION = "Scheduled Execution"


class ModeError(Exception):
    """A call that the calling sequence does not allow where it was made."""


_SETTABLE: Mapping[str, Mode | None] = {
    "fmi3SetDebugLogging": None,
    "fmi3GetFMUState": None,
    "fmi3SetFMUState": None,
    "fmi3FreeFMUState": None,
    "fmi3SerializedFMUStateSize": None,
    "fmi3SerializeFMUState": None,
    "fmi3DeserializeFMUState": None,
    "fmi3Reset": Mode.INSTANTIATED,
    "fmi3FreeInstance": Mode.FREED,
}
"""The FMU State Settable super-state: legal wherever an instance exists.

``fmi3Reset`` and ``fmi3FreeInstance`` are in here because the specification puts them
there, and they are the two that matter for the checks -- they are why no mode is a trap.
"""

_STRUCTURE: Mapping[str, Mode | None] = {
    "fmi3GetNumberOfVariableDependencies": None,
    "fmi3GetVariableDependencies": None,
}

_READ: Mapping[str, Mode | None] = {
    "fmi3Get{VariableType}": None,
    "fmi3GetContinuousStateDerivatives": None,
    "fmi3GetContinuousStates": None,
    "fmi3GetNominalsOfContinuousStates": None,
    "fmi3GetEventIndicators": None,
    "fmi3GetDirectionalDerivative": None,
    "fmi3GetAdjointDerivative": None,
}


def _exit_initialization(interface: Interface) -> Mode:
    """Where ``fmi3ExitInitializationMode`` lands, which is the one real fork."""
    if interface is Interface.MODEL_EXCHANGE:
        return Mode.EVENT
    if interface is Interface.CO_SIMULATION:
        return Mode.STEP
    return Mode.CLOCK_ACTIVATION


def table(interface: Interface) -> Mapping[Mode, Mapping[str, Mode | None]]:
    """Every mode, mapped to the calls legal there and where each one leaves the instance.

    ``None`` as a target means the call is legal and does not move: most calls are like
    that, and saying so explicitly is what lets a guard reject everything not listed rather
    than having to know which absences are deliberate.
    """
    resume = _exit_initialization(interface)
    modes: dict[Mode, dict[str, Mode | None]] = {
        Mode.INSTANTIATED: {
            "fmi3Get{VariableType}": None,
            "fmi3Set{VariableType}": None,
            "fmi3GetNumberOfContinuousStates": None,
            "fmi3GetNumberOfEventIndicators": None,
            "fmi3EnterConfigurationMode": Mode.CONFIGURATION,
            "fmi3EnterInitializationMode": Mode.INITIALIZATION,
            **_STRUCTURE,
        },
        Mode.CONFIGURATION: {
            "fmi3Set{VariableType}": None,
            "fmi3ExitConfigurationMode": Mode.INSTANTIATED,
        },
        Mode.INITIALIZATION: {
            "fmi3Set{VariableType}": None,
            "fmi3GetShift{VariableType}": None,
            "fmi3SetShift{VariableType}": None,
            "fmi3GetInterval{VariableType}": None,
            "fmi3SetInterval{VariableType}": None,
            "fmi3ExitInitializationMode": resume,
            **_READ,
            **_STRUCTURE,
        },
        Mode.RECONFIGURATION: {
            "fmi3Set{VariableType}": None,
            "fmi3ExitConfigurationMode": resume,
        },
        Mode.TERMINATED: {**_READ, "fmi3GetOutputDerivatives": None, **_STRUCTURE},
    }
    if interface is not Interface.SCHEDULED_EXECUTION:
        # Event Mode belongs to Model Exchange and Co-Simulation. Scheduled Execution does
        # not have one -- its discrete work happens in Clock Activation Mode -- and putting
        # it in the table for all three was a transcription slip that `check_modes` caught
        # as a mode nothing could enter.
        modes[Mode.EVENT] = {
            "fmi3Set{VariableType}": None,
            "fmi3GetClock": None,
            "fmi3SetClock": None,
            "fmi3GetInterval{VariableType}": None,
            "fmi3GetNumberOfContinuousStates": None,
            "fmi3GetNumberOfEventIndicators": None,
            "fmi3EvaluateDiscreteStates": None,
            "fmi3UpdateDiscreteStates": None,
            "fmi3EnterConfigurationMode": Mode.RECONFIGURATION,
            "fmi3Terminate": Mode.TERMINATED,
            **_READ,
            **_STRUCTURE,
        }
    if interface is Interface.MODEL_EXCHANGE:
        modes[Mode.EVENT]["fmi3EnterContinuousTimeMode"] = Mode.CONTINUOUS_TIME
        modes[Mode.CONTINUOUS_TIME] = {
            "fmi3SetTime": None,
            "fmi3SetContinuousStates": None,
            "fmi3CompletedIntegratorStep": None,
            "fmi3EnterEventMode": Mode.EVENT,
            "fmi3Terminate": Mode.TERMINATED,
            **_READ,
            **_STRUCTURE,
        }
    if interface is Interface.CO_SIMULATION:
        modes[Mode.EVENT]["fmi3EnterStepMode"] = Mode.STEP
        modes[Mode.STEP] = {
            "fmi3Set{VariableType}": None,
            "fmi3GetOutputDerivatives": None,
            "fmi3DoStep": None,
            "fmi3EnterEventModeCS": Mode.EVENT,
            "fmi3EnterConfigurationMode": Mode.RECONFIGURATION,
            "fmi3IntermediateUpdateCallback": Mode.INTERMEDIATE_UPDATE,
            "fmi3Terminate": Mode.TERMINATED,
            **_READ,
            **_STRUCTURE,
        }
        modes[Mode.INTERMEDIATE_UPDATE] = {
            "fmi3Get{VariableType}": None,
            "fmi3Set{VariableType}": None,
            "fmi3GetOutputDerivatives": None,
            "return": Mode.STEP,
            **_STRUCTURE,
        }
    if interface is Interface.SCHEDULED_EXECUTION:
        modes[Mode.CLOCK_ACTIVATION] = {
            "fmi3Set{VariableType}": None,
            "fmi3GetClock": None,
            "fmi3SetClock": None,
            "fmi3GetInterval{VariableType}": None,
            "fmi3SetInterval{VariableType}": None,
            "fmi3ActivateModelPartition": None,
            "fmi3ClockUpdateCallback": Mode.CLOCK_UPDATE,
            "fmi3EnterConfigurationMode": Mode.RECONFIGURATION,
            "fmi3Terminate": Mode.TERMINATED,
            **_READ,
            **_STRUCTURE,
        }
        modes[Mode.CLOCK_UPDATE] = {
            "fmi3GetClock": None,
            "fmi3GetInterval{VariableType}": None,
            "return": Mode.CLOCK_ACTIVATION,
            **_STRUCTURE,
        }
    # The FMU State Settable super-state, expanded. Written out per mode rather than kept as
    # a super-state because a guard has to answer "is this call legal *here*" without
    # walking a hierarchy, and the expansion is the same answer with no traversal. The
    # instance no longer exists once freed, so that is the one mode it does not reach.
    for mode, calls in modes.items():
        if mode is not Mode.FREED:
            calls.update({name: target for name, target in _SETTABLE.items() if name not in calls})
    modes[Mode.FREED] = {}
    return modes


@dataclass
class Lifecycle:
    """Where an instance is, and what may be called there.

    The same object serves an exported FMU (which asks "may I do this?" before doing it) and
    an importer (which asks "where am I?" before choosing what to call next), because both
    questions are the table read in different directions.
    """

    interface: Interface
    mode: Mode = Mode.INSTANTIATED
    history: list[str] = field(default_factory=list[str])
    """Every call made, in order -- the thing to put in the error message when the sequence
    goes wrong, since by then the mistake is usually several calls back."""

    @property
    def allowed(self) -> frozenset[str]:
        """The calls legal in the current mode."""
        return frozenset(table(self.interface)[self.mode])

    def call(self, name: str) -> Mode:
        """Make ``name`` and return the mode it leaves the instance in.

        Raises :class:`ModeError` when the sequence does not allow it, which is what an
        exported FMU turns into ``fmi3Error``.
        """
        moves = table(self.interface)[self.mode]
        if name not in moves:
            raise ModeError(
                f"{name} is not allowed in {self.mode}"
                f" ({self.interface}); after {' -> '.join(self.history) or 'instantiation'}."
                f" Allowed here: {', '.join(sorted(moves))}"
            )
        self.history.append(name)
        target = moves[name]
        if target is not None:
            self.mode = target
        return self.mode
