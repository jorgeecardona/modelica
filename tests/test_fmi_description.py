"""``modelDescription.xml`` against the model it came from.

`modelica.fmi.describe` writes the half of an FMU that is not a binary. What can be checked
without a compiler is exactly what an importer checks before it runs a step: that every
variable is there, that its causality and variability are the FMI spellings of the Modelica
ones, that a continuous state points at its own derivative, and that the value references
agree with the order the emitted ``observe`` fills.

Checked against the corpus rather than against a fixture, because the failure this guards is a
model whose shape the writer did not anticipate -- an alias, a discrete variable, a state whose
derivative alias elimination renamed.
"""

from __future__ import annotations

import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from modelica.ir.causalize import causalize
from modelica.ir.source import flatten
from modelica.lang import parse

if TYPE_CHECKING:
    from modelica.ir.system import System

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"
MODEL_FILES = sorted(MODELS.glob("*.mo"))

FMI_CAUSALITY = frozenset(
    {
        "parameter",
        "calculatedParameter",
        "input",
        "output",
        "local",
        "independent",
        "structuralParameter",
    }
)
FMI_VARIABILITY = frozenset({"constant", "fixed", "tunable", "discrete", "continuous"})


def _system(stem: str) -> System:
    spec = tomllib.loads((MODELS / f"{stem}.toml").read_text(encoding="utf-8"))
    files = [LIBRARY / f"{name}.mo" for name in spec["requires"]] + [MODELS / f"{stem}.mo"]
    return causalize(flatten([parse(f.read_text(encoding="utf-8")) for f in files], stem))


def _described(stem: str) -> ET.Element:
    describe = pytest.importorskip("modelica.fmi.description").describe
    return ET.fromstring(describe(_system(stem)))  # noqa: S314 - our own output


@pytest.mark.parametrize("model", MODEL_FILES, ids=[path.stem for path in MODEL_FILES])
def test_every_variable_is_described_once_with_a_legal_causality(model: Path) -> None:
    """The table an importer reads first, over the whole corpus.

    Every observed variable appears exactly once, its value reference is its slot in the
    emitted ``observe`` order, and both enumerations are spellings FMI 3.0 defines. A wrong
    causality is the failure that makes a tool reject the FMU before running it, and it is
    invisible until a tool tries.
    """
    emit = pytest.importorskip("modelica.sim.emit").emit
    root = _described(model.stem)
    code = emit(_system(model.stem), observed_jacobian=False)

    described = list(root.iterfind("./ModelVariables/*"))
    names = [element.get("name") for element in described]
    assert names == list(code.observed), "the description and `observe` disagree about order"
    assert len(set(names)) == len(names), "a variable is described twice"

    for slot, element in enumerate(described):
        assert element.get("valueReference") == str(slot), element.get("name")
        assert element.get("causality") in FMI_CAUSALITY, element.get("causality")
        assert element.get("variability") in FMI_VARIABILITY, element.get("variability")


@pytest.mark.parametrize("model", MODEL_FILES, ids=[path.stem for path in MODEL_FILES])
def test_a_state_points_at_its_own_derivative(model: Path) -> None:
    """FMI declares a state by naming its derivative's value reference, not its own.

    The direction is the opposite of `Mode.derivative` and is the one place this mapping is
    easy to write backwards -- so it is checked against the schedule rather than against
    itself: the element the state points at has to be the variable causalization decided is
    that state's derivative, which for an aliased model is the modeller's own name and not
    ``der_x``.
    """
    system = _system(model.stem)
    root = _described(model.stem)
    described = list(root.iterfind("./ModelVariables/*"))
    references = {element.get("name"): element.get("valueReference") for element in described}
    derivative = dict(system.modes[0].derivative)

    for element in described:
        pointer = element.get("derivative")
        name = element.get("name")
        if pointer is None:
            assert name not in derivative or derivative[name] not in references, name
            continue
        assert name in derivative, f"{name} points at a derivative and is not a state"
        assert pointer == references[derivative[name]], name

    listed = {
        element.get("valueReference")
        for element in root.iterfind("./ModelStructure/ContinuousStateDerivative")
    }
    assert listed == {references[derivative[state]] for state in system.states}


def test_a_parameter_is_fixed_and_carries_its_value() -> None:
    """`NewtonCooling` reads as it should, attribute by attribute.

    One model spelled out, because the corpus-wide tests above check *consistency* and this
    checks that the consistent thing is the right thing. ``fixed`` rather than ``tunable`` is
    the choice worth pinning: Modelica's `parameter` is constant through a run, and calling it
    tunable would promise an importer a change the equations do not support.
    """
    root = _described("NewtonCooling")
    parameters = {
        element.get("name"): element
        for element in root.iterfind("./ModelVariables/*")
        if element.get("causality") == "parameter"
    }
    assert set(parameters) == {"T_amb", "k"}
    assert parameters["k"].get("variability") == "fixed"
    assert parameters["k"].get("start") == "0.5"
    assert parameters["k"].get("unit") == "1/s"
    assert parameters["k"].get("description") == "Cooling coefficient"


def test_the_document_is_fmi_three() -> None:
    """The version, and the absence of an instantiation token.

    FMI 3.0 requires the binary and the description to agree on the token, so writing one here
    -- with no binary in existence -- would be a promise this file cannot keep.
    """
    root = _described("RCCircuit")
    assert root.tag == "fmiModelDescription"
    assert root.get("fmiVersion") == "3.0"
    assert root.get("instantiationToken") is None
    assert root.find("./ModelExchange") is not None


def test_every_unit_a_variable_names_is_defined_before_the_variables() -> None:
    """A dangling unit reference, and the schema order that hides one.

    Two claims. Every ``unit=`` on a variable resolves to a ``<Unit>`` -- an importer reading
    one that does not is entitled to reject the FMU. And ``UnitDefinitions`` comes *before*
    ``ModelVariables``, because FMI 3.0 fixes the order of these children and a document with
    the right content in the wrong order is rejected by a validating reader while looking
    perfectly fine to a person.
    """
    root = _described("RoastingMachine")
    order = [child.tag for child in root]
    assert order.index("UnitDefinitions") < order.index("ModelVariables")
    assert order.index("ModelVariables") < order.index("ModelStructure")

    defined = {unit.get("name", "") for unit in root.iterfind("./UnitDefinitions/Unit")}
    used = {
        element.get("unit", "")
        for element in root.iterfind("./ModelVariables/*")
        if element.get("unit")
    }
    assert used <= defined, f"undefined units: {sorted(used - defined)}"


def test_degrees_celsius_carries_its_offset() -> None:
    """The one unit in the corpus that is not a product of powers.

    Converting ``degC`` to kelvin means adding 273.15, and ``offset`` is the only place that
    can be said. Getting it wrong is a twenty-degree error in somebody else's plot, which is
    exactly the kind of mistake an exporter makes silently.
    """
    root = _described("NewtonCooling")
    celsius = next(
        unit for unit in root.iterfind("./UnitDefinitions/Unit") if unit.get("name") == "degC"
    )
    base = celsius.find("./BaseUnit")
    assert base is not None
    assert base.get("K") == "1"
    assert base.get("offset") == "273.15"


def test_a_unit_this_package_cannot_decompose_is_named_and_not_guessed() -> None:
    """``W/K`` gets a `<Unit>` with no `<BaseUnit>`, which is legal and honest.

    Parsing Modelica's unit grammar to derive exponents is a real piece of work; guessing them
    is worse than declining. A named unit with no decomposition says what is true -- this is
    the string the model used -- and claims nothing about converting it.
    """
    root = _described("TwoBodyHeat")
    named = {unit.get("name", ""): unit for unit in root.iterfind("./UnitDefinitions/Unit")}
    assert "W/K" in named, sorted(named)
    assert named["W/K"].find("./BaseUnit") is None
    assert named["K"].find("./BaseUnit") is not None, "the ones we do know still carry theirs"
