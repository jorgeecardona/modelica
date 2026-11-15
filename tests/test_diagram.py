"""The diagram package, checked against the example corpus.

Three properties matter, and each of them can break quietly:

* **Every model draws.** A picture that raises on one model in ten is not a tool. So the
  whole corpus goes through :func:`~modelica.diagram.schematic` here.
* **The picture agrees with the flattener.** Both read the same classes. If the diagram
  showed a component the flattener does not instantiate, the picture would be a lie, and
  it would be a convincing one.
* **The output does not move.** These images go into a documentation site that a
  pipeline rebuilds. A layout that used random numbers would produce a new diff every
  time, and nobody would read the diffs after that.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from modelica import diagram
from modelica.diagram import blocks, incidence, of, schematic
from modelica.diagram.__main__ import main
from modelica.ir.flatten import FlattenError
from modelica.lang import parse
from modelica.sim import causalize, flatten

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
LIBRARY = EXAMPLES / "library"
MODELS = EXAMPLES / "models"
MODEL_FILES = sorted(MODELS.glob("*.mo"))


def _files(model: Path) -> list[Path]:
    """The model file and every library its spec requires."""
    spec = tomllib.loads(model.with_suffix(".toml").read_text(encoding="utf-8"))
    return [model, *(LIBRARY / f"{name}.mo" for name in spec["requires"])]


def _ids(paths: list[Path]) -> list[str]:
    return [path.stem for path in paths]


# --------------------------------------------------------------------------------------
# The whole corpus draws
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_every_model_has_a_schematic(model: Path) -> None:
    graph = of(*_files(model), model=model.stem)
    assert graph.name == model.stem
    assert graph.kind == "schematic"
    root = ET.fromstring(graph.to_svg())  # noqa: S314 - the string under test is our own
    assert root.tag.endswith("svg")


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_the_schematic_names_what_the_flattener_instantiates(model: Path) -> None:
    """No box may name a component the flat model does not have.

    A component ``r`` puts variables ``r.something`` in the flat model. If the diagram
    invented a component, or kept one the flattener dropped, this finds it.
    """
    files = _files(model)
    graph = of(*files, model=model.stem)
    flat = flatten([parse(path.read_text(encoding="utf-8")) for path in files], model.stem)
    prefixes = {name.split(".")[0] for name in (v.name for v in flat.variables) if "." in name}
    drawn = {node.id for node in graph.nodes if node.kind == "component"}
    assert drawn <= prefixes or not prefixes


@pytest.mark.parametrize("model", MODEL_FILES, ids=_ids(MODEL_FILES))
def test_drawing_is_repeatable(model: Path) -> None:
    first = of(*_files(model), model=model.stem).to_svg()
    second = of(*_files(model), model=model.stem).to_svg()
    assert first == second


# --------------------------------------------------------------------------------------
# The schematic of a circuit
# --------------------------------------------------------------------------------------


@pytest.fixture
def rlc() -> diagram.Graph:
    return of(MODELS / "RLCCircuit.mo", LIBRARY / "Electrical.mo")


def test_components_are_boxes(rlc: diagram.Graph) -> None:
    assert [node.id for node in rlc.nodes if node.kind == "component"] == [
        "source",
        "r",
        "l",
        "c",
        "ground",
    ]


def test_a_component_carries_its_type_and_its_settings(rlc: diagram.Graph) -> None:
    """The type is named the way the *source* names it, alias and all.

    ``RLCCircuit`` writes ``import E = Electrical;``, so its components are declared as
    ``E.Resistor`` and the picture says ``E.Resistor``. Resolving the alias away here would
    print a name that appears nowhere in the file the reader has open; the diagram's job is
    to show the model as written, and what the flattener makes of the name is its own.
    """
    assert rlc["r"].type_name == "E.Resistor"
    assert rlc["r"].detail == ("R = 1",)
    assert [port.name for port in rlc["r"].ports] == ["p", "n"]


def test_three_connectors_meeting_make_one_junction(rlc: diagram.Graph) -> None:
    """``c.n``, ``source.n`` and ``ground.p`` are one connection set, not three lines.

    That is what the flattener does with them: one sum of flows, taken once. Drawing
    three separate lines would put the sum in a place it is not taken.
    """
    junctions = [node for node in rlc.nodes if node.kind == "junction"]
    assert len(junctions) == 1
    joined = set(rlc.neighbours(junctions[0].id))
    assert joined == {"c", "source", "ground"}


def test_a_physical_connection_has_no_direction(rlc: diagram.Graph) -> None:
    assert not any(edge.directed for edge in rlc.edges)
    assert all(edge.kind == "connection" for edge in rlc.edges)


# --------------------------------------------------------------------------------------
# Signals, which do have a direction
# --------------------------------------------------------------------------------------

SIGNALS = """
package Blocks
  connector RealInput
    input Real value;
  end RealInput;

  connector RealOutput
    output Real value;
  end RealOutput;

  model Constant
    parameter Real k = 1;
    RealOutput y;
  equation
    y.value = k;
  end Constant;

  model Gain
    parameter Real k = 2;
    RealInput u;
    RealOutput y;
  equation
    y.value = k * u.value;
  end Gain;

  model Sink
    RealInput u;
    Real seen;
  equation
    seen = u.value;
  end Sink;
end Blocks;

model Chain
  Blocks.Constant source(k = 3);
  Blocks.Gain gain(k = 2);
  Blocks.Sink sink;
equation
  connect(source.y, gain.u);
  connect(gain.y, sink.u);
end Chain;
"""


def test_a_signal_connection_points_from_the_output_to_the_input() -> None:
    graph = schematic(parse(SIGNALS), "Chain")
    edges = {(edge.source, edge.target): edge for edge in graph.edges}
    assert set(edges) == {("source", "gain"), ("gain", "sink")}
    assert all(edge.directed for edge in edges.values())
    assert all(edge.kind == "signal" for edge in edges.values())


def test_a_signal_connection_written_backwards_still_points_forwards() -> None:
    """``connect(gain.u, source.y)`` is the same connection, and the same arrow."""
    source = SIGNALS.replace("connect(source.y, gain.u);", "connect(gain.u, source.y);")
    graph = schematic(parse(source), "Chain")
    assert ("source", "gain") in {(edge.source, edge.target) for edge in graph.edges}


def test_the_arrow_survives_the_svg() -> None:
    text = schematic(parse(SIGNALS), "Chain").to_svg()
    assert "marker-end" in text


# --------------------------------------------------------------------------------------
# Opening a component
# --------------------------------------------------------------------------------------

NESTED = """
package Electrical
  connector Pin
    Real v;
    flow Real i;
  end Pin;

  model Resistor
    parameter Real R = 1;
    Pin p;
    Pin n;
  equation
    p.v - n.v = R * p.i;
    p.i + n.i = 0;
  end Resistor;

  model Divider "Two resistors in series, as one component"
    Pin p;
    Pin n;
    Resistor upper(R = 10);
    Resistor lower(R = 20);
  equation
    connect(p, upper.p);
    connect(upper.n, lower.p);
    connect(lower.n, n);
  end Divider;
end Electrical;

model Probe
  Electrical.Divider divider;
  Electrical.Resistor load(R = 5);
equation
  connect(divider.n, load.p);
end Probe;
"""


def test_a_closed_component_is_one_box() -> None:
    graph = schematic(parse(NESTED), "Probe", depth=1)
    assert {node.id for node in graph.nodes if node.kind == "component"} == {"divider", "load"}
    assert not graph.is_nested


def test_an_opened_component_shows_what_is_inside_it() -> None:
    graph = schematic(parse(NESTED), "Probe", depth=2)
    assert "divider.upper" in graph
    assert graph["divider.upper"].parent == "divider"
    assert graph["divider.upper"].detail == ("R = 10",)
    inside = {node.id for node in graph.children("divider")}
    assert inside == {"divider.p", "divider.n", "divider.upper", "divider.lower"}


def test_an_outer_connection_lands_on_the_port_of_an_opened_component() -> None:
    """With ``divider`` opened, ``connect(divider.n, load.p)`` reaches the port node."""
    graph = schematic(parse(NESTED), "Probe", depth=2)
    assert ("divider.n", "load") in {(edge.source, edge.target) for edge in graph.edges}


def test_the_frame_of_an_opened_component_is_drawn() -> None:
    text = schematic(parse(NESTED), "Probe", depth=2).to_svg()
    assert 'class="frame"' in text


# --------------------------------------------------------------------------------------
# The other two views
# --------------------------------------------------------------------------------------


def test_incidence_joins_equations_to_the_variables_in_them() -> None:
    flat = flatten(
        parse((MODELS / "NewtonCooling.mo").read_text(encoding="utf-8")), "NewtonCooling"
    )
    graph = incidence(flat)
    assert graph.kind == "incidence"
    equations = [node for node in graph.nodes if node.kind == "equation"]
    assert len(equations) == len(flat.equations)
    for edge in graph.edges:
        assert edge.source.startswith("e:")
        assert edge.target.startswith("v:")


def test_incidence_leaves_the_parameters_out() -> None:
    """A parameter is known before the solver starts, so no equation is spent on one."""
    flat = flatten(
        parse((MODELS / "NewtonCooling.mo").read_text(encoding="utf-8")), "NewtonCooling"
    )
    drawn = {node.label for node in incidence(flat).nodes if node.kind in ("variable", "state")}
    assert "k" not in drawn
    assert "T" in drawn
    assert "k" in {node.label for node in incidence(flat, parameters=True).nodes}


def test_a_state_is_marked_as_one() -> None:
    flat = flatten(
        parse((MODELS / "NewtonCooling.mo").read_text(encoding="utf-8")), "NewtonCooling"
    )
    states = {node.label for node in incidence(flat).nodes if node.kind == "state"}
    assert states == set(flat.states)


def test_blocks_run_in_an_order_with_no_cycle() -> None:
    files = _files(MODELS / "RLCCircuit.mo")
    system = causalize(flatten([parse(f.read_text(encoding="utf-8")) for f in files], "RLCCircuit"))
    graph = blocks(system)
    assert len(graph.nodes) == len(system.blocks)
    # Every edge points forwards. Causalization sorted the blocks, so an edge that went
    # backwards would mean the sort was wrong, not that the picture was.
    for edge in graph.edges:
        assert int(edge.source.removeprefix("b:")) < int(edge.target.removeprefix("b:"))
        assert edge.directed


# --------------------------------------------------------------------------------------
# Writing it out
# --------------------------------------------------------------------------------------


def test_mermaid_names_every_component(rlc: diagram.Graph) -> None:
    text = rlc.to_mermaid()
    assert text.startswith("graph ")
    for name in ("source", "r", "l", "c", "ground"):
        assert f'"{name}<br/>' in text


def test_dot_is_one_digraph(rlc: diagram.Graph) -> None:
    text = rlc.to_dot()
    assert text.startswith('digraph "RLCCircuit" {')
    assert text.rstrip().endswith("}")
    assert text.count("->") == len(rlc.edges)


def test_the_svg_needs_nothing_from_the_network(rlc: diagram.Graph) -> None:
    """No script, no stylesheet, no font, no image. It has to work offline and inline."""
    text = rlc.to_svg()
    assert "<script" not in text
    assert "http://www.w3.org/2000/svg" in text
    assert "https://" not in text.replace("http://www.w3.org/2000/svg", "")


def test_the_svg_takes_its_colour_from_the_page(rlc: diagram.Graph) -> None:
    """Light page or dark page, one image. Nothing here names a colour.

    A named colour is what makes a second, dark copy of an image necessary. Every line
    and every character here inherits the colour of the text around it instead.
    """
    text = rlc.to_svg()
    assert "currentColor" in text
    assert not re.search(r"#[0-9a-fA-F]{3}\b", text)
    assert not re.search(r"\b(black|white|rgb|hsl)\b", text)


def test_a_notebook_shows_the_graph(rlc: diagram.Graph) -> None:
    # The display protocol asks for exactly this name, so calling it is the test.
    assert rlc._repr_html_().startswith("<svg")  # pyright: ignore[reportPrivateUsage]


def test_write_picks_the_format_from_the_suffix(rlc: diagram.Graph, tmp_path: Path) -> None:
    assert rlc.write(tmp_path / "a.svg").read_text().startswith("<svg")
    assert rlc.write(tmp_path / "a.dot").read_text().startswith("digraph")
    assert rlc.write(tmp_path / "a.md").read_text().startswith("```mermaid")
    with pytest.raises(ValueError, match="cannot write a graph"):
        rlc.write(tmp_path / "a.png")


# --------------------------------------------------------------------------------------
# Saying no clearly
# --------------------------------------------------------------------------------------


def test_a_package_is_not_a_model() -> None:
    with pytest.raises(FlattenError, match="is a package, not a model"):
        schematic(parse(SIGNALS), "Blocks")


def test_it_says_what_the_choices_are_when_the_files_hold_several_models() -> None:
    with pytest.raises(diagram.AmbiguousModelError, match="NewtonCooling"):
        of(MODELS / "RLCCircuit.mo", MODELS / "NewtonCooling.mo", LIBRARY / "Electrical.mo")


def test_one_model_among_libraries_needs_no_name() -> None:
    """A library holds its models inside a package, so it does not make the choice hard."""
    assert of(MODELS / "RLCCircuit.mo", LIBRARY / "Electrical.mo").name == "RLCCircuit"


def test_the_command_writes_a_file(tmp_path: Path) -> None:
    out = tmp_path / "rlc.svg"
    code = main(
        [
            str(MODELS / "RLCCircuit.mo"),
            str(LIBRARY / "Electrical.mo"),
            "--model",
            "RLCCircuit",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert out.read_text().startswith("<svg")


def test_the_command_reports_a_bad_file_without_a_traceback(tmp_path: Path) -> None:
    bad = tmp_path / "Broken.mo"
    bad.write_text("model Broken\n  Real x\nend Broken;\n")
    assert main([str(bad)]) == 1


# --------------------------------------------------------------------------------------
# The picture on the docs home page
# --------------------------------------------------------------------------------------

DOCS_SCHEMATIC = EXAMPLES.parent / "docs" / "assets" / "rlc.svg"


def test_docs_home_page_schematic_is_what_the_code_draws() -> None:
    """The SVG that ``docs/index.md`` inlines is the one the code draws today.

    Regenerate it when the drawing changes: ``python -m modelica.diagram
    examples/models/RLCCircuit.mo examples/library/Electrical.mo > docs/assets/rlc.svg``.
    """
    drawn = of(MODELS / "RLCCircuit.mo", LIBRARY / "Electrical.mo").to_svg()
    assert DOCS_SCHEMATIC.read_text(encoding="utf-8").strip() == drawn.strip()


def test_icon_text_other_than_the_name_is_drawn_inside_the_symbol() -> None:
    """The "+" of a source is part of the symbol, not a second copy of the caption."""
    drawn = of(MODELS / "RLCCircuit.mo", LIBRARY / "Electrical.mo").to_svg()
    plus = re.findall(r'<text class="small"[^>]*>\+</text>', drawn)
    names = re.findall(r'<text class="name"[^>]*>source</text>', drawn)
    assert len(plus) == 1
    assert len(names) == 1
