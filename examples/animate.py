"""Render an example model as a movie -- the picture a mechanism is, rather than its columns.

A trajectory of a *mechanism* is a shape in space, and a chart of `x` against `t` is the one
picture that does not show it. Everything else in this corpus reads as a line plot; a
pendulum does not. So this draws the model instead: bodies at their coordinates, rods
between them, the path a bob has swept, and one panel underneath for the scalar worth
watching -- usually total energy, which is also how you find out the equations are wrong.

The scene lives in the model's own `.toml`, next to the reference points and the invariants,
because it is the same kind of statement: *this* is what the model looks like. Coordinates
are Modelica expressions over the model's variables, parsed and emitted by the toolchain
itself (`parse_expression` + `emit_expr`), so `x` means the model's `x` and nothing has to
be re-derived in Python.

    [animate]
    fps = 30
    speed = 1.0                 # seconds of model time per second of video
    xlim = [-1.4, 1.4]
    ylim = [-1.5, 0.3]

    [[animate.part]]
    kind = "rod"                # also: spring, mass, pivot, track
    from = ["0", "0"]
    to = ["x", "y"]

    [[animate.part]]
    kind = "mass"
    at = ["x", "y"]
    trace = 3.0                 # seconds of path kept behind it

    [[animate.plot]]
    expr = "0.5*m*(vx^2 + vy^2) + m*g*y"
    label = "total energy (J)"

Nothing here is imported by the library. matplotlib and ffmpeg are the docs' dependencies,
for the reason spelled out in `modelica.sim.view`: a `Result` draws its own chart in vanilla
JavaScript precisely so the *package* needs neither, and here we are the notebook, not the
package.

    uv run --extra sim --extra docs python examples/animate.py Pendulum --out pendulum.mp4
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from modelica.lang import parse, parse_expression
from modelica.sim import Result, causalize, flatten, simulate
from modelica.sim.emit import emit_expr

mpl.use("Agg")

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from numpy.typing import NDArray

HERE = Path(__file__).parent
MODELS = HERE / "models"
LIBRARY = HERE / "library"

# The dark steps of a validated categorical palette, on the dark surface they were stepped
# for. Two hues carry meaning here and no more: one for structure that is driven (the
# spring, the counterweight), one for what swings freely. A third arrives only if a scene
# asks for it.
SURFACE = "#1a1a19"
INK = "#ffffff"
MUTED = "#c3c2b7"
SERIES = ("#3987e5", "#d95926", "#199e70")

GRID = "#3a3a37"
"""Recessive enough to read as paper, not as data."""

COLLAPSED = 1e-9
"""A spring shorter than this has no direction to zigzag along; draw the segment."""


# --------------------------------------------------------------------------------------
# The scene, as it is written in the spec
# --------------------------------------------------------------------------------------

KINDS = ("rod", "spring", "mass", "pivot", "track")


@dataclass(frozen=True)
class Part:
    """One drawable thing, with its coordinates still as Modelica source."""

    kind: str
    points: tuple[tuple[str, str], ...]
    size: float = 1.0
    trace: float = 0.0
    coils: int = 12
    color: int = 0

    @classmethod
    def read(cls, spec: dict[str, object]) -> Part:
        kind = str(spec.get("kind", "rod"))
        if kind not in KINDS:
            raise SystemExit(f"unknown part kind {kind!r}; expected one of {', '.join(KINDS)}")
        if kind in ("rod", "spring", "track"):
            points = (_pair(spec, "from"), _pair(spec, "to"))
        else:
            points = (_pair(spec, "at"),)
        return cls(
            kind=kind,
            points=points,
            size=float(spec.get("size", 1.0)),  # pyright: ignore[reportArgumentType]
            trace=float(spec.get("trace", 0.0)),  # pyright: ignore[reportArgumentType]
            coils=int(spec.get("coils", 12)),  # pyright: ignore[reportArgumentType]
            color=int(spec.get("color", 0)),  # pyright: ignore[reportArgumentType]
        )


@dataclass(frozen=True)
class Panel:
    """A scalar to draw under the mechanism, as it accumulates."""

    expr: str
    label: str = ""
    color: int = 0
    ylim: tuple[float, float] | None = None
    """Fix the scale. Worth doing for a conserved quantity: left to itself the axis zooms
    into the integrator's last digit, and a line that is flat to 1 part in 10^9 comes out
    looking like violent noise -- the opposite of what it is there to show."""


@dataclass(frozen=True)
class Scene:
    """What to draw, how fast, and how much of the plane to show."""

    parts: tuple[Part, ...]
    plots: tuple[Panel, ...] = ()
    fps: int = 30
    speed: float = 1.0
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    title: str = ""
    trail_width: float = 1.8
    width: float = 7.0
    """Inches across. The height is *derived*, never given: a mechanism drawn on unequal
    axes is a different mechanism, so the frame is sized to the plane it has to show."""

    @classmethod
    def read(cls, spec: dict[str, object]) -> Scene:
        parts = tuple(Part.read(p) for p in spec.get("part", []))  # pyright: ignore[reportGeneralTypeIssues]
        if not parts:
            raise SystemExit("an [animate] section with no [[animate.part]] draws nothing")
        plots = tuple(
            Panel(
                expr=str(p["expr"]),
                label=str(p.get("label", "")),
                color=int(p.get("color", 0)),
                ylim=_limits(p.get("ylim")),
            )
            for p in spec.get("plot", [])  # pyright: ignore[reportGeneralTypeIssues]
        )
        return cls(
            parts=parts,
            plots=plots,
            fps=int(spec.get("fps", 30)),  # pyright: ignore[reportArgumentType]
            speed=float(spec.get("speed", 1.0)),  # pyright: ignore[reportArgumentType]
            xlim=_limits(spec.get("xlim")),
            ylim=_limits(spec.get("ylim")),
            title=str(spec.get("title", "")),
            width=float(spec.get("width", 7.0)),  # pyright: ignore[reportArgumentType]
        )


def _pair(spec: dict[str, object], key: str) -> tuple[str, str]:
    try:
        x, y = spec[key]  # pyright: ignore[reportGeneralTypeIssues]
    except KeyError:
        raise SystemExit(f'a {spec.get("kind", "rod")!r} part needs `{key} = ["x", "y"]`') from None
    return str(x), str(y)


def _limits(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    low, high = value  # pyright: ignore[reportGeneralTypeIssues]
    return float(low), float(high)


def _frame(
    scene: Scene, tracks: dict[tuple[str, str], NDArray[np.float64]]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The window on the plane: what the spec asked for, or everything the run ever uses."""
    if scene.xlim and scene.ylim:
        return scene.xlim, scene.ylim
    xs = np.concatenate([t[0] for t in tracks.values()])
    ys = np.concatenate([t[1] for t in tracks.values()])
    pad = 0.08 * max(np.ptp(xs), np.ptp(ys), 1e-3)
    return (
        scene.xlim or (float(xs.min()) - pad, float(xs.max()) + pad),
        scene.ylim or (float(ys.min()) - pad, float(ys.max()) + pad),
    )


# --------------------------------------------------------------------------------------
# Expressions over a result -- the same road the invariants take
# --------------------------------------------------------------------------------------


def _reader(
    expr: str, names: Sequence[str]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Compile a Modelica expression into something callable on the whole result at once.

    `Result.values` is one row per variable, so binding `v` to it and emitting each name as
    `v[row]` evaluates the expression over every sample in one go -- the same trick the
    invariant checks in `tests/test_examples.py` use one column at a time.
    """
    symbols = {name: f"v[{row}]" for row, name in enumerate(names)}
    code = compile(emit_expr(parse_expression(expr), symbols), f"<animate:{expr}>", "eval")

    def read(values: NDArray[np.float64]) -> NDArray[np.float64]:
        got = eval(code, {"np": np, "v": values})  # noqa: S307 - source we emitted ourselves
        return np.broadcast_to(np.asarray(got, dtype=float), (values.shape[1],))

    return read


def _track(scene: Scene, result: object) -> dict[tuple[str, str], NDArray[np.float64]]:
    """Every coordinate the scene mentions, evaluated once, as `(2, frames)` arrays."""
    values = result.values  # pyright: ignore[reportAttributeAccessIssue]
    names = result.names  # pyright: ignore[reportAttributeAccessIssue]
    wanted = {point for part in scene.parts for point in part.points}
    return {
        point: np.stack([_reader(point[0], names)(values), _reader(point[1], names)(values)])
        for point in wanted
    }


# --------------------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------------------


def _spring(
    a: NDArray[np.float64], b: NDArray[np.float64], coils: int, width: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A zigzag from `a` to `b`: flat lead-ins, then `coils` alternating offsets.

    Drawn between the *current* endpoints rather than at a fixed pitch, so a spring that is
    stretched shows it by opening out -- which is the whole reason the spring is in the
    picture and not just in the equations.
    """
    span = b - a
    length = float(np.hypot(*span))
    if length < COLLAPSED:
        return np.array([a[0], b[0]]), np.array([a[1], b[1]])
    along = span / length
    across = np.array([-along[1], along[0]])

    lead = 0.15
    s = np.concatenate([[0.0], np.linspace(lead, 1 - lead, 2 * coils + 1), [1.0]])
    offset = np.zeros_like(s)
    offset[2:-2:2] = width
    offset[1:-2:2] = -width
    offset[0] = offset[1] = offset[-1] = offset[-2] = 0.0

    points = a[:, None] + along[:, None] * (s * length) + across[:, None] * offset
    return points[0], points[1]


@dataclass
class _Artists:
    """The mutable half: matplotlib objects, and the trails they have accumulated."""

    lines: list[Line2D] = field(default_factory=list)
    trails: list[LineCollection] = field(default_factory=list)
    keep: list[int] = field(default_factory=list)
    """Frames of path each trail holds -- zero for a part that leaves none."""
    panels: list[Line2D] = field(default_factory=list)
    clock: object = None


def _mechanism(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def _panel(ax: Axes, panel: Panel, series: NDArray[np.float64], time: NDArray[np.float64]) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(visible=True, color=GRID, linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(float(time[0]), float(time[-1]))
    # No offset. A conserved quantity's axis otherwise reads `1e-10 -3.449951246e1`, which
    # is a true statement rendered as an unreadable one. A scale *factor* is still allowed:
    # that one is honest, because it multiplies rather than shifts.
    ax.ticklabel_format(axis="y", useOffset=False)

    if panel.ylim:
        ax.set_ylim(*panel.ylim)
    else:
        # Never zoom in past a hundredth of the quantity's own size. Below that the axis is
        # showing solver noise, and the reader has no way to know that from the picture.
        middle = (float(series.min()) + float(series.max())) / 2
        half = max(0.6 * float(np.ptp(series)), 0.01 * abs(middle), 1e-9)
        ax.set_ylim(middle - half, middle + half)
    ax.set_xlabel("time (s)", color=MUTED, fontsize=9)
    if panel.label:
        ax.set_ylabel(panel.label, color=MUTED, fontsize=9)


def build(  # noqa: C901, PLR0915 - one figure assembled top to bottom; splitting it hides the order
    scene: Scene, result: object
) -> tuple[Figure, Callable[[int], list[object]], int]:
    """The figure, the function that moves it to a frame, and how many frames there are.

    Split out from writing a movie because the movie is the slow half: a still at one
    instant is the same drawing, and it is what you want while a scene is being written.
    """
    time = np.asarray(result.time, dtype=float)  # pyright: ignore[reportAttributeAccessIssue]
    tracks = _track(scene, result)
    frames = len(time)

    # Height follows the plane, not a guess: the mechanism axes get exactly the aspect its
    # limits have, and each panel a fixed strip under it. Guessing instead leaves a
    # correctly-drawn mechanism marooned in a tall grey field, which is how this read first.
    xlim, ylim = _frame(scene, tracks)
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    stage = max(2.0, min(scene.width * aspect, 2.2 * scene.width))
    panel = 1.45
    fig = plt.figure(figsize=(scene.width, stage + panel * len(scene.plots)), facecolor=SURFACE)
    grid = fig.add_gridspec(
        1 + len(scene.plots),
        1,
        height_ratios=[stage, *[panel] * len(scene.plots)],
        hspace=0.5,
        left=0.13,
        right=0.97,
        top=0.93,
        bottom=0.08,
    )

    ax = fig.add_subplot(grid[0])
    _mechanism(ax, xlim, ylim)
    if scene.title:
        ax.set_title(scene.title, color=INK, fontsize=12, pad=12)

    art = _Artists()
    for part in scene.parts:
        colour = SERIES[part.color % len(SERIES)]
        if part.kind == "track":
            a, b = (tracks[p][:, 0] for p in part.points)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GRID, lw=1.2, zorder=1)
            art.lines.append(ax.plot([], [], alpha=0)[0])  # keep the artist list aligned
        elif part.kind == "pivot":
            art.lines.append(
                ax.plot([], [], "o", color=MUTED, ms=6 * part.size, zorder=4, mec=SURFACE)[0]
            )
        elif part.kind == "mass":
            art.lines.append(
                ax.plot([], [], "o", color=colour, ms=14 * part.size, zorder=5, mec=SURFACE, mew=2)[
                    0
                ]
            )
        else:
            width = 3.0 if part.kind == "rod" else 2.0
            art.lines.append(ax.plot([], [], "-", color=MUTED, lw=width, zorder=3)[0])

        trail = LineCollection([], colors=colour, linewidths=scene.trail_width, zorder=2)
        ax.add_collection(trail)
        art.trails.append(trail)
        art.keep.append(int(part.trace * scene.fps))

    series = [_reader(p.expr, result.names)(result.values) for p in scene.plots]  # pyright: ignore[reportAttributeAccessIssue]
    for row, (panel, values) in enumerate(zip(scene.plots, series, strict=True), start=1):
        pax = fig.add_subplot(grid[row])
        _panel(pax, panel, values, time)
        art.panels.append(
            pax.plot([], [], color=SERIES[panel.color % len(SERIES)], lw=2.0)[0],
        )

    art.clock = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
        va="top",
        family="monospace",
    )

    def update(frame: int) -> list[object]:
        for part, line, trail, keep in zip(
            scene.parts, art.lines, art.trails, art.keep, strict=True
        ):
            points = [tracks[p][:, frame] for p in part.points]
            if part.kind == "spring":
                line.set_data(*_spring(points[0], points[1], part.coils, 0.05 * part.size))
            elif part.kind in ("rod", "track"):
                line.set_data([points[0][0], points[1][0]], [points[0][1], points[1][1]])
            else:
                line.set_data([points[0][0]], [points[0][1]])

            if keep:
                start = max(0, frame - keep)
                path = tracks[part.points[0]][:, start : frame + 1].T
                if len(path) > 1:
                    segments = np.stack([path[:-1], path[1:]], axis=1)
                    trail.set_segments(list(segments))
                    # Fade to nothing at the tail: the head reads as *now* without a
                    # second colour, and an un-faded trail turns into a scribble.
                    trail.set_alpha(list(np.linspace(0.06, 0.75, len(segments))))

        for line, values in zip(art.panels, series, strict=True):
            line.set_data(time[: frame + 1], values[: frame + 1])

        art.clock.set_text(f"t = {time[frame]:7.3f} s")  # pyright: ignore[reportAttributeAccessIssue]
        return [*art.lines, *art.trails, *art.panels, art.clock]

    return fig, update, frames


def render(scene: Scene, result: object, out: Path, *, dpi: int = 140) -> Path:
    """Draw `result` through `scene` and write a movie to `out` -- `.mp4` or `.gif`."""
    fig, update, frames = build(scene, result)
    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / scene.fps, blit=False)
    writer = (
        PillowWriter(fps=scene.fps)
        if out.suffix == ".gif"
        else FFMpegWriter(fps=scene.fps, bitrate=4000, codec="libx264")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(out), writer=writer, dpi=dpi, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return out


def still(scene: Scene, result: object, out: Path, when: float, *, dpi: int = 140) -> Path:
    """The one frame nearest model time `when`, as an image."""
    fig, update, _ = build(scene, result)
    time = np.asarray(result.time, dtype=float)  # pyright: ignore[reportAttributeAccessIssue]
    update(int(np.argmin(np.abs(time - when))))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------
# Driving it from the corpus
# --------------------------------------------------------------------------------------


def run(name: str, scene: Scene, stop: float, spec: dict[str, object]) -> object:
    """Simulate, then keep exactly one sample per frame.

    `simulate(times=...)` *adds* the requested instants to the solver's own steps rather
    than replacing them, which is right for a result you are going to ask questions of and
    wrong for one you are going to draw: a 20 s run came back with 20349 samples and the
    movie tried to be 20349 frames long. So the grid is asked for and then enforced.
    """
    sources = [
        parse((LIBRARY / f"{lib}.mo").read_text(encoding="utf-8"))
        for lib in spec.get("requires", [])  # pyright: ignore[reportGeneralTypeIssues]
    ]
    sources.append(parse((MODELS / f"{name}.mo").read_text(encoding="utf-8")))
    system = causalize(flatten(sources, name))

    simulation: dict[str, object] = spec.get("simulate", {})  # pyright: ignore[reportAssignmentType]
    grid = np.linspace(0.0, stop, max(2, round(scene.fps * stop / scene.speed)))
    result = simulate(
        system,
        stop=stop,
        times=grid,
        tolerance=float(simulation.get("tolerance", 1e-8)),  # pyright: ignore[reportArgumentType]
        method=str(simulation.get("method", "Radau")),
    )
    return Result(
        name=result.name,
        time=grid,
        names=result.names,
        values=np.stack([np.interp(grid, result.time, row) for row in result.values]),
        events=result.events,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="a model in examples/models, by name")
    parser.add_argument("--out", type=Path, help="output file; .mp4 or .gif (default: <model>.mp4)")
    parser.add_argument(
        "--still",
        type=float,
        metavar="T",
        help="write one frame at model time T instead of a movie",
    )
    parser.add_argument("--stop", type=float, help="override the spec's horizon")
    parser.add_argument("--fps", type=int, help="override the scene's frame rate")
    parser.add_argument("--speed", type=float, help="model seconds per second of video")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)

    source = MODELS / f"{args.model}.toml"
    if not source.exists():
        parser.error(f"no such model: {source}")
    spec: dict[str, object] = tomllib.loads(source.read_text(encoding="utf-8"))
    if "animate" not in spec:
        parser.error(f"{source.name} has no [animate] section -- nothing says what to draw")

    scene = Scene.read(spec["animate"])  # pyright: ignore[reportArgumentType]
    if args.fps or args.speed:
        scene = replace(scene, fps=args.fps or scene.fps, speed=args.speed or scene.speed)

    simulation: dict[str, object] = spec.get("simulate", {})  # pyright: ignore[reportAssignmentType]
    animate: dict[str, object] = spec["animate"]  # pyright: ignore[reportAssignmentType]
    stop = args.stop or float(animate.get("stop", simulation.get("stop", 10.0)))  # pyright: ignore[reportArgumentType]

    result = run(args.model, scene, stop, spec)
    print(f"{args.model}: {len(result.time)} frames over {stop:g} s of model time")  # pyright: ignore[reportAttributeAccessIssue]
    if args.still is not None:
        out = args.out or Path(f"{args.model}.png")
        still(scene, result, out, args.still, dpi=args.dpi)
    else:
        out = args.out or Path(f"{args.model}.mp4")
        render(scene, result, out, dpi=args.dpi)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
