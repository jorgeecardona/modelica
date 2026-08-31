"""Reachability and agreement over the Modelica Standard Library.

The corpus in ``examples/`` is written by the people who wrote the compiler, so every
number it produces is a measurement of the corpus more than of the toolchain. This script
produces the number nobody here got to choose the inputs for: over every example model in
the MSL at a pinned version, how many parse, flatten, causalize and simulate — and of the
ones that simulate, how many agree with OpenModelica.

Two rules from the issue that filed this (docs/issues, reachability-and-agreement):

* **Every refusal is attributed.** A model that does not reach the end reports which stage
  refused it and why, so the output is a work list ordered by how many models each missing
  construct blocks. "arrays block 340 models" is a decision; "28 %" alone is a headline.
* **Measuring and improving are different jobs.** This script only measures.

The MSL is fetched at ``MSL_VERSION`` into ``.cache/msl/`` (gitignored), never vendored.
The library on disk is the one-class-per-file layout: a directory with a ``package.mo``
is a package, its sibling ``.mo`` files and subdirectories are its members. The parser
discards ``within`` (the IR nests classes structurally), so this script reassembles the
nesting from the directory layout itself.

An *example model* is a non-``package.mo`` file under the ``Modelica`` package whose text
carries an ``experiment`` annotation, named by its path: ``Modelica/Blocks/Examples/
PID_Controller.mo`` is ``Modelica.Blocks.Examples.PID_Controller``. That rule is stated
here because the denominator is the half of the fraction that is easiest to fudge.

Agreement needs OpenModelica (``omc`` on PATH) as the reference — not a second run of
this toolchain. When ``omc`` is missing the agreement number is reported as unmeasured,
with the command to change that, rather than silently narrowed to the models we like.
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from modelica import ir  # noqa: E402
from modelica.ir.causalize import causalize  # noqa: E402
from modelica.ir.flatten import flatten  # noqa: E402
from modelica.lang import parse  # noqa: E402

MSL_VERSION = "4.1.0"
MSL_URL = (
    f"https://github.com/modelica/ModelicaStandardLibrary/archive/refs/tags/v{MSL_VERSION}.tar.gz"
)
CACHE = REPO / ".cache" / "msl" / f"v{MSL_VERSION}"

# Per-stage wall-clock budget, seconds. A stage that spins past this is recorded as a
# timeout and the walk moves on; a probe that hangs on model 217 of 476 measures nothing.
PARSE_TIMEOUT = 30
FLATTEN_TIMEOUT = 60
SIMULATE_TIMEOUT = 120

# Address-space ceiling, bytes. A pathological flatten must become a MemoryError in this
# process, attributed to its model, rather than the machine's problem.
MEMORY_LIMIT = 4 << 30

STAGES = ("parse", "flatten", "causalize", "simulate")
PAST = {
    "parse": "parsed",
    "flatten": "flattened",
    "causalize": "causalized",
    "simulate": "simulated",
}

WORK_LIST_ROWS = 25


class StageTimeoutError(Exception):
    """A stage ran past its budget."""


def _alarm(_sig: int, _frame: FrameType | None) -> None:
    raise StageTimeoutError


class deadline:  # noqa: N801 - used as a context manager, spelled like one
    """SIGALRM-based timeout: single-threaded code only, which this is."""

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds

    def __enter__(self) -> None:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(self.seconds)

    def __exit__(self, *exc: object) -> None:
        signal.alarm(0)


# -- fetch ----------------------------------------------------------------------------


def fetch() -> Path | None:
    """The MSL source tree at the pinned version, fetched if absent. None means no network."""
    marker = CACHE / "Modelica" / "package.mo"
    if marker.exists():
        return CACHE
    print(f"fetching MSL v{MSL_VERSION} from {MSL_URL} ...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            with urllib.request.urlopen(MSL_URL, timeout=120) as response:
                shutil.copyfileobj(response, tmp.file)
            tmp.file.flush()
            _extract(Path(tmp.name))
    except OSError as error:
        print(f"could not fetch the MSL: {error}")
        print(f"to run this by hand: download {MSL_URL} and unpack the `Modelica`,")
        print(f"`ModelicaServices` and `Complex.mo` entries into {CACHE}/")
        return None
    return CACHE


def _extract(tarball: Path) -> None:
    """Unpack only the Modelica source: no Resources, no test libraries, no images."""
    prefix = f"ModelicaStandardLibrary-{MSL_VERSION}/"
    keep = ("Modelica/", "ModelicaServices/", "Complex.mo")
    CACHE.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tar:
        for member in tar:
            name = member.name.removeprefix(prefix)
            if name == member.name or not name.startswith(keep):
                continue
            if not (name.endswith((".mo", "package.order"))):
                continue
            member.name = name
            tar.extract(member, CACHE, filter="data")


# -- assemble -------------------------------------------------------------------------


@dataclass
class FileResult:
    """One source file's parse outcome."""

    path: Path
    qualified: str
    """The dotted name of the class the file defines, from its path."""
    error: str = ""
    classes: tuple[ir.Class, ...] = ()


@dataclass
class Library:
    """The MSL as one Program, plus the per-file parse record the Program cannot hold."""

    program: ir.Program
    files: dict[str, FileResult] = field(default_factory=dict)
    """Keyed by qualified name."""
    blocked: dict[str, str] = field(default_factory=dict)
    """Subtrees whose package.mo refused: dotted prefix -> the refusal, verbatim."""

    @property
    def refused(self) -> dict[str, str]:
        """Every qualified name that never made it into the Program, with its refusal."""
        return {q: f.error for q, f in self.files.items() if f.error} | self.blocked


def _parse_file(path: Path, qualified: str, library: Library) -> FileResult:
    result = FileResult(path, qualified)
    try:
        with deadline(PARSE_TIMEOUT):
            result.classes = parse(path.read_text(encoding="utf-8-sig"), str(path)).classes
    except (SyntaxError, StageTimeoutError, RecursionError, MemoryError) as error:
        result.error = _first_line(error)
    library.files[qualified] = result
    return result


def _load_tree(directory: Path, qualified: str, library: Library) -> ir.Class | None:
    """The package a directory holds, with every member grafted in — or None if refused.

    A refused ``package.mo`` takes its whole subtree with it: without the package there is
    no scope for the members to live in. The refusal is recorded against the prefix so
    every example under it is attributed to this file, not to a mysterious lookup failure.
    """
    package = _parse_file(directory / "package.mo", qualified, library)
    if package.error or len(package.classes) != 1:
        library.blocked[qualified] = package.error or "package.mo defines more than one class"
        return None
    members: list[ir.Class] = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / "package.mo").exists():
            sub = _load_tree(child, f"{qualified}.{child.name}", library)
            if sub is not None:
                members.append(sub)
        elif child.suffix == ".mo" and child.name != "package.mo":
            parsed = _parse_file(child, f"{qualified}.{child.stem}", library)
            members.extend(parsed.classes)
    root = package.classes[0]
    return replace(root, classes=root.classes + tuple(members))


def assemble(msl: Path) -> Library:
    """Parse the whole library into one Program, recording every refusal on the way."""
    library = Library(program=ir.Program())
    roots: list[ir.Class] = []
    for name in ("Modelica", "ModelicaServices"):
        tree = _load_tree(msl / name, name, library)
        if tree is not None:
            roots.append(tree)
    complex_file = _parse_file(msl / "Complex.mo", "Complex", library)
    roots.extend(complex_file.classes)
    library.program = ir.Program(classes=tuple(roots))
    return library


# -- enumerate ------------------------------------------------------------------------


@dataclass
class Example:
    """One MSL example model: a file with an experiment annotation."""

    qualified: str
    path: Path
    stop: float
    tolerance: float
    stage: str = ""
    """The last stage that succeeded; "simulate" means reachable."""
    error: str = ""
    """Why the next stage refused, verbatim; empty when reachable."""
    seconds: float = 0.0


_EXPERIMENT = re.compile(r"\bexperiment\s*\(")
_STOP = re.compile(r"\bStopTime\s*=\s*([0-9eE.+-]+)")
_TOLERANCE = re.compile(r"\bTolerance\s*=\s*([0-9eE.+-]+)")


def enumerate_examples(msl: Path) -> list[Example]:
    examples = []
    for path in sorted((msl / "Modelica").rglob("*.mo")):
        if path.name == "package.mo":
            continue
        text = path.read_text(encoding="utf-8-sig")
        if not _EXPERIMENT.search(text):
            continue
        relative = path.relative_to(msl).with_suffix("")
        stop = _STOP.search(text)
        tolerance = _TOLERANCE.search(text)
        examples.append(
            Example(
                qualified=".".join(relative.parts),
                path=path,
                stop=float(stop.group(1)) if stop else 1.0,
                tolerance=float(tolerance.group(1)) if tolerance else 1e-6,
            ),
        )
    return examples


# -- drive ----------------------------------------------------------------------------


def drive(example: Example, library: Library) -> None:
    """One example through the pipeline, stopping at the first stage that refuses."""
    started = time.monotonic()
    example.stage, example.error = "", ""
    try:
        blocker = _parse_blocker(example, library)
        if blocker is not None:
            example.error = blocker
            return
        example.stage = "parse"
        try:
            with deadline(FLATTEN_TIMEOUT):
                flat = flatten(library.program, example.qualified)
        except Exception as error:
            # A lookup that fails because the *looked-up* file refused to parse is a parse
            # refusal wearing a flatten costume — 260 of the first run's 261 flatten
            # failures were this. Attribute it to the construct that actually refused.
            blamed = _blame_dependency(_first_line(error), library)
            if blamed is None:
                raise
            example.error = blamed
            return
        example.stage = "flatten"
        with deadline(FLATTEN_TIMEOUT):
            system = causalize(flat)
        example.stage = "causalize"
        # Deferred: scipy lives behind the `sim` extra, and the parse/flatten half of the
        # measurement must still run in an environment that never installed it.
        from modelica.sim import simulate  # noqa: PLC0415

        with deadline(SIMULATE_TIMEOUT):
            simulate(system, stop=example.stop, tolerance=example.tolerance)
        example.stage = "simulate"
    except StageTimeoutError:
        example.error = f"timed out after {_budget(example.stage)}s"
    except (Exception, RecursionError, MemoryError) as error:  # noqa: BLE001 - each refusal is the datum
        example.error = _first_line(error)
    finally:
        example.seconds = time.monotonic() - started


def _parse_blocker(example: Example, library: Library) -> str | None:
    """The parse refusal that blocks this example, its own file's or an ancestor's."""
    own = library.files.get(example.qualified)
    if own is not None and own.error:
        return own.error
    parts = example.qualified.split(".")
    for depth in range(1, len(parts)):
        prefix = ".".join(parts[:depth])
        if prefix in library.blocked:
            return f"package.mo of {prefix}: {library.blocked[prefix]}"
    return None


_NO_NESTED = re.compile(r"'([\w.]+)' has no nested class '(\w+)'")
_RESOLVING = re.compile(r"nothing named '([\w.]+)'|\(resolving '([\w.]+)'\)")


def _blame_dependency(message: str, library: Library) -> str | None:
    """The parse refusal behind a flatten lookup failure, or None when it is flatten's own.

    The names a lookup error carries may be absolute (``'Modelica.Clocked' has no nested
    class 'ClockSignals'`` composes to an absolute name) or written relative to the model,
    so a refused name is matched by equality, by containing the candidate as a prefix, or
    by ending with it.
    """
    candidates: list[str] = []
    nested = _NO_NESTED.search(message)
    if nested is not None:
        candidates.append(f"{nested.group(1)}.{nested.group(2)}")
    candidates.extend(
        next(group for group in match.groups() if group) for match in _RESOLVING.finditer(message)
    )
    for candidate in candidates:
        for name, why in library.refused.items():
            if (
                name == candidate
                or candidate.startswith(name + ".")
                or name.endswith("." + candidate)
            ):
                return f"dependency {name}: {why}"
    return None


def _budget(stage: str) -> int:
    return SIMULATE_TIMEOUT if stage == "causalize" else FLATTEN_TIMEOUT


def _first_line(error: BaseException) -> str:
    text = str(error).strip() or type(error).__name__
    return text.splitlines()[0]


# -- attribute ------------------------------------------------------------------------

_VIA = re.compile(r"^(package\.mo of [\w.]+|dependency [\w.]+): ")
_QUOTED = re.compile(r"'[^']*'")
_TOKEN_LENGTH = 12
"""A quoted token this short with no dot is a keyword or operator: 'algorithm', ':='.
That is the informative half of a syntax error, so grouping must keep it; a longer or
dotted quote is a model-specific name and gets blanked."""
_POSITION = re.compile(r"\bat \S+:\d+(:\d+)?|\s*\([^()]*,\s*line \d+\)")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(\.\w+)+\b")
_NUMBER = re.compile(r"\b\d+(\.\d+)?\b")


def normalise(message: str) -> str:
    """One refusal message with the model-specific parts blanked, for grouping.

    An own-file, ancestor-package and dependency refusal of the same construct land in one
    bucket on purpose: the work list orders *constructs* by how many models they block,
    and which file a construct first bit is the JSON's detail, not the headline's.
    """

    def blank(match: re.Match[str]) -> str:
        content = match.group(0)[1:-1]
        keep = "." not in content and len(content) <= _TOKEN_LENGTH
        return match.group(0) if keep else "'·'"

    text = _VIA.sub("", message)
    text = _POSITION.sub("", text)
    text = _QUOTED.sub(blank, text)
    text = _DOTTED.sub("·", text)
    return _NUMBER.sub("·", text).strip()


# -- agreement ------------------------------------------------------------------------


def agreement(reachable: list[Example], library: Library) -> str:
    """The agreement line: measured against omc, or the reason it was not.

    The reference is OpenModelica driven as a subprocess in script mode — never a second
    run of this toolchain. Comparison is left coarse on purpose: final-time states within
    a relative 1e-3, because a finer criterion than the two tools' tolerances is noise.
    """
    if not reachable:
        return "agreeing      0 of 0 simulated — nothing reached the reference"
    omc = shutil.which("omc")
    if omc is None:
        return (
            f"agreeing      unmeasured over {len(reachable)} simulated: OpenModelica "
            "(`omc`) is not on PATH; install it and rerun `moon run :coverage`"
        )
    agreeing = sum(_agrees(example, omc, library) for example in reachable)
    return f"agreeing      {agreeing} of {len(reachable)} simulated match omc at final time"


def _agrees(example: Example, omc: str, library: Library) -> bool:
    script = (
        f'loadModel(Modelica, {{"{MSL_VERSION}"}});\n'
        f"simulate({example.qualified}, stopTime={example.stop}, "
        f'tolerance={example.tolerance}, outputFormat="csv");\n'
    )
    with tempfile.TemporaryDirectory() as scratch:
        mos = Path(scratch) / "run.mos"
        mos.write_text(script)
        try:
            run = subprocess.run(  # noqa: S603 - omc over our own script
                [omc, str(mos)],
                cwd=scratch,
                capture_output=True,
                timeout=SIMULATE_TIMEOUT * 2,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        if run.returncode != 0:
            return False
        produced = list(Path(scratch).glob("*_res.csv"))
        if not produced:
            return False
        reference = _final_row(produced[0])
        from modelica.sim import simulate  # noqa: PLC0415 - same deferral as in drive()

        result = simulate(
            causalize(flatten(library.program, example.qualified)),
            stop=example.stop,
            tolerance=example.tolerance,
        )
        for name, expected in reference.items():
            try:
                ours = float(result[name][-1])
            except (KeyError, IndexError, TypeError):
                continue
            if abs(ours - expected) > 1e-3 * max(abs(expected), 1.0):
                return False
        return True


def _final_row(csv: Path) -> dict[str, float]:
    lines = csv.read_text().strip().splitlines()
    names = [name.strip('"') for name in lines[0].split(",")]
    values = [float(value) for value in lines[-1].split(",")]
    return dict(zip(names, values, strict=False))


# -- report ---------------------------------------------------------------------------


def report(examples: list[Example], library: Library, out: Path | None) -> None:
    total = len(examples)
    reached: dict[str, int] = dict.fromkeys(STAGES, 0)
    for example in examples:
        for stage in STAGES:
            reached[stage] += example.stage in STAGES[STAGES.index(stage) :]
    print()
    print(
        f"coverage over the Modelica Standard Library v{MSL_VERSION} — {total} example models",
    )
    print(
        "(an example: a non-package.mo file under package Modelica whose text carries an "
        "experiment annotation)",
    )
    print()
    for stage in STAGES:
        count = reached[stage]
        marker = "   <- reachable" if stage == "simulate" else ""
        print(f"  {PAST[stage]:<12}{count:>5} / {total}  {100 * count / total:5.1f} %{marker}")
    reachable = [example for example in examples if example.stage == "simulate"]
    print(f"  {agreement(reachable, library)}")
    print()
    _work_list(examples)
    if out is not None:
        payload = {
            "msl": MSL_VERSION,
            "date": datetime.now(tz=UTC).date().isoformat(),
            "summary": {stage: reached[stage] for stage in STAGES} | {"total": total},
            "examples": [
                {
                    "model": example.qualified,
                    "reached": example.stage or "nothing",
                    "error": example.error,
                    "seconds": round(example.seconds, 3),
                }
                for example in examples
            ],
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"per-model detail: {out}")


def _work_list(examples: list[Example]) -> None:
    """The real product: what to build next, ordered by how many models it unblocks."""
    blockers: Counter[tuple[str, str]] = Counter()
    samples: dict[tuple[str, str], str] = {}
    for example in examples:
        if example.stage == "simulate":
            continue
        failed = STAGES[STAGES.index(example.stage) + 1] if example.stage else "parse"
        if example.error.startswith("dependency "):
            failed = "parse"
        key = (failed, normalise(example.error))
        blockers[key] += 1
        samples.setdefault(key, example.qualified)
    print("blockers, by models blocked:")
    for (stage, message), count in blockers.most_common(WORK_LIST_ROWS):
        print(f"  {count:>5}  {stage:<10} {message}")
        print(f"         e.g. {samples[stage, message]}")
    if len(blockers) > WORK_LIST_ROWS:
        rest = len(blockers) - WORK_LIST_ROWS
        print(f"  ... and {rest} rarer refusals; the JSON has every one")


# -- main -----------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write per-model JSON here")
    parser.add_argument("--only", default="", help="substring filter on model names, for debugging")
    parser.add_argument("--limit", type=int, default=0, help="measure only the first N examples")
    arguments = parser.parse_args()

    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))
    sys.setrecursionlimit(20_000)

    msl = fetch()
    if msl is None:
        return 0

    print("parsing the library ...")
    library = assemble(msl)
    refused = sum(1 for file in library.files.values() if file.error)
    print(f"  {len(library.files)} files, {refused} refused to parse")

    examples = enumerate_examples(msl)
    if arguments.only:
        examples = [example for example in examples if arguments.only in example.qualified]
    if arguments.limit:
        examples = examples[: arguments.limit]
    print(f"driving {len(examples)} example models ...")
    for count, example in enumerate(examples, start=1):
        drive(example, library)
        if count % 50 == 0:
            print(f"  {count} / {len(examples)}")

    report(examples, library, arguments.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
