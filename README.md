# modelica

A pure-Python Modelica toolchain: **parse** Modelica, **flatten** it to a DAE, **simulate**
it, and interoperate with the two Modelica Association exchange standards — **FMI 3.0**
for single components and **SSP 2.0** for systems of them.

> Distribution name is `modelica`; the git repo keeps its historical name `pymodelica`.
> (PyPI's `pymodelica` is held by an abandoned Modelon/JModelica.org upload.)

```bash
pip install modelica          # core: parse + model structure, zero dependencies
pip install "modelica[sim]"   # + numpy/scipy for simulation
```

## Status

Pre-alpha. This is a **full rewrite** of a 2010 Python 2 experiment — none of that code
survives, only the goal. Python 3.11+.

## Why

The Python side of Modelica is split: [FMPy](https://github.com/CATIA-Systems/FMPy) and
[PyFMI](https://github.com/modelon-community/PyFMI) *consume* FMUs that some other tool
compiled, and the tools that actually compile Modelica (Dymola, OpenModelica, JModelica)
are large non-Python programs. Nothing in Python takes Modelica source all the way to a
running simulation, and nothing treats the model — flat equations, the incidence structure,
the index — as an ordinary Python data structure you can inspect and rewrite.

That gap is the project: a readable, dependency-light implementation where every stage is a
value you can hold.

## Layers

Each layer is usable without the ones above it.

| Module | Does |
| --- | --- |
| `modelica.lang` | Modelica source → tokens → AST (hand-written lexer/parser, positions carried for diagnostics) |
| `modelica.ir` | AST → flat model (variables, equations, expression IR), then matching / BLT sorting / index reduction |
| `modelica.sim` | flat model → ODE/DAE problem, solver backends, event (zero-crossing) handling |
| `modelica.fmi` | FMI 3.0 — read and simulate an FMU; export a flat model as one |
| `modelica.ssp` | SSP 2.0 — `.ssd`/`.ssv`/`.ssb` inside a `.ssp`, and a co-simulation master over the components |

See [the architecture notes](docs/architecture.md) for the data model and the order things
get built in.

## Development

```bash
make install-hooks   # once after cloning: pre-commit (auto version bump + ruff), pre-push (full check)
make check           # lint + typecheck + test, exactly what CI gates on
make docs            # build the mkdocs site strictly
```

The toolchain is pinned in `.prototools` (moon + uv) — `proto install` brings up
everything. All task ordering lives in `moon.yml`; CI is a single `moon run :ci`.

## Releasing

There is no manual release step. `pyproject.toml`'s `version` is the single source of
truth:

* the pre-commit hook auto-bumps the patch version whenever `src/**/*.py` changes;
* `scripts/ci-version-guard` fails the pipeline if shipped source moved without a bump
  (the non-bypassable counterpart);
* on `main`, `moon run :release` publishes to PyPI **only if that version isn't there
  yet**, via GitLab OIDC Trusted Publishing — no API token is stored anywhere.

Docs are published to GitLab Pages from the same pipeline.

## License

MIT © Jorge Cardona

## Trademarks

Modelica® is a registered trademark of the [Modelica Association](https://modelica.org/),
as are FMI and SSP. They are used here in their descriptive sense — to identify the
language this software parses and the standards it implements.

**This project is not affiliated with, sponsored by, or endorsed by the Modelica
Association.** It is an independent open-source implementation. For the language itself,
the specification and the Modelica Standard Library, go to
[modelica.org](https://modelica.org/).
