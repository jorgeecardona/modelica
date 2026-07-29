# modelica

Pure-Python Modelica toolchain: parse Modelica, flatten it to a DAE, simulate it, and
exchange models via **FMI 3.0** and **SSP 2.0**.

```bash
pip install modelica          # core: parse + model structure, zero dependencies
pip install "modelica[sim]"   # + numpy/scipy for simulation
```

!!! warning "Pre-alpha"
    A full rewrite of a 2010 prototype, in progress. The API is not stable and the layers
    below are being built bottom-up. Python 3.12+.

## The layers

| Module | Does |
| --- | --- |
| `modelica.lang` | Modelica source → tokens → AST |
| `modelica.ir` | AST → flat model; matching, BLT sorting, index reduction |
| `modelica.sim` | flat model → ODE/DAE problem; solvers and events |
| `modelica.fmi` | FMI 3.0 — import, simulate, and export FMUs |
| `modelica.ssp` | SSP 2.0 — system composition and a co-simulation master |

Start with [Architecture](architecture.md) for the data model and the build order.

## Standards targeted

* **Modelica Language Specification 3.x**
* **FMI 3.0** — [fmi-standard.org](https://fmi-standard.org/)
* **SSP 2.0** — [ssp-standard.org](https://ssp-standard.org/)

## Trademarks

Modelica® is a registered trademark of the [Modelica Association](https://modelica.org/),
as are FMI and SSP. They are used here descriptively, to identify the language this
software parses and the standards it implements.

This project is **not affiliated with, sponsored by, or endorsed by the Modelica
Association** — it is an independent open-source implementation. The language itself, its
specification and the Modelica Standard Library live at
[modelica.org](https://modelica.org/).
