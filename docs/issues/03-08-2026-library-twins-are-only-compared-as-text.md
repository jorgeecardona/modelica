---
title: Library twins are compared as normalised text, because a library has no flat model
status: done
area: ir
owner: cute-replicant-6e76d
filed: 2026-08-03
fixed_in: ed73272
kind: story
scope: M
verify: test
---

## Measured

Two checks hold the two front ends together, and they do not cover the same files.

- `tests/test_examples.py:195`, `test_python_twin_says_the_same_thing`: renders the Python twin
  through `modelica.lang.unparse` and diffs it against the `.mo` file after `_normalise`
  (`tests/test_examples.py:144`). It runs over **every** example, libraries included.
- `tests/test_examples.py:239`, `test_front_ends_flatten_to_the_same_model`: flattens both and
  compares the `FlatModel` values. Its docstring calls it "the real equivalence test: not the
  same text, the same mathematical object". It runs over `MODEL_FILES` **only**.

So the four libraries in `examples/library/` (`Electrical`, `Rotational`, `Machines`, `Blocks`)
are held together by text alone. That is measured from the parametrisation of the two tests,
not inferred.

The reason is structural rather than an oversight: a library is a package of classes and has no
flat model of its own. There is nothing to flatten until a model instantiates it.

## Inferred

Why text is the wrong instrument here, inferred from what it can and cannot see: a normalised
diff fails on formatting that means nothing, and passes on differences that mean something and
happen to render the same. It also pins the *printer*. Any change to `modelica.lang.unparse` —
spacing, how a modification is laid out, whether an annotation is emitted — breaks tests about
the corpus rather than tests about the printer.

That last point stopped being hypothetical on branch `adopt-cabildo-issues`, where
`modelica.lang.parse._annotation` changed from discarding annotations to keeping them.
Anything the printer now emits and did not emit before has to be matched by hand in every
Python twin.

Not inferred, and worth stating: the text check does cover things flattening throws away —
class structure, `extends` before it is resolved, annotations, documentation strings. Deleting
it would lose that. This issue is not "delete the text check".

## Done

- Every class in a library is compared as a **value**: parse the `.mo` into `ir.Class` values,
  build the twin through `modelica.build`, and compare the two `ir.Class` trees directly. That
  is the same object both front ends promise to produce (`docs/architecture.md:41`: "Both roads
  end at the same `ir.Class`"), and it is available for libraries where a flat model is not.
- The comparison says which class and which member differs when it fails, not which line of
  rendered text.
- The normalised-text check either goes, or stays with its purpose written at the top of it: if
  it is now a test of `modelica.lang.unparse` rather than of the corpus, it should say so and
  live beside the round-trip tests in `tests/test_lang.py`.

## Deliberately not done

No change to the models' flat-model comparison. It is the stronger check for anything that can
be flattened, and this issue only covers what cannot.

## What landed

`ed73272`. 898 tests pass.

`test_python_twin_builds_the_same_value` parses the `.mo` into an `ir.Class` and compares it
with the one `modelica.build` produces, over libraries and models alike. `_disagreement` walks
the two in step and names the first place they part company —
`Electrical.OnePort.p.annotation`, with both sides printed — rather than dumping two screens
of dataclass.

**It found something on the first run, which is the answer to whether text was the wrong
instrument: six of the twenty-seven examples were not equal.** One cause, in all six.
Modelica has no negative literals — `R = -1` is a unary minus in front of `1`, and that is
what the parser builds — but Python folds the sign before `_lift` ever sees the value, so the
Python front end built `IntLiteral(-1)` against the parser's `Unary("-", IntLiteral(1))`.
Equal in value, unequal as IR, and the printer renders both as `-1`, which is precisely why
the text diff never saw it. `_lift` now undoes the folding, which also makes it agree with
`__neg__`, which was already building `Unary`.

The text check stayed, and stopped being about the twins. It compares each `.mo` against
`unparse(parse(...))` of itself: corpus hygiene, that every example is written in the form the
printer produces. The Python side does not appear in it. That answers the third Done bullet —
it is now a test of the printer against the corpus, it says so, and it no longer breaks every
twin when the printer changes.

## Deliberately not done, found while doing it

- **The text check did not move to `tests/test_lang.py`.** The bullet offered that if it
  became a printer test, and it did — but it needs `LIBRARY_FILES`/`MODEL_FILES` and
  `_normalise`, which live here, and `tests/test_lang.py` already round-trips the same corpus
  through a different property. Two tests over one corpus in two files is worse than one test
  next to its fixtures.
- **`_normalise` still folds numeric spelling.** It is now only comparing the printer's output
  against the file, and `1e4` against `10000.0` is a difference between two ways of writing a
  number rather than between the two things being compared. The *value* check does not fold
  anything, which is where it matters.
