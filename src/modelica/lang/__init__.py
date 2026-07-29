"""Modelica concrete syntax: source text in, :mod:`modelica.ir` values out -- and back.

Both directions live here and are checked against each other. :func:`parse` reads a ``.mo``
file into IR; :func:`unparse` writes IR back out as source, which is what lets a model built
in Python be read by someone who only knows Modelica. The two share one precedence table, so
``parse(unparse(x)) == x`` holds structurally rather than by luck.

The parser covers the language the printer can emit and refuses the rest by name -- arrays,
``for``, ``algorithm``, function bodies, ``redeclare`` -- rather than misreading it.
"""

from modelica.lang.lexer import ModelicaSyntaxError, Token, tokenize
from modelica.lang.parse import parse, parse_expression
from modelica.lang.unparse import unparse, unparse_class, unparse_equation, unparse_expr

__all__ = [
    "ModelicaSyntaxError",
    "Token",
    "parse",
    "parse_expression",
    "tokenize",
    "unparse",
    "unparse_class",
    "unparse_equation",
    "unparse_expr",
]
