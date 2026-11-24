"""Physical quantities written as text, converted to exact SI values.

The Query Spec writes given values as text with an optional SI prefix and unit, e.g. "50/pi μF" or
"1.5 A". Values are kept exact (SymPy rationals and constants) so the solver stays symbolic.
"""

import re

import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from .errors import SpecError

PREFIX = {
    "p": sp.Rational(1, 10**12),
    "n": sp.Rational(1, 10**9),
    "μ": sp.Rational(1, 10**6),
    "µ": sp.Rational(1, 10**6),
    "m": sp.Rational(1, 1000),
    "k": sp.Integer(1000),
    "M": sp.Integer(10**6),
}
BASE_UNITS = ("Ω", "V", "A", "W", "F", "H", "C", "J", "s", "Hz", "VA", "var", "g")

_QUANTITY = re.compile(r"\s*(.+?)\s*([pnμµmkM]?)(VA|var|Ω|V|A|W|F|H|C|J|s|Hz|g)?\s*")
# Complex values of AC circuits (three-phase sources, impedances): polar "230∠-15°" (angle in degrees) and
# rectangular "2+j8", "4 - j3", "8j".
_POLAR = re.compile(r"(.+?)\s*∠\s*([-+−]?\s*[\d.]+(?:/[\d.]+)?)\s*°?")
_J_BEFORE = re.compile(r"(?<![\w.])j\s*(?=[\d.(])")
_J_AFTER = re.compile(r"(?<=[\d.)])\s*j(?!\w)")
_J_ALONE = re.compile(r"(?<![\w.])j(?![\w.(])")


def _complex_text(text: str) -> str:
    """SymPy text of a number written in polar or rectangular form; other text is returned unchanged."""
    m = _POLAR.fullmatch(text.strip())
    if m:
        angle = m.group(2).replace("−", "-").replace(" ", "")
        return f"({m.group(1)})*exp(I*pi*({angle})/180)"
    text = _J_BEFORE.sub("I*", text)
    text = _J_AFTER.sub("*I", text)
    return _J_ALONE.sub("I", text)


def parse_quantity(text: str) -> sp.Expr:
    """Exact SI value of a quantity written as text: '50/pi μF' -> 50/(10^6·pi); '1.5 A' -> 3/2.

    A prefix letter counts only when a unit follows it: '5 m' reads as 5.
    """
    return split_quantity(text)[0]


def split_quantity(text: str) -> tuple[sp.Expr, str | None]:
    """Exact SI value and base unit of a quantity written as text: '2 kΩ' -> (2000, 'Ω'); '3' -> (3, None)."""
    m = _QUANTITY.fullmatch(text)
    if m is None:
        raise SpecError(f"cannot read quantity {text!r}")
    try:
        value = sp.nsimplify(sp.sympify(_complex_text(m.group(1)), locals={"pi": sp.pi, "I": sp.I}), rational=True)
    except (sp.SympifyError, SyntaxError, TypeError) as ex:
        raise SpecError(f"cannot read quantity {text!r}: {ex}") from ex
    return value * (PREFIX.get(m.group(2), 1) if m.group(3) else 1), m.group(3)


def unit_factor(unit: str | None) -> sp.Expr:
    """Exact factor from a prefixed unit to its SI base unit: 'mA' -> 1/1000, 'kΩ' -> 1000, 'A' -> 1, 'lần' -> 1."""
    unit = (unit or "").strip()
    if len(unit) > 1 and unit[0] in PREFIX and unit[1:] in BASE_UNITS:
        return PREFIX[unit[0]]
    return sp.Integer(1)


def unit_scale(unit: str | None) -> float:
    """unit_factor as a float, for comparing numeric answers."""
    return float(unit_factor(unit))


def read_answer(answer: str, unit: str | None) -> tuple[float, float]:
    """(an answer written as text, in SI units; factor from its unit to SI): "1,5" in "mA" -> (0.0015, 0.001).
    Expressions with implicit multiplication are read too: "20sqrt(5)", "250pi", "(11sqrt(2))/8"."""
    scale = unit_scale(unit)
    text = answer.replace(",", ".")
    try:
        value = sp.nsimplify(text)
    except (sp.SympifyError, SyntaxError, TypeError):
        value = parse_expr(text, local_dict={"pi": sp.pi},
                           transformations=standard_transformations + (implicit_multiplication_application,))
    return float(value) * scale, scale
