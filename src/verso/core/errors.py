"""Exception hierarchy of the VERSO pipeline.

Callers branch on the exception type, never on the message text. VersoError derives from ValueError so
that code written before the hierarchy existed (``except ValueError``) keeps working.
"""

from collections.abc import Iterable


class VersoError(ValueError):
    """Base class of every error raised by VERSO."""


class ConfigError(VersoError):
    """A run configuration is missing, invalid, or asks for something the chosen mode does not do."""


class NetlistError(VersoError):
    """A netlist is malformed or cannot be expanded (unknown element type, missing switch state, ...)."""


class SpecError(VersoError):
    """A Query Spec cannot be parsed or refers to something that does not exist."""


class SpecSyntaxError(SpecError):
    """An expression of a Query Spec does not follow the grammar; column is 1-based, 0 when unknown."""

    def __init__(self, message: str, column: int = 0) -> None:
        super().__init__(message)
        self.column = column


class PerceptionError(VersoError):
    """A diagram image cannot be read or holds no drawing to recognize."""


class SolverError(VersoError):
    """The solver could not produce a single value for the requested target."""


class NoSolutionError(SolverError):
    """The equations are inconsistent or no root is physical (the answer is "no solution")."""


class UnderdeterminedError(SolverError):
    """The equations do not fix the requested quantity (missing data); parameters lists the parameters the target
    still depends on (keys such as R(R2)), when they are known."""

    def __init__(self, message: str, parameters: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.parameters = list(parameters)


class MultipleSolutionsError(SolverError):
    """Several physical roots give different values of the target."""

    def __init__(self, message: str, values: Iterable[float] = ()) -> None:
        super().__init__(message)
        self.values = list(values)


class IndeterminateSignError(SolverError):
    """The sign of an expression depends on parameter values, so compare/trend cannot decide."""
