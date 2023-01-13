import abc
from typing import Iterable, Literal, TypeVar, cast

import dantzig.rust as rs
from dantzig.model import AffExpr, Constraint, LinExpr, Variable


class Solution:

    _solution: rs.PySolution
    _sense: Literal["minimize", "maximize"]

    def __init__(
        self, *, solution: rs.PySolution, sense: Literal["minimize", "maximize"]
    ) -> None:
        if sense not in ["minimize", "maximize"]:
            raise ValueError("'sense' must be one of ['minimize', 'maximize']")
        self._solution = solution
        self._sense = sense

    @property
    def objective_value(self) -> float:
        if self._sense == "minimize":
            return cast(float, -self._solution.objective_value)
        if self._sense == "maximize":
            return cast(float, self._solution.objective_value)
        raise ValueError("'sense' must be one of ['minimize', 'maximize']")

    def __getitem__(self, variable: Variable) -> float:
        return cast(float, self._solution[variable.to_rust_variable()])


T = TypeVar("T", bound="Optimize")


class Optimize(abc.ABC):

    objective: AffExpr
    constraints: list[Constraint]

    @property
    @abc.abstractmethod
    def sense(self) -> Literal["minimize", "maximize"]:
        raise NotImplementedError

    def __init__(self, objective: Variable | LinExpr | AffExpr) -> None:
        self.objective = objective.to_affexpr()
        self.constraints = []

    def subject_to(self: T, constraints: Constraint | list[Constraint]) -> T:
        """Add constraints to the problem.

        Parameters
        ----------
        constraints
            A single constraint, or a list of multiple constraints, to add to the model.

        Returns
        -------
        T
            An instance of ``self`` so you can chain calls to ``subject_to``.
        """
        if isinstance(constraints, list):
            self.constraints.extend(constraints)
        elif isinstance(constraints, Constraint):
            self.constraints.append(constraints)
        else:
            raise TypeError(f"unexpected constraint type {type(constraints)}")
        return self

    st = subject_to

    def yield_rust_inequalities(self) -> Iterable[rs.PyInequality]:
        for constraint in self.constraints:
            yield from constraint.rust_inequalities()

    @abc.abstractmethod
    def solve(self) -> Solution:
        """Solve the problem."""
        raise NotImplementedError


class Minimize(Optimize):
    """
    Model a minimization problem.

    Parameters
    ----------
    objective : Variable | LinExpr | AffExpr
        The objective function to be minimized.

    Examples
    --------
    >>> import dantzig as dz
    >>>
    >>> x = dz.Variable(lb=1.0, ub=None)
    >>> y = dz.Variable(lb=None, ub=2.0)
    >>>
    >>> result = dz.Minimize(x - 5 * y).solve()
    >>> assert result[x] == 1.0
    >>> assert result[y] == 2.0

    Notes
    -----
    In general, a user will not have to worry about constructing a ``LinExpr``
    or an ``AffExpr``. They will be constructed automatically through linear
    operations on ``Variable`` objects.
    """

    @property
    def sense(self) -> Literal["minimize", "maximize"]:
        return "minimize"

    def solve(self) -> Solution:
        objective = self.objective.__neg__().to_rust_affexpr()
        constraints = list(self.yield_rust_inequalities())
        return Solution(solution=rs.solve(objective, constraints), sense=self.sense)


class Maximize(Optimize):
    """
    Model a maximization problem.

    Parameters
    ----------
    objective : Variable | LinExpr | AffExpr
        The objective function to be maximized.

    Examples
    --------
    >>> import dantzig as dz
    >>>
    >>> x = dz.Variable(lb=1.0, ub=None)
    >>> y = dz.Variable(lb=None, ub=2.0)
    >>>
    >>> result = dz.Maximize(y - 5 * x).solve()
    >>> assert result[x] == 1.0
    >>> assert result[y] == 2.0

    Notes
    -----
    In general, a user will not have to worry about constructing a ``LinExpr``
    or an ``AffExpr``. They will be constructed automatically through linear
    operations on ``Variable`` objects.
    """

    @property
    def sense(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def solve(self) -> Solution:
        objective = self.objective.to_rust_affexpr()
        constraints = list(self.yield_rust_inequalities())
        return Solution(solution=rs.solve(objective, constraints), sense=self.sense)
