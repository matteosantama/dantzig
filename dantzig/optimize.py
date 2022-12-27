import abc
from typing import Literal, TypeVar, cast

import dantzig.rust as rs
from dantzig.model import AffExpr, Constraint, LinExpr, Variable


class Solution:

    _solution: rs.Solution
    _sense: Literal["minimize", "maximize"]

    def __init__(
        self, *, solution: rs.Solution, sense: Literal["minimize", "maximize"]
    ) -> None:
        if sense not in ["minimize", "maximize"]:
            raise ValueError("'sense' must be one of ['minimize', 'maximize']")
        self._solution = solution
        self._sense = sense

    @property
    def objective_value(self) -> float:
        if self._sense == "minimize":
            return cast(float, self._solution.objective_value)
        if self._sense == "maximize":
            return cast(float, -self._solution.objective_value)
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
        if isinstance(constraints, list):
            self.constraints.extend(constraints)
        elif isinstance(constraints, Constraint):
            self.constraints.append(constraints)
        else:
            raise TypeError(f"unexpected constraint type {type(constraints)}")
        return self

    st = subject_to

    @abc.abstractmethod
    def solve(self) -> Solution:
        raise NotImplementedError


class Minimize(Optimize):
    @property
    def sense(self) -> Literal["minimize", "maximize"]:
        return "minimize"

    def solve(self) -> Solution:
        objective = self.objective.to_rust_affexpr()
        constraints = [c.to_rust_constraint() for c in self.constraints]
        return Solution(solution=rs.solve(objective, constraints), sense=self.sense)


class Maximize(Optimize):
    @property
    def sense(self) -> Literal["minimize", "maximize"]:
        return "maximize"

    def solve(self) -> Solution:
        objective = self.objective.__neg__().to_rust_affexpr()
        constraints = [c.to_rust_constraint() for c in self.constraints]
        return Solution(solution=rs.solve(objective, constraints), sense=self.sense)
