from __future__ import annotations

from typing import Iterable, cast, overload

import dantzig.rust as rs


class Variable:

    _name: str | None
    _variable: rs.Variable

    def __init__(
        self, *, lb: int | float | None, ub: int | float | None, name: str | None = None
    ) -> None:
        self._name = name
        self._variable = rs.Variable(lb=lb, ub=ub)

    @classmethod
    def free(cls, name: str | None = None) -> Variable:
        return cls(lb=None, ub=None, name=name)

    @classmethod
    def nonneg(cls, name: str | None = None) -> Variable:
        return cls(lb=0.0, ub=None, name=name)

    nn = nonneg

    @classmethod
    def nonpos(cls, name: str | None = None) -> Variable:
        return cls(lb=None, ub=0.0, name=name)

    np = nonpos

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def id(self) -> int:
        return cast(int, self._variable.id)

    @property
    def lb(self) -> float | None:
        return cast("float | None", self._variable.lb)

    @property
    def ub(self) -> float | None:
        return cast("float | None", self._variable.ub)

    def to_rust_variable(self) -> rs.Variable:
        return self._variable

    def to_linexpr(self) -> LinExpr:
        return LinExpr.from_rust_variable(self._variable)

    def to_affexpr(self) -> AffExpr:
        return AffExpr.from_rust_variable(self._variable)

    @overload
    def __add__(self, rhs: float | int | AffExpr) -> AffExpr:
        ...

    @overload
    def __add__(self, rhs: Variable | LinExpr) -> LinExpr:
        ...

    def __add__(
        self, rhs: float | int | AffExpr | Variable | LinExpr
    ) -> AffExpr | LinExpr:
        return self.to_linexpr() + rhs

    def __radd__(self, lhs: int | float) -> AffExpr:
        return self + lhs

    @overload
    def __sub__(self, rhs: float | int | AffExpr) -> AffExpr:
        ...

    @overload
    def __sub__(self, rhs: Variable | LinExpr) -> LinExpr:
        ...

    def __sub__(
        self, rhs: float | int | AffExpr | Variable | LinExpr
    ) -> AffExpr | LinExpr:
        return self.to_linexpr() - rhs

    def __rsub__(self, lhs: float | int) -> AffExpr:
        return self.to_linexpr().__neg__() + lhs

    def __mul__(self, rhs: float | int) -> LinExpr:
        if not isinstance(rhs, (int, float)):
            raise TypeError("Variable.__mul__() only supports int and float")
        return self.to_linexpr() * rhs

    def __rmul__(self, lhs: float | int) -> LinExpr:
        return self * lhs

    def __eq__(  # type: ignore[override]
        self, rhs: float | int | Variable | LinExpr | AffExpr
    ) -> Constraint:
        return self.to_affexpr() == rhs

    def __le__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        return self.to_affexpr() <= rhs

    def __ge__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        return self.to_affexpr() >= rhs

    def __neg__(self) -> LinExpr:
        return self.to_linexpr().__neg__()

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Variable(id={self.id}, lb={self.lb}, ub={self.ub})"


class LinExpr:

    _linexpr: rs.LinExpr

    @classmethod
    def reduce(cls, terms: list[tuple[float, rs.Variable]]) -> LinExpr:
        obj = cls.__new__(cls)
        obj._linexpr = rs.LinExpr.py_reduce(terms)
        return obj

    @classmethod
    def from_rust_variable(cls, variable: rs.Variable) -> LinExpr:
        obj = cls.__new__(cls)
        obj._linexpr = rs.LinExpr.py_from_variable(variable)
        return obj

    @classmethod
    def from_rust_linexpr(cls, linexpr: rs.LinExpr) -> LinExpr:
        obj = cls.__new__(cls)
        obj._linexpr = linexpr
        return obj

    def to_rust_linexpr(self) -> rs.LinExpr:
        return self._linexpr

    def to_linexpr(self) -> LinExpr:
        return self

    def to_affexpr(self) -> AffExpr:
        return AffExpr.from_rust_linexpr(self._linexpr)

    def iter_terms(self) -> Iterable[tuple[float, Variable]]:
        yield from self._linexpr.terms

    def iter_variables(self) -> Iterable[Variable]:
        for _, var in self.iter_terms():
            yield var

    def iter_coefs(self) -> Iterable[float]:
        for coef, _ in self.iter_terms():
            yield coef

    @property
    def coefs(self) -> list[float]:
        return list(self.iter_coefs())

    def coef(self, var: Variable) -> float:
        index = self._linexpr.cipher[var.id]
        return cast(float, self._linexpr.terms[index][0])

    @property
    def cipher(self) -> dict[int, int]:
        return cast("dict[int, int]", self._linexpr.cipher)

    @overload
    def __add__(self, rhs: float | int | AffExpr) -> AffExpr:
        ...

    @overload
    def __add__(self, rhs: Variable | LinExpr) -> LinExpr:
        ...

    def __add__(
        self, rhs: float | int | AffExpr | Variable | LinExpr
    ) -> AffExpr | LinExpr:
        if isinstance(rhs, (float, int, AffExpr)):
            return self.to_affexpr() + rhs
        if isinstance(rhs, Variable):
            return self + rhs.to_linexpr()
        if isinstance(rhs, LinExpr):
            return LinExpr.reduce(terms=self._linexpr.terms + rhs._linexpr.terms)
        raise TypeError(f"LinExpr.__add__() does not support {type(rhs)}")

    def __radd__(self, lhs: float | int) -> AffExpr:
        return self + lhs

    @overload
    def __sub__(self, rhs: float | int | AffExpr) -> AffExpr:
        ...

    @overload
    def __sub__(self, rhs: Variable | LinExpr) -> LinExpr:
        ...

    def __sub__(
        self, rhs: float | int | AffExpr | Variable | LinExpr
    ) -> AffExpr | LinExpr:
        if isinstance(rhs, (float, int, AffExpr)):
            return self.to_affexpr() - rhs
        if isinstance(rhs, Variable):
            return self - rhs.to_linexpr()
        if isinstance(rhs, LinExpr):
            return self + -rhs
        raise TypeError(f"LinExpr.__sub__() does not support {type(rhs)}")

    def __rsub__(self, lhs: float | int) -> AffExpr:
        return self.__neg__() + lhs

    def __mul__(self, rhs: float | int) -> LinExpr:
        if not isinstance(rhs, (int, float)):
            raise TypeError("LinExpr.__mul__() only supports int and float")
        return LinExpr.from_rust_linexpr(self._linexpr * rhs)

    def __rmul__(self, lhs: float | int) -> LinExpr:
        return self * lhs

    def __eq__(  # type: ignore[override]
        self, rhs: float | int | Variable | LinExpr | AffExpr
    ) -> Constraint:
        return self.to_affexpr() == rhs

    def __le__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        return self.to_affexpr() <= rhs

    def __ge__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        return self.to_affexpr() >= rhs

    def __neg__(self) -> LinExpr:
        return LinExpr.from_rust_linexpr(self._linexpr.__neg__())

    def __contains__(self, item: Variable) -> bool:
        return item.id in self._linexpr.cipher


class AffExpr:

    _affexpr: rs.AffExpr

    def __init__(self, *, linexpr: LinExpr, constant: int | float) -> None:
        self._affexpr = rs.AffExpr(linexpr=linexpr.to_rust_linexpr(), constant=constant)

    @classmethod
    def from_rust_variable(cls, variable: rs.Variable) -> AffExpr:
        return AffExpr(linexpr=LinExpr.from_rust_variable(variable), constant=0.0)

    @classmethod
    def from_rust_linexpr(cls, linexpr: rs.LinExpr) -> AffExpr:
        return AffExpr(linexpr=LinExpr.from_rust_linexpr(linexpr), constant=0.0)

    def to_rust_affexpr(self) -> rs.AffExpr:
        return self._affexpr

    def to_affexpr(self) -> AffExpr:
        return self

    @property
    def linexpr(self) -> LinExpr:
        return LinExpr.from_rust_linexpr(self._affexpr.linexpr)

    @property
    def constant(self) -> float:
        return cast(float, self._affexpr.constant)

    def __add__(self, rhs: float | int | AffExpr | Variable | LinExpr) -> AffExpr:
        if isinstance(rhs, (int, float)):
            return AffExpr(linexpr=self.linexpr, constant=self.constant + rhs)
        if isinstance(rhs, (Variable, LinExpr)):
            return self + rhs.to_affexpr()
        if isinstance(rhs, AffExpr):
            return AffExpr(
                linexpr=self.linexpr + rhs.linexpr,
                constant=self.constant + rhs.constant,
            )
        raise TypeError(f"AffExpr.__add__() does not support {type(rhs)}")

    def __radd__(self, lhs: float | int) -> AffExpr:
        return self + lhs

    def __sub__(self, rhs: float | int | AffExpr | Variable | LinExpr) -> AffExpr:
        if isinstance(rhs, (int, float)):
            return AffExpr(linexpr=self.linexpr, constant=self.constant - rhs)
        if isinstance(rhs, (Variable, LinExpr)):
            return self - rhs.to_affexpr()
        if isinstance(rhs, AffExpr):
            return AffExpr(
                linexpr=self.linexpr - rhs.linexpr,
                constant=self.constant - rhs.constant,
            )
        raise TypeError(f"AffExpr.__sub__() does not support {type(rhs)}")

    def __rsub__(self, lhs: float | int) -> AffExpr:
        return -self + lhs

    def __mul__(self, rhs: int | float) -> AffExpr:
        if not isinstance(rhs, (int, float)):
            raise TypeError("AffExpr.__mul__() only supports int and float")
        return AffExpr(linexpr=rhs * self.linexpr, constant=rhs * self.constant)

    def __rmul__(self, lhs: int | float) -> AffExpr:
        return self * lhs

    def __eq__(  # type: ignore[override]
        self, rhs: float | int | Variable | LinExpr | AffExpr
    ) -> Constraint:
        affexpr = self - rhs
        return Constraint(
            linexpr=affexpr.linexpr, constant=-affexpr.constant, is_equality=True
        )

    def __le__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        affexpr = self - rhs
        return Constraint(
            linexpr=affexpr.linexpr, constant=-affexpr.constant, is_equality=False
        )

    def __ge__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        affexpr = self - rhs
        return Constraint(
            linexpr=-affexpr.linexpr, constant=affexpr.constant, is_equality=False
        )

    def __neg__(self) -> AffExpr:
        return AffExpr(linexpr=self.linexpr.__neg__(), constant=-self.constant)


class Constraint:

    _constraint: rs.Constraint

    def __init__(self, *, linexpr: LinExpr, constant: float, is_equality: bool) -> None:
        self._constraint = rs.Constraint(
            linexpr=linexpr.to_rust_linexpr(),
            constant=constant,
            is_equality=is_equality,
        )

    def to_rust_constraint(self) -> rs.Constraint:
        return self._constraint
