from __future__ import annotations

from typing import cast, overload

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

    _linexpr: rs.PyLinExpr

    def __init__(self, *, linexpr: rs.PyLinExpr) -> None:
        self._linexpr = linexpr

    @classmethod
    def from_rust_variable(cls, variable: rs.Variable) -> LinExpr:
        return cls(linexpr=rs.PyLinExpr(coefs=[1.0], vars=[variable]))

    def to_rust_linexpr(self) -> rs.LinExpr:
        return self._linexpr

    def to_affexpr(self) -> AffExpr:
        return AffExpr(linexpr=self, constant=0.0)

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
            return LinExpr(linexpr=self._linexpr + rhs._linexpr)
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
            return self + rhs.__neg__()
        raise TypeError(f"LinExpr.__sub__() does not support {type(rhs)}")

    def __rsub__(self, lhs: float | int) -> AffExpr:
        return self.__neg__() + lhs

    def __mul__(self, rhs: float | int) -> LinExpr:
        if not isinstance(rhs, (int, float)):
            raise TypeError("LinExpr.__mul__() only supports int and float")
        return LinExpr(linexpr=self._linexpr * rhs)

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
        return LinExpr(linexpr=self._linexpr.__neg__())


class AffExpr:

    _affexpr: rs.AffExpr

    def __init__(self, *, linexpr: LinExpr, constant: int | float) -> None:
        self._affexpr = rs.AffExpr(linexpr=linexpr.to_rust_linexpr(), constant=constant)

    @classmethod
    def from_rust_variable(cls, variable: rs.Variable) -> AffExpr:
        return cls(linexpr=LinExpr.from_rust_variable(variable), constant=0.0)

    def to_rust_affexpr(self) -> rs.AffExpr:
        return self._affexpr

    def to_affexpr(self) -> AffExpr:
        return self

    @property
    def linexpr(self) -> LinExpr:
        return LinExpr(linexpr=self._affexpr.pylinexpr)

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
        return Constraint.equality(
            linexpr=affexpr.linexpr, b=affexpr.constant.__neg__()
        )

    def __le__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        affexpr = self - rhs
        return Constraint.less_than_eq(
            linexpr=affexpr.linexpr, b=affexpr.constant.__neg__()
        )

    def __ge__(self, rhs: float | int | Variable | LinExpr | AffExpr) -> Constraint:
        affexpr = self - rhs
        return Constraint.greater_than_eq(
            linexpr=affexpr.linexpr.__neg__(), b=affexpr.constant
        )

    def __neg__(self) -> AffExpr:
        return AffExpr(linexpr=self.linexpr.__neg__(), constant=-self.constant)


class Constraint:

    _inequality: rs.PyInequality

    def __init__(self, *, inequality: rs.PyInequality) -> None:
        self._inequality = inequality

    @classmethod
    def equality(cls, *, linexpr: LinExpr, b: float | int) -> Constraint:
        slack = Variable.nonneg()
        return cls(inequality=rs.PyInequality(linexpr=linexpr + slack, b=b))

    @classmethod
    def less_than_eq(cls, *, linexpr: LinExpr, b: float | int) -> Constraint:
        return cls(inequality=rs.PyInequality(linexpr=linexpr, b=b))

    @classmethod
    def greater_than_eq(cls, *, linexpr: LinExpr, b: float | int) -> Constraint:
        return cls(inequality=rs.PyInequality(linexpr=linexpr.__neg__(), b=b.__neg__()))

    def to_rust_inequality(self) -> rs.PyInequality:
        return self._inequality
