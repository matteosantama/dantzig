import pytest

import dantzig as dz


def test_unbounded_error() -> None:
    x = dz.Variable.nonneg()
    with pytest.raises(dz.exceptions.UnboundedError):
        dz.Min(-1.0 * x).solve()


def test_infeasible_error() -> None:
    x = dz.Variable.nonneg()
    y = dz.Variable.nonneg()
    with pytest.raises(dz.exceptions.InfeasibleError):
        dz.Min(x + y).st([x + y == 1, x + y == 2]).solve()
