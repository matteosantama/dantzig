import dantzig as dz
from dantzig.model import AffExpr, LinExpr


def linexprs_equal(x: LinExpr, y: LinExpr) -> bool:
    """Doesn't validate order, only contents."""

    def expand(z: LinExpr) -> dict[int, float]:
        return {var.id: coef for coef, var in z.iter_terms()}

    return expand(x) == expand(y)


def affexprs_equal(x: AffExpr, y: AffExpr) -> bool:
    return linexprs_equal(x.linexpr, y.linexpr) and x.constant == y.constant


def test_linexpr_operations() -> None:
    x = dz.Variable.nonneg()
    y = dz.Variable.nonneg()

    assert linexprs_equal(-x, -1.0 * x)
    assert linexprs_equal(-x, x * -1.0)
    assert linexprs_equal(x + x, 2 * x)
    assert linexprs_equal(x + x, x * 2)
    assert linexprs_equal(x - y, x + -y)
    assert linexprs_equal(x - y, -y + x)
    assert linexprs_equal(x + y + x, 2 * x + y)
    assert linexprs_equal(x + y + x, y + 2 * x)
    assert linexprs_equal(2 * x + 2 * y, 2 * (x + y))
    assert linexprs_equal(2 * x + 2 * y, (x + y) * 2)
    assert linexprs_equal(x * 2 + y * 2, 2 * (x + y))
    assert linexprs_equal(x * 2 + y * 2, (x + y) * 2)
    assert linexprs_equal(-(x + y), -x - y)
    assert linexprs_equal(-(x + y), -y - x)
    assert linexprs_equal(2 * x - x, x.to_linexpr())


def test_affexpr_operations() -> None:
    x = dz.Variable.free()
    y = dz.Variable.free()

    assert affexprs_equal(x + 5.0, 5.0 + x)
    assert affexprs_equal(2 * x + 2, 2 * (x + 1))
    assert affexprs_equal(2 * x + 2, (x + 1) * 2)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 2 * x + 2 * y + 10)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 2 * x + 2 * y + 10.0)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 2 * x + 10 + 2 * y)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 2 * x + 10.0 + 2 * y)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 10 + 2 * x + 2 * y)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 10.0 + 2 * x + 2 * y)
    assert affexprs_equal((x + y + 5) + (x + y + 5), 2 * (x + y + 5))
    assert affexprs_equal((x + y + 5) + (x + y + 5), (x + y + 5) * 2)
    assert affexprs_equal(-(x + y + 1), -1 * (x + y + 1))
    assert affexprs_equal(-(x + y + 1), 0.0 - (x + y + 1))
    assert affexprs_equal(x + y + 1, 0.0 + (x + y + 1))
