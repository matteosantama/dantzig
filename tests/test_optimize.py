import dantzig as dz


def test_minimization_problem() -> None:
    x = dz.Variable.nonneg()
    y = dz.Variable.nonneg()

    solution = dz.Minimize(2 * x - 2 * y).subject_to(y == 3).solve()
    assert solution.objective_value == -6.0
    assert solution[x] == 0.0
    assert solution[y] == 3.0

    solution = (
        dz.Minimize(2 * x - 2 * y).subject_to([y <= 5, x >= y + 1, y == 5.0]).solve()
    )
    assert solution.objective_value == 2.0
    assert solution[x] == 6.0
    assert solution[y] == 5.0

    z = dz.Variable.nonneg()

    solution = dz.Min(x + y - z).st(x + y + z <= 1).solve()
    assert solution.objective_value == -1.0
    assert solution[x] == 0.0
    assert solution[y] == 0.0
    assert solution[z] == 1.0

    solution = dz.Min(x + y + z).st(x - y == -2).solve()
    assert solution.objective_value == 2.0
    assert solution[x] == 0.0
    assert solution[y] == 2.0
    assert solution[z] == 0.0


def test_minimization_maximization_equivalence() -> None:
    x = dz.Var.nn()  # type: ignore[call-arg]
    y = dz.Var.nn()  # type: ignore[call-arg]

    min_sol = dz.Min(-x).st(x + y <= 1).solve()
    max_sol = dz.Max(x).st(x + y <= 1).solve()

    assert min_sol.objective_value == -1.0 == -max_sol.objective_value
    assert min_sol[x] == 1.0 == max_sol[x]
    assert min_sol[y] == 0.0 == max_sol[y]


def test_non_standard_variables() -> None:
    x = dz.Var(lb=-2.0, ub=2.0)
    y = dz.Var.free()
    z = dz.Var.np()  # type: ignore[call-arg]

    solution = dz.Min(x + y + z).st([y == 4, -3.0 <= x <= 3.0, z >= -1]).solve()
    assert solution.objective_value == 1.0
    assert solution[x] == -2.0
    assert solution[y] == 4.0
    assert solution[z] == -1.0
