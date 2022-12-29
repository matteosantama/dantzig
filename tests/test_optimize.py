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


def test_inventory_balance_problem() -> None:
    p = [0.5, 3.5, 5.0]
    h = [1.0, 5.5, 1.5]
    d = [50, 75, 100]

    d_1, d_2, d_3 = d

    x_1 = dz.Variable.nonneg()
    x_2 = dz.Variable.nonneg()
    x_3 = dz.Variable.nonneg()
    x = [x_1, x_2, x_3]

    z_1 = dz.Variable.nonneg()
    z_2 = dz.Variable.nonneg()
    z_3 = dz.Variable.nonneg()
    z = [z_1, z_2, z_3]

    purchase_cost = sum(p_t * x_t for p_t, x_t in zip(p, x))
    inventory_holding_cost = sum(h_t * z_t for h_t, z_t in zip(h, z))

    assert isinstance(purchase_cost, dz.model.AffExpr)
    assert isinstance(inventory_holding_cost, dz.model.AffExpr)

    total_cost = purchase_cost + inventory_holding_cost

    problem = dz.Minimize(total_cost).subject_to(
        [
            x_1 >= d_1,
            x_2 + z_1 >= d_2,
            x_3 + z_2 >= d_3,
            z_1 == x_1 - d_1,
            z_2 == x_2 + z_1 - d_2,
            z_3 == x_3 + z_2 - d_3,
        ]
    )
    soln = problem.solve()

    assert soln.objective_value == 637.5
    assert soln[x_1] == 125.0
    assert soln[x_2] == 0.0
    assert soln[x_3] == 100.0
