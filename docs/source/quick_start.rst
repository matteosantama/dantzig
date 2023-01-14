Quick start
===========

Installation
------------

To use Dantzig, first install it using ``pip``:

.. code-block:: console

    (.venv) $ pip install dantzig


Usage
-----

Dantzig prides itself on being both lightweight (zero-dependency) and concise.
The API is designed to be extremely expressive and terse, saving you keystrokes without
sacrificing clarity. To this end, many common functions have a succinct alias.

.. code-block:: python

    import dantzig as dz

    x = dz.Variable(lb=0.0, ub=None)
    y = dz.Variable(lb=0.0, ub=None)
    z = dz.Variable(lb=0.0, ub=None)

    soln = dz.Minimize(x + y - z).subject_to(x + y + z == 1).solve()

    assert soln.objective_value == -1.0
    assert soln[x] == 0.0
    assert soln[y] == 0.0
    assert soln[z] == 1.0

This code block can alternatively be written

.. code-block:: python

    from dantzig import Min, Var

    x = Var.nn()
    y = Var.nn()
    z = Var.nn()

    soln = Min(x + y - z).st(x + y + z == 1)


Example
-------

In this section, we will solve a classic inventory balance problem.

Suppose you are the manager of a retail store and over the next three time periods you need to order inventory from your supplier to satisfy demand.

- A priori, you know how much demand there will be each time period (an unrealistic assumption to be sure, but our model can be extended to accommodate uncertain demand) denoted by :math:`d_t` for :math:`t \in \{1, 2, 3\}`.
- The supplier charges a different per-unit price :math:`p_t` in each time period.
- You face inventory holding costs :math:`h_t` for any inventory carried over from period :math:`t` to period :math:`t + 1`.
- In each time period, you decide how much inventory to order, given by :math:`x_t`, and then sell to your customers.

Naturally, you seek a policy that minimizes total cost. This problem can be formulated as a linear program:

.. math::

    \begin{align*}
    \min_x \quad & \sum_{t = 1}^3 p_t x_t + \sum_{t = 1}^3 h_t z_t \\
    \text{s.t.} \quad & x_1 \geq d_1 \\
    & x_2 + z_1 \geq d_2 \\
    & x_3 + z_2 \geq d_3 \\
    & z_1 = x_1 - d_1 \\
    & z_2 = x_2 + z_1 - d_2 \\
    & z_3 = x_3 + z_2 - d_3 \\
    & x_t \geq 0 \\
    & z_t \geq 0
    \end{align*}

where :math:`z_t` is an auxiliary variable that represents the amount of inventory you carry from period :math:`t` to period :math:`t + 1`. Notice that the constraints require that demand is fully satisfied, and we never have negative orders or inventory.

.. code-block:: python

    import dantzig as dz

    p = [0.5, 3.5, 5.0]
    h = [1.0, 5.5, 1.5]
    d = [50, 75, 100]

    d_1, d_2, d_3 = d

    x_1 = dz.Var.nn()
    x_2 = dz.Var.nn()
    x_3 = dz.Var.nn()
    x = [x_1, x_2, x_3]

    z_1 = dz.Var.nn()
    z_2 = dz.Var.nn()
    z_3 = dz.Var.nn()
    z = [z_1, z_2, z_3]

    purchase_cost = sum(p_t * x_t for p_t, x_t in zip(p, x))
    inventory_holding_cost = sum(h_t * z_t for h_t, z_t in zip(h, z))
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

The optimal policy is apparently to place orders only in periods one and three.