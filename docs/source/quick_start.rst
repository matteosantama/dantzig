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
