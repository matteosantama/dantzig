.. dantzig documentation master file, created by
   sphinx-quickstart on Wed Jan 11 09:58:27 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dantzig: A Rust-powered LP library for Python.
==============================================

Dantzig is a **lightweight** and **concise** linear programming solver suitable for small
and large-scale problems alike.

Dantzig is implemented in both Rust and Python, meaning you get the expressiveness
and flexibility of a Python frontend plus the raw computing speed of a Rust backend.

Dantzig supports

- A solver featuring a parametric self-dual algorithm
- Arbitrarily restricted variables, including completely unrestricted free variables
- ``==``, ``<=``, and ``>=`` constraints
- Both minimization and maximization problems
- A numerically stable LU factorization with partial pivoting routine for robust linear algebra operations
- Memory-efficient sparse matrix representations
- Modern Python type-checking

.. note::

   This project is under active development. While we continue to improve the
   library, please submit feature requests and bug reports.

.. toctree::
   :maxdepth: 2

   quick_start
   api_reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
