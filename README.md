## Dantzig: A Lightweight, Concise LP Solver

Dantzig is a lightweight and concise linear programming solver suitable for small 
and large-scale problems alike.

Dantzig is implemented in both Rust and Python, meaning you get the expressiveness 
and flexibility of a Python frontend plus the raw computing speed of a Rust backend. 

Dantzig supports

- Both minimization and maximization problems
- Arbitrarily restricted variables, including completely unrestricted free variables
- Equality, "less than or equal to", and "greater than or equal to" constraints
- SIMD-accelerated linear algebra operations
- Modern Python type-checking

Dantzig is currently beta software. Please help us improve the library by reporting bugs through GitHub issues. 

### Installation

Dantzig can be installed with pip

```shell
pip install dantzig 
```

for Python >=3.10.

### Design Philosophies

Dantzig prides itself on being both **lightweight** (zero-dependency) and **concise**.
The API is designed to be extremely expressive and terse, without sacrificing clarity. 
To this end, Dantzig provides several short aliases for the most common operations.

A few examples are listed below,

- `Var == Variable`
- `Min == Minimize`
- `Max == Maximize`
- `Var.free() == Var(lb=0.0, ub=0.0)`
- `Var.nn() == Var.nonneg() == Var(lb=0.0, ub=None)`
- `Var.np() == Var.nonpos() == Var(lb=None, ub=0.0)`

and you will find more sprinkled throughout the library.

### Examples

```python
import dantzig as dz

x = dz.Variable(lb=0.0, ub=None)
y = dz.Variable(lb=0.0, ub=None)
z = dz.Variable(lb=0.0, ub=None)

soln = dz.Minimize(x + y - z).subject_to(x + y + z == 1).solve()

assert soln.objective_value == -1.0
assert soln[x] == 0.0
assert soln[y] == 0.0
assert soln[z] == 1.0
```

Using aliases, the previous example can alternately be written

```python
from dantzig import Min, Var

x = Var.np()
y = Var.np()
z = Var.np()

soln = Min(x + y - z).st(x + y + z == 1)
```


### Road Map

- [ ] Mixed integer linear programing
- [ ] Built-in support for multidimensional variables and interoperability with `numpy`
- [ ] More efficient matrix storage, ie. CSR format
- [ ] Code profiling to identify and resolve performance bottlenecks
- [ ] Support for alternate algorithms beyond simplex