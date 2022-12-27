from dantzig import exceptions
from dantzig.model import Variable
from dantzig.optimize import Maximize, Minimize

Var = Variable
Min = Minimize
Max = Maximize

__all__ = ["Variable", "Var", "Minimize", "Min", "Maximize", "Max", "exceptions"]
