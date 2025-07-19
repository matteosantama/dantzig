class UnboundedError(Exception):
    """Raised when the model has an unbounded objective."""


class InfeasibleError(Exception):
    """Raised when the model is infeasible (empty feasible region)."""
