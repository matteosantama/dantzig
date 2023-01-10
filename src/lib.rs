mod error;
mod linalg;
mod model;
mod pyobjs;
mod simplex;

use crate::error::Error;
use crate::model::Inequality;
use crate::pyobjs::{PyAffExpr, PyInequality, PyLinExpr, PySolution, Variable};
use crate::simplex::Simplex;
use pyo3::prelude::*;

pyo3::import_exception!(dantzig.exceptions, UnboundedError);
pyo3::import_exception!(dantzig.exceptions, InfeasibleError);

#[pyfunction]
fn solve(objective: PyAffExpr, constraints: Vec<PyInequality>) -> PyResult<PySolution> {
    let objective = objective.into();
    let constraints = constraints.into_iter().map(Inequality::from).collect();
    Simplex::new(objective, constraints)
        .solve()
        .map(PySolution::from)
        .map_err(|err| match err {
            Error::Unbounded => UnboundedError::new_err("The objective is unbounded"),
            Error::Infeasible => InfeasibleError::new_err("The model is infeasible"),
        })
}

#[pymodule]
fn rust(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Variable>()?;
    m.add_class::<PyLinExpr>()?;
    m.add_class::<PyAffExpr>()?;
    m.add_class::<PyInequality>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
