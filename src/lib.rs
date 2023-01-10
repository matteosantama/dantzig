mod error;
mod linalg;
mod linalg2;
mod model;
mod model2;
mod pyobjs;
mod simplex;
mod simplex2;

use crate::error::Error;
use crate::model2::Inequality;
use crate::pyobjs::{PyAffExpr, PyInequality, PySolution, Variable};
use crate::simplex2::Simplex;
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
    m.add_class::<PyAffExpr>()?;
    m.add_class::<PyInequality>()?;
    m.add_class::<PySolution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
