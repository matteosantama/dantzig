#![feature(portable_simd, array_chunks)]
mod error;
mod linalg;
mod model;
mod simplex;

use error::Error;
use model::{AffExpr, Constraint, LinExpr, Solution, StandardForm, Variable};

use pyo3::prelude::*;

pyo3::import_exception!(dantzig.exceptions, UnboundedError);
pyo3::import_exception!(dantzig.exceptions, InfeasibleError);

#[pyfunction]
fn solve(objective: AffExpr, constraints: Vec<Constraint>) -> PyResult<Solution> {
    StandardForm::standardize(objective, constraints)
        .solve()
        .map_err(|err| match err {
            Error::Unbounded => UnboundedError::new_err("The objective is unbounded"),
            Error::Infeasible => InfeasibleError::new_err("The model is infeasible"),
        })
}

#[pymodule]
fn rust(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Variable>()?;
    m.add_class::<LinExpr>()?;
    m.add_class::<AffExpr>()?;
    m.add_class::<Constraint>()?;
    m.add_class::<Solution>()?;
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    Ok(())
}
