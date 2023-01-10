use crate::model::LinExpr;
use crate::simplex::Simplex;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct Variable {
    #[pyo3(get)]
    pub(crate) id: usize,
    #[pyo3(get)]
    pub(crate) lb: Option<f64>,
    #[pyo3(get)]
    pub(crate) ub: Option<f64>,
}

#[pymethods]
impl Variable {
    #[new]
    #[args("*", lb, ub)]
    pub(crate) fn new(lb: Option<f64>, ub: Option<f64>) -> Self {
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
            lb,
            ub,
        }
    }
}

impl Variable {
    pub(crate) fn nonneg() -> Self {
        Self::new(Some(0.0), None)
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct PyLinExpr {
    pub(crate) linexpr: LinExpr,
    id_to_index: HashMap<usize, usize>,
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct PyAffExpr {
    pub(crate) pylinexpr: PyLinExpr,
    pub(crate) constant: f64,
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct PyInequality {
    pub(crate) pylinexpr: PyLinExpr,
    pub(crate) b: f64,
}

#[pyclass(module = "dantzig.rust")]
pub(crate) struct PySolution {
    #[pyo3(get)]
    objective_value: f64,
    solution: HashMap<usize, f64>,
}

#[pymethods]
impl PySolution {
    fn __getitem__(&self, variable: &Variable) -> f64 {
        self.solution.get(&variable.id).cloned().unwrap_or(0.0)
    }
}

impl From<Simplex> for PySolution {
    fn from(simplex: Simplex) -> Self {
        Self {
            objective_value: simplex.objective_value(),
            solution: simplex.solution(),
        }
    }
}
