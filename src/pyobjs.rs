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

#[pymethods]
impl PyLinExpr {
    #[new]
    fn new(coefs: Vec<f64>, vars: Vec<Variable>) -> Self {
        let id_to_index = vars
            .iter()
            .enumerate()
            .map(|(i, var)| (var.id, i))
            .collect();
        Self {
            linexpr: LinExpr { coefs, vars },
            id_to_index,
        }
    }

    fn __neg__(&self) -> Self {
        Self {
            linexpr: self.linexpr.clone().__neg__(),
            id_to_index: self.id_to_index.clone(),
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        todo!()
    }

    fn __mul__(&self, constant: f64) -> Self {
        Self {
            linexpr: self.linexpr.clone().__add__(constant),
            id_to_index: self.id_to_index.clone(),
        }
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct PyAffExpr {
    pub(crate) pylinexpr: PyLinExpr,
    pub(crate) constant: f64,
}

#[pymethods]
impl PyAffExpr {
    #[new]
    #[args("*", linexpr, constant)]
    fn new(linexpr: PyLinExpr, constant: f64) -> Self {
        Self {
            pylinexpr: linexpr,
            constant,
        }
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub(crate) struct PyInequality {
    pub(crate) pylinexpr: PyLinExpr,
    pub(crate) b: f64,
}

#[pymethods]
impl PyInequality {
    #[new]
    #[args("*", linexpr, b)]
    fn new(linexpr: PyLinExpr, b: f64) -> Self {
        Self {
            pylinexpr: linexpr,
            b,
        }
    }
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
