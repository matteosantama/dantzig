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
    fn new(lb: Option<f64>, ub: Option<f64>) -> Self {
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

    pub(crate) fn bounded(lb: f64, ub: f64) -> Self {
        Self::new(Some(lb), Some(ub))
    }
}

pub(crate) struct LinExpr {
    pub(crate) coefs: Vec<f64>,
    pub(crate) vars: Vec<Variable>,
}

impl LinExpr {
    fn split_variables(self, key: &HashMap<usize, (Variable, Variable)>) -> Self {
        let mut coefs = Vec::with_capacity(2 * self.coefs.len());
        let mut vars = Vec::with_capacity(2 * self.vars.len());
        for (coef, var) in self.coefs.into_iter().zip(self.vars) {
            let (pos, neg) = &key[&var.id];
            coefs.push(coef);
            vars.push(pos.clone());
            coefs.push(-coef);
            vars.push(neg.clone())
        }
        Self { coefs, vars }
    }
}

impl From<&[(f64, &Variable)]> for LinExpr {
    fn from(value: &[(f64, &Variable)]) -> Self {
        let coefs = value.iter().map(|v| v.0).collect();
        let vars = value.iter().map(|v| v.1).cloned().collect();
        Self { coefs, vars }
    }
}

pub(crate) struct AffExpr {
    linexpr: LinExpr,
    constant: f64,
}

impl AffExpr {
    pub(crate) fn new(linexpr: &[(f64, &Variable)], constant: f64) -> Self {
        Self {
            linexpr: LinExpr::from(linexpr),
            constant,
        }
    }

    pub(crate) fn constant(&self) -> f64 {
        self.constant
    }

    pub(crate) fn split_variables(self, key: &HashMap<usize, (Variable, Variable)>) -> Self {
        Self {
            linexpr: self.linexpr.split_variables(key),
            ..self
        }
    }

    pub(crate) fn coef(&self, i: usize) -> f64 {
        self.linexpr.coefs[i]
    }

    pub(crate) fn iter_vars(&self) -> impl Iterator<Item = &Variable> {
        self.linexpr.vars.iter()
    }
}

pub(crate) struct Inequality {
    pub(crate) linexpr: LinExpr,
    pub(crate) b: f64,
}

impl Inequality {
    pub(crate) fn new(linexpr: &[(f64, &Variable)], b: f64) -> Self {
        Self {
            linexpr: LinExpr::from(linexpr),
            b,
        }
    }

    pub(crate) fn split_variables(self, key: &HashMap<usize, (Variable, Variable)>) -> Self {
        Self {
            linexpr: self.linexpr.split_variables(key),
            ..self
        }
    }

    pub(crate) fn push_term(&mut self, coef: f64, var: &Variable) {
        self.linexpr.coefs.push(coef);
        self.linexpr.vars.push(var.clone());
    }
}
