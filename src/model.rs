use crate::pyobjs::{PyAffExpr, PyInequality, PyLinExpr, Variable};
use std::collections::HashMap;
use std::ops::Neg;

#[derive(Clone, Debug)]
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

    pub(crate) fn __neg__(self) -> Self {
        Self {
            coefs: self.coefs.into_iter().map(|x| -x).collect(),
            ..self
        }
    }

    pub(crate) fn __add__(self, constant: f64) -> Self {
        Self {
            coefs: self.coefs.into_iter().map(|x| constant * x).collect(),
            ..self
        }
    }
}

impl From<&[(f64, &Variable)]> for LinExpr {
    fn from(value: &[(f64, &Variable)]) -> Self {
        let coefs = value.iter().map(|v| v.0).collect();
        let vars = value.iter().map(|v| v.1).cloned().collect();
        Self { coefs, vars }
    }
}

impl From<PyLinExpr> for LinExpr {
    fn from(value: PyLinExpr) -> Self {
        value.linexpr
    }
}

#[derive(Debug)]
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

impl From<PyAffExpr> for AffExpr {
    fn from(value: PyAffExpr) -> Self {
        Self {
            linexpr: LinExpr::from(value.pylinexpr),
            constant: value.constant,
        }
    }
}

impl From<LinExpr> for AffExpr {
    fn from(linexpr: LinExpr) -> Self {
        Self {
            linexpr,
            constant: 0.0,
        }
    }
}

#[derive(Debug)]
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

    pub(crate) fn less_than_eq(linexpr: LinExpr, b: f64) -> Self {
        Self { linexpr, b }
    }

    pub(crate) fn greater_than_eq(linexpr: LinExpr, b: f64) -> Self {
        Self {
            linexpr: linexpr.__neg__(),
            b: b.neg(),
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

impl From<PyInequality> for Inequality {
    fn from(value: PyInequality) -> Self {
        Self {
            linexpr: LinExpr::from(value.pylinexpr),
            b: value.b,
        }
    }
}
