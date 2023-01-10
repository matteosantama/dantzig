use crate::pyobjs::{PyAffExpr, PyInequality, PyLinExpr, Variable};
use std::collections::HashMap;

#[derive(Clone)]
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

impl From<PyLinExpr> for LinExpr {
    fn from(value: PyLinExpr) -> Self {
        value.linexpr
    }
}

pub(crate) struct AffExpr {
    linexpr: LinExpr,
    constant: f64,
}

impl AffExpr {
    #[cfg(test)]
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

impl From<PyInequality> for Inequality {
    fn from(value: PyInequality) -> Self {
        Self {
            linexpr: LinExpr::from(value.pylinexpr),
            b: value.b,
        }
    }
}
