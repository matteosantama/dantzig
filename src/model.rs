use crate::error::Error;
use crate::simplex;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::iter::once;
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[pyclass(module = "dantzig.rust")]
#[derive(Clone, Debug)]
pub struct Variable {
    #[pyo3(get)]
    id: usize,
    #[pyo3(get)]
    lb: Option<f64>,
    #[pyo3(get)]
    ub: Option<f64>,
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
    fn standard() -> Self {
        Self::new(Some(0.0), None)
    }
}

struct Sign {
    pos: Variable,
    neg: Variable,
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone, Debug)]
pub struct LinExpr {
    #[pyo3(get)]
    terms: Vec<(f64, Variable)>,
    #[pyo3(get)]
    positions: HashMap<usize, usize>,
}

#[pymethods]
impl LinExpr {
    #[classmethod]
    fn py_reduce(_: &PyType, terms: Vec<(f64, Variable)>) -> Self {
        let mut positions = HashMap::new();
        let mut consolidated_terms = Vec::new();

        for (coef, var) in terms.iter().cloned() {
            let id = var.id;
            let len = positions.len();
            if let Entry::Vacant(e) = positions.entry(id) {
                e.insert(len);
                consolidated_terms.push((coef, var))
            } else {
                let (prev_coef, _) = consolidated_terms[positions[&id]];
                consolidated_terms[positions[&id]] = (prev_coef + coef, var)
            }
        }

        Self {
            terms: consolidated_terms,
            positions,
        }
    }

    #[classmethod]
    fn py_from_variable(_: &PyType, variable: Variable) -> Self {
        Self::new_trusted([(1.0, variable)].to_vec())
    }

    fn __neg__(&self) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .cloned()
                .map(|(coef, var)| (-coef, var))
                .collect(),
            positions: self.positions.clone(),
        }
    }

    fn __mul__(&self, rhs: f64) -> Self {
        Self {
            terms: self
                .terms
                .iter()
                .cloned()
                .map(|(coef, var)| (rhs * coef, var))
                .collect(),
            positions: self.positions.clone(),
        }
    }
}

impl LinExpr {
    fn new_trusted(terms: Vec<(f64, Variable)>) -> Self {
        let positions = terms
            .iter()
            .enumerate()
            .map(|(i, term)| (term.1.id, i))
            .collect();
        Self {
            terms: terms.to_vec(),
            positions,
        }
    }

    fn split(self, signs: &HashMap<usize, Sign>) -> Self {
        let terms = self
            .terms
            .into_iter()
            .flat_map(|(coef, var)| {
                let sign = signs.get(&var.id).expect("Variable missing from sign map");
                [(coef, sign.pos.clone()), (-coef, sign.neg.clone())]
            })
            .collect();
        Self::new_trusted(terms)
    }

    fn align(self, positions: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        let terms = variables
            .iter()
            .cloned()
            .map(|var| match self.positions.get(&var.id) {
                None => 0.0,
                Some(&j) => self.terms[j].0,
            })
            .zip(variables.iter().cloned())
            .collect();

        Self {
            terms,
            positions: positions.clone(),
        }
    }

    fn coefs(&self) -> Vec<f64> {
        return self.terms.iter().map(|x| x.0).collect();
    }

    fn inject_slack_variable(mut self) -> Self {
        let variable = Variable::standard();
        self.positions.insert(variable.id, self.positions.len());
        self.terms.push((1.0, variable));
        self
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone, Debug)]
pub struct AffExpr {
    #[pyo3(get)]
    linexpr: LinExpr,
    #[pyo3(get)]
    constant: f64,
}

#[pymethods]
impl AffExpr {
    #[new]
    fn new(linexpr: LinExpr, constant: f64) -> Self {
        Self { linexpr, constant }
    }
}

impl AffExpr {
    fn split(self, signs: &HashMap<usize, Sign>) -> Self {
        Self {
            linexpr: self.linexpr.split(signs),
            ..self
        }
    }

    fn align(self, positions: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        Self {
            linexpr: self.linexpr.align(positions, variables),
            ..self
        }
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone, Debug)]
pub struct Constraint {
    #[pyo3(get)]
    linexpr: LinExpr,
    #[pyo3(get)]
    constant: f64,
    #[pyo3(get)]
    is_equality: bool,
}

#[pymethods]
impl Constraint {
    #[new]
    fn new(linexpr: LinExpr, constant: f64, is_equality: bool) -> Self {
        Self {
            linexpr,
            constant,
            is_equality,
        }
    }
}

impl Constraint {
    fn new_inequality_trusted(terms: &[(f64, Variable)], constant: f64) -> Self {
        let linexpr = LinExpr::new_trusted(terms.to_vec());
        Constraint::new(linexpr, constant, false)
    }

    fn split(self, signs: &HashMap<usize, Sign>) -> Self {
        Self {
            linexpr: self.linexpr.split(signs),
            ..self
        }
    }

    fn align(self, positions: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        Self {
            linexpr: self.linexpr.align(positions, variables),
            ..self
        }
    }

    fn slacken(self) -> Self {
        match self.is_equality {
            true => self,
            false => Self {
                linexpr: self.linexpr.inject_slack_variable(),
                constant: self.constant,
                is_equality: true,
            },
        }
    }
}

pub struct StandardForm {
    objective: AffExpr,
    constraints: Vec<Constraint>,
    positions: HashMap<usize, usize>,
    signs: HashMap<usize, Sign>,
}

impl StandardForm {
    /// Accomplishes four main objectives:
    ///
    ///     1. Align all linear expressions so they are the same length.
    ///     2. Convert all inequality constraints to equality constraints, introducing
    ///         slack variables as needed.
    ///     3. Map all non-standard variables to the difference of two standard variables,
    ///         adding additional constraints as needed.
    ///     4. TODO: Negates any constraint with a negative constant so that all constants are
    ///         non-negative.
    ///
    pub fn standardize(objective: AffExpr, constraints: Vec<Constraint>) -> Self {
        let terms = once(&objective.linexpr.terms)
            .chain(constraints.iter().map(|c| &c.linexpr.terms))
            .flatten();

        let mut extra_constraints = Vec::new();
        let mut signs = HashMap::new();

        for (_, var) in terms {
            if let Entry::Vacant(e) = signs.entry(var.id) {
                // TODO: We don't need to necessarily map every variable to two new ones;
                //  for example, we can avoid making "negative parts" for variables that are
                //  non-negative.
                let pos = Variable::standard();
                let neg = Variable::standard();

                if let Some(ub) = var.ub {
                    let constraint = Constraint::new_inequality_trusted(
                        &[(1.0, pos.clone()), (-1.0, neg.clone())],
                        ub,
                    );
                    extra_constraints.push(constraint);
                }
                if let Some(lb) = var.lb {
                    let constraint = Constraint::new_inequality_trusted(
                        &[(1.0, neg.clone()), (-1.0, pos.clone())],
                        -lb,
                    );
                    extra_constraints.push(constraint);
                }
                e.insert(Sign { pos, neg });
            }
        }

        let objective = objective.split(&signs);
        let constraints = constraints
            .into_iter()
            .map(|c| c.split(&signs))
            .chain(extra_constraints)
            .collect::<Vec<Constraint>>();

        let constraints = constraints
            .into_iter()
            .map(|constraint| constraint.slacken())
            .collect::<Vec<Constraint>>();

        let mut positions = HashMap::new();
        let mut variables = Vec::new();

        let iter =
            once(objective.linexpr.clone()).chain(constraints.iter().map(|c| c.linexpr.clone()));

        for linexpr in iter {
            for (_, var) in linexpr.terms {
                let len = positions.len();
                if let Entry::Vacant(e) = positions.entry(var.id) {
                    e.insert(len);
                    variables.push(var);
                }
            }
        }

        let objective = objective.align(&positions, &variables);
        let constraints = constraints
            .into_iter()
            .map(|c| c.align(&positions, &variables))
            .collect::<Vec<Constraint>>();

        Self {
            objective,
            constraints,
            positions,
            signs,
        }
    }

    pub(crate) fn solve(self) -> Result<Solution, Error> {
        let objective = self.objective.linexpr.coefs();
        let constraints = self.constraints.iter().map(|x| x.linexpr.coefs()).collect();
        let rhs = self.constraints.iter().map(|x| x.constant).collect();

        simplex::solve(objective, constraints, rhs).map(|(objective_value, x)| {
            let mut positions = HashMap::new();
            let mut x_new = Vec::new();

            for (id, sign) in self.signs {
                let pos = x[self.positions[&sign.pos.id]];
                let neg = x[self.positions[&sign.neg.id]];
                x_new.push(pos - neg);
                positions.insert(id, positions.len());
            }

            Solution {
                objective_value: objective_value + self.objective.constant,
                x: x_new,
                positions,
            }
        })
    }
}

#[pyclass(module = "dantzig.rust")]
pub struct Solution {
    #[pyo3(get)]
    objective_value: f64,
    #[pyo3(get)]
    x: Vec<f64>,
    #[pyo3(get)]
    positions: HashMap<usize, usize>,
}

#[pymethods]
impl Solution {
    fn __getitem__(&self, variable: &Variable) -> f64 {
        self.x[self.positions[&variable.id]]
    }
}
