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

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
pub struct LinExpr {
    /// An ordered sequence of Variable (coefficient, ID).
    #[pyo3(get)]
    terms: Vec<(f64, Variable)>,
    /// A mapping from Variable ID to position within LinExpr.terms
    #[pyo3(get)]
    cipher: HashMap<usize, usize>,
}

#[pymethods]
impl LinExpr {
    #[classmethod]
    fn py_reduce(_: &PyType, terms: Vec<(f64, Variable)>) -> Self {
        let mut cipher = HashMap::new();
        let mut consolidated_terms = Vec::new();

        for (coef, var) in terms.iter().cloned() {
            let id = var.id;
            let len = cipher.len();
            if let Entry::Vacant(e) = cipher.entry(id) {
                e.insert(len);
                consolidated_terms.push((coef, var))
            } else {
                let (prev_coef, _) = consolidated_terms[cipher[&id]];
                consolidated_terms[cipher[&id]] = (prev_coef + coef, var)
            }
        }

        Self {
            terms: consolidated_terms,
            cipher,
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
            cipher: self.cipher.clone(),
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
            cipher: self.cipher.clone(),
        }
    }
}

impl LinExpr {
    fn new_trusted(terms: Vec<(f64, Variable)>) -> Self {
        let cipher = terms
            .iter()
            .enumerate()
            .map(|(i, term)| (term.1.id, i))
            .collect();
        Self {
            terms: terms.to_vec(),
            cipher,
        }
    }

    fn decompose_variables(self, decomposition: &HashMap<usize, (Variable, Variable)>) -> Self {
        let terms = self
            .terms
            .into_iter()
            .flat_map(|(coef, var)| {
                let (pos_part, neg_part) = decomposition
                    .get(&var.id)
                    .expect("Variable missing from decomposition map");
                [(coef, pos_part.clone()), (-coef, neg_part.clone())]
            })
            .collect();
        Self::new_trusted(terms)
    }

    fn align_to_cipher(&self, cipher: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        let terms = variables
            .iter()
            .cloned()
            .map(|var| match self.cipher.get(&var.id) {
                None => 0.0,
                Some(&j) => self.terms[j].0,
            })
            .zip(variables.iter().cloned())
            .collect();

        Self {
            terms,
            cipher: cipher.clone(),
        }
    }

    fn coefs(&self) -> Vec<f64> {
        return self.terms.iter().map(|x| x.0).collect();
    }

    fn inject_slack_variable(mut self) -> Self {
        let variable = Variable::standard();
        self.cipher.insert(variable.id, self.cipher.len());
        self.terms.push((1.0, variable));
        self
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(Clone)]
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
    fn decompose_variables(self, decomposition: &HashMap<usize, (Variable, Variable)>) -> Self {
        Self {
            linexpr: self.linexpr.decompose_variables(decomposition),
            ..self
        }
    }

    fn align_to_cipher(self, cipher: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        Self {
            linexpr: self.linexpr.align_to_cipher(cipher, variables),
            ..self
        }
    }
}

#[pyclass(module = "dantzig.rust")]
#[derive(FromPyObject)]
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

    fn decompose_variables(self, decomposition: &HashMap<usize, (Variable, Variable)>) -> Self {
        Self {
            linexpr: self.linexpr.decompose_variables(decomposition),
            ..self
        }
    }

    fn align_to_cipher(&self, cipher: &HashMap<usize, usize>, variables: &[Variable]) -> Self {
        Self {
            linexpr: self.linexpr.align_to_cipher(cipher, variables),
            ..*self
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
    cipher: HashMap<usize, usize>,
    /// Mapping from Variable.id to positive and negative part variables
    decomposition: HashMap<usize, (Variable, Variable)>,
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
        let mut decomposition = HashMap::new();

        for (_, var) in terms {
            if let Entry::Vacant(e) = decomposition.entry(var.id) {
                // TODO: We don't need to necessarily map every variable to two new ones;
                //  for example, we can avoid making "negative parts" for variables that are
                //  non-negative.
                let pos_part = Variable::standard();
                let neg_part = Variable::standard();

                if let Some(ub) = var.ub {
                    let constraint = Constraint::new_inequality_trusted(
                        &[(1.0, pos_part.clone()), (-1.0, neg_part.clone())],
                        ub,
                    );
                    extra_constraints.push(constraint);
                }
                if let Some(lb) = var.lb {
                    let constraint = Constraint::new_inequality_trusted(
                        &[(1.0, neg_part.clone()), (-1.0, pos_part.clone())],
                        -lb,
                    );
                    extra_constraints.push(constraint);
                }
                e.insert((pos_part, neg_part));
            }
        }

        let objective = objective.decompose_variables(&decomposition);
        let constraints = constraints
            .into_iter()
            .map(|c| c.decompose_variables(&decomposition))
            .into_iter()
            .chain(extra_constraints)
            .collect::<Vec<Constraint>>();

        let constraints = constraints
            .into_iter()
            .map(|constraint| constraint.slacken())
            .collect::<Vec<Constraint>>();

        let mut cipher = HashMap::new();
        let mut variables = Vec::new();

        let iter =
            once(objective.linexpr.clone()).chain(constraints.iter().map(|c| c.linexpr.clone()));

        for linexpr in iter {
            for (_, var) in linexpr.terms {
                let len = cipher.len();
                if let Entry::Vacant(e) = cipher.entry(var.id) {
                    e.insert(len);
                    variables.push(var);
                }
            }
        }

        let objective = objective.align_to_cipher(&cipher, &variables);
        let constraints = constraints
            .iter()
            .map(|c| c.align_to_cipher(&cipher, &variables))
            .collect::<Vec<Constraint>>();

        Self {
            objective,
            constraints,
            cipher,
            decomposition,
        }
    }

    pub fn solve(self) -> Result<Solution, Error> {
        let objective = self.objective.linexpr.coefs();
        let constraints = self.constraints.iter().map(|x| x.linexpr.coefs()).collect();
        let rhs = self.constraints.iter().map(|x| x.constant).collect();

        simplex::solve(objective, constraints, rhs).map(|(objective_value, x)| {
            let mut cipher = HashMap::new();
            let mut x_new = Vec::new();

            for (id, (pos_part, neg_part)) in self.decomposition {
                let pos_part_val = x[self.cipher[&pos_part.id]];
                let neg_part_val = x[self.cipher[&neg_part.id]];
                x_new.push(pos_part_val - neg_part_val);
                cipher.insert(id, cipher.len());
            }

            Solution {
                objective_value: objective_value + self.objective.constant,
                x: x_new,
                cipher,
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
    cipher: HashMap<usize, usize>,
}

#[pymethods]
impl Solution {
    fn __getitem__(&self, variable: &Variable) -> f64 {
        self.x[self.cipher[&variable.id]]
    }
}
