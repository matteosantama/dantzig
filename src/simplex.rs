#![allow(dead_code)]
use crate::error::Error;
use crate::linalg::{lu_solve, CscMatrix, Matrix};
use crate::model::{AffExpr, Inequality, LinExpr};
use crate::pyobjs::Variable;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};

const EPSILON: f64 = 1e-12;

struct Equality {
    linexpr: LinExpr,
    b: f64,
    slack: Variable,
}

/// Turn an Inequality into an Equality by injecting a non-negative slack
/// variable.
impl From<Inequality> for Equality {
    fn from(mut inequality: Inequality) -> Self {
        let slack = Variable::nonneg();

        inequality.push_term(1.0, &slack);

        Self {
            linexpr: inequality.linexpr,
            b: inequality.b,
            slack,
        }
    }
}

struct Objective {
    coefficients: Vec<f64>,
    constant: f64,
}

impl Objective {
    fn new(objective: AffExpr, id_to_index: &HashMap<usize, usize>) -> Self {
        let mut coefficients = vec![0.0; id_to_index.len()];
        for (i, var) in objective.iter_vars().enumerate() {
            coefficients[id_to_index[&var.id]] = objective.coef(i);
        }
        Self {
            coefficients,
            constant: objective.constant(),
        }
    }
}

fn chain_variable_ids<'a>(
    objective: &'a AffExpr,
    constraints: &'a [Equality],
) -> impl Iterator<Item = usize> + 'a {
    objective.iter_vars().map(|var| var.id).chain(
        constraints
            .iter()
            .flat_map(|equality| equality.linexpr.vars.iter().map(|var| var.id)),
    )
}

fn sparsify(constraints: Vec<Equality>, id_to_index: &HashMap<usize, usize>) -> CscMatrix {
    let m = constraints.len();
    let n = id_to_index.len();

    let coords = constraints
        .into_iter()
        .map(|c| c.linexpr)
        .enumerate()
        .flat_map(|(i, linexpr)| {
            linexpr
                .vars
                .into_iter()
                .map(|var| id_to_index[&var.id])
                .zip(linexpr.coefs)
                .map(move |(j, coef)| (i, j, coef))
        })
        .collect::<Vec<(usize, usize, f64)>>();

    Matrix::coords(m, n, &coords).to_sparse()
}

/// https://dl.icdst.org/pdfs/files3/faa54c1f53965a11b03f9a13b023f9b2.pdf
pub(crate) struct Simplex {
    objective: Objective,
    constraints: CscMatrix,

    // Indices of the basic and non-basic variables, respectively.
    b: Vec<usize>,
    n: Vec<usize>,

    // Map a variable index to the index within `b` or `n`.
    b_key: HashMap<usize, usize>,
    n_key: HashMap<usize, usize>,

    // Map an original variable ID to the IDs of its positive and negative
    // parts, respectively.
    key: HashMap<usize, (Variable, Variable)>,

    // If `index_to_id[i] == k`, then variable `i` (identified by position) has ID `k`.
    // It should also always be the case that `id_to_index[index_to_id[k]] == k`.
    index_to_id: Vec<usize>,
    id_to_index: HashMap<usize, usize>,

    // These are the "solution vectors" corresponding to the primal and dual problems.
    x: Vec<f64>,
    z: Vec<f64>,

    x_bar: Vec<f64>,
    z_bar: Vec<f64>,
}

impl Debug for Simplex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let x = &self.x;
        let z = &self.z;
        write!(f, "Simplex {{ {x:?} {z:?} }}")
    }
}

impl Simplex {
    pub(crate) fn new(objective: AffExpr, constraints: Vec<Inequality>) -> Self {
        let mut extra_constraints = vec![];
        let mut key = HashMap::new();
        objective
            .iter_vars()
            .chain(
                constraints
                    .iter()
                    .flat_map(|inequality| inequality.linexpr.vars.iter()),
            )
            .for_each(|var| {
                if let Entry::Vacant(e) = key.entry(var.id) {
                    // TODO: We don't necessarily have to introduce TWO new variables for every
                    //  original variable. In many (most?) cases we can get away with a single
                    //  positive or negative part. This would help both speed and numerical stability.
                    let pos = Variable::nonneg();
                    let neg = Variable::nonneg();

                    if let Some(ub) = var.ub {
                        let constraint = Inequality::new(&[(1.0, &pos), (-1.0, &neg)], ub);
                        extra_constraints.push(constraint)
                    }
                    if let Some(lb) = var.lb {
                        let constraint = Inequality::new(&[(-1.0, &pos), (1.0, &neg)], -lb);
                        extra_constraints.push(constraint)
                    }
                    e.insert((pos, neg));
                }
            });

        let objective = objective.split_variables(&key);
        let constraints = constraints
            .into_iter()
            .map(|inequality| inequality.split_variables(&key))
            .chain(extra_constraints);

        let mut slack_ids = HashMap::new();
        let constraints = constraints
            .map(|inequality| {
                let equality = Equality::from(inequality);
                slack_ids.insert(equality.slack.id, equality.b);
                equality
            })
            .collect::<Vec<_>>();

        let mut index_to_id = Vec::new();
        let mut id_to_index = HashMap::new();

        for id in chain_variable_ids(&objective, &constraints) {
            if let Entry::Vacant(e) = id_to_index.entry(id) {
                e.insert(index_to_id.len());
                index_to_id.push(id);
            }
        }

        let objective = Objective::new(objective, &id_to_index);

        let m = constraints.len();

        let mut b_key = HashMap::with_capacity(m);
        let mut b = Vec::with_capacity(m);
        let mut x = Vec::with_capacity(m);

        let mut n_key = HashMap::with_capacity(index_to_id.len() - m);
        let mut n = Vec::with_capacity(index_to_id.len() - m);
        let mut z = Vec::with_capacity(index_to_id.len() - m);

        for id in &index_to_id {
            let i = id_to_index[id];
            if let Entry::Occupied(entry) = slack_ids.entry(*id) {
                b_key.insert(i, b.len());
                b.push(i);
                x.push(*entry.get())
            } else {
                n_key.insert(i, n.len());
                n.push(i);
                z.push(-objective.coefficients[i]);
            }
        }

        // STEP 5: Prepare the remaining initialization parameters.
        let x_bar = vec![1.0; x.len()];
        let z_bar = vec![1.0; z.len()];

        let constraints = sparsify(constraints, &id_to_index);

        Self {
            objective,
            constraints,
            b,
            n,
            b_key,
            n_key,
            key,
            index_to_id,
            id_to_index,
            x,
            z,
            x_bar,
            z_bar,
        }
    }

    fn solve_for_dx(&self, j: usize, basis_matrix: &CscMatrix) -> Vec<f64> {
        let b = self.constraints.column(j);
        lu_solve(basis_matrix.clone().to_dense(), b)
    }

    fn solve_for_dz(&self, i: usize, basis_matrix: &CscMatrix) -> Vec<f64> {
        let mut e = vec![0.0; basis_matrix.nrows()];
        e[self.b_key[&i]] = 1.0;
        let v = lu_solve(basis_matrix.clone().to_dense().t(), e);
        self.constraints.collect_columns(&self.n).neg_t_dot(v)
    }

    /// Index `j` enters the basis and `i` exits.
    fn swap(&mut self, i: usize, j: usize) {
        assert!(self.b_key.contains_key(&i) && !self.b_key.contains_key(&j));
        assert!(self.n_key.contains_key(&j) && !self.n_key.contains_key(&i));

        self.b[self.b_key[&i]] = j;
        self.n[self.n_key[&j]] = i;

        self.b_key.insert(j, self.b_key[&i]);
        self.n_key.insert(i, self.n_key[&j]);

        assert!(self.b_key.remove(&i).is_some());
        assert!(self.n_key.remove(&j).is_some());
    }

    fn pivot(&mut self, i: usize, j: usize, dx: &[f64], dz: &[f64]) {
        let b_i = self.b_key[&i];
        let n_j = self.n_key[&j];

        let t = safe_divide(self.x[b_i], dx[b_i]);
        let s = safe_divide(self.z[n_j], dz[n_j]);
        let t_bar = safe_divide(self.x_bar[b_i], dx[b_i]);
        let s_bar = safe_divide(self.z_bar[n_j], dz[n_j]);

        pivot(&mut self.x, dx, b_i, t);
        pivot(&mut self.x_bar, dx, b_i, t_bar);
        pivot(&mut self.z, dz, n_j, s);
        pivot(&mut self.z_bar, dz, n_j, s_bar);

        self.swap(i, j)
    }

    fn basis_matrix(&self) -> CscMatrix {
        self.constraints.collect_columns(&self.b)
    }

    fn status(self) -> Status {
        let opt_j = find_first_pivot(&self.n, &self.z, &self.z_bar);
        let opt_i = find_first_pivot(&self.b, &self.x, &self.x_bar);

        match (opt_j, opt_i) {
            (Some(j), Some(i)) => {
                let primal = -self.x[self.b_key[&i]] / self.x_bar[self.b_key[&i]];
                let dual = -self.z[self.n_key[&j]] / self.z_bar[self.n_key[&j]];

                match primal <= EPSILON && dual <= EPSILON {
                    true => Status::Optimal(self),
                    false => {
                        let step = match primal < dual {
                            true => Mu::primal_step(self, j, dual),
                            false => Mu::dual_step(self, i, primal),
                        };
                        Status::Suboptimal(step)
                    }
                }
            }
            (Some(j), None) => {
                let len = -self.z[self.n_key[&j]] / self.z_bar[self.n_key[&j]];
                let step = Mu::primal_step(self, j, len);
                Status::Suboptimal(step)
            }
            (None, Some(i)) => {
                let len = -self.x[self.b_key[&i]] / self.x_bar[self.b_key[&i]];
                let step = Mu::dual_step(self, i, len);
                Status::Suboptimal(step)
            }
            (None, None) => panic!("unexpected code path"),
        }
    }

    fn primal_step(mut self, j: usize, mu: f64) -> Result<Self, Error> {
        let basis_matrix = self.basis_matrix();

        let dx = self.solve_for_dx(j, &basis_matrix);
        let i =
            find_second_pivot(mu, &self.x, &self.x_bar, &dx, &self.b).ok_or(Error::Unbounded)?;
        let dz = self.solve_for_dz(i, &basis_matrix);

        self.pivot(i, j, &dx, &dz);
        Ok(self)
    }

    fn dual_step(mut self, i: usize, mu: f64) -> Result<Self, Error> {
        let basis_matrix = self.basis_matrix();

        let dz = self.solve_for_dz(i, &basis_matrix);
        let j =
            find_second_pivot(mu, &self.z, &self.z_bar, &dz, &self.n).ok_or(Error::Infeasible)?;
        let dx = self.solve_for_dx(j, &basis_matrix);

        self.pivot(i, j, &dx, &dz);
        Ok(self)
    }

    pub(crate) fn solve(self) -> Result<Self, Error> {
        match self.status() {
            Status::Optimal(simplex) => Ok(simplex),
            Status::Suboptimal(step) => {
                let result = match step {
                    Step::Primal(mu) => mu.simplex.primal_step(mu.index, mu.length)?,
                    Step::Dual(mu) => mu.simplex.dual_step(mu.index, mu.length)?,
                };
                result.solve()
            }
        }
    }

    pub(crate) fn objective_value(&self) -> f64 {
        self.objective.constant
            + self
                .b_key
                .iter()
                .map(|(x, y)| self.objective.coefficients[*x] * self.x[*y])
                .sum::<f64>()
    }

    pub(crate) fn solution(&self) -> HashMap<usize, f64> {
        self.key
            .iter()
            .map(|(&id, (pvar, nvar))| {
                let pos = self
                    .b_key
                    .get(&self.id_to_index[&pvar.id])
                    .map(|t| self.x[*t])
                    .unwrap_or(0.0);
                let neg = self
                    .b_key
                    .get(&self.id_to_index[&nvar.id])
                    .map(|t| self.x[*t])
                    .unwrap_or(0.0);
                (id, pos - neg)
            })
            .collect()
    }
}

struct Mu {
    simplex: Simplex,
    index: usize,
    length: f64,
}

impl Mu {
    fn primal_step(simplex: Simplex, j: usize, length: f64) -> Step {
        let mu = Mu {
            simplex,
            index: j,
            length,
        };
        Step::Primal(mu)
    }

    fn dual_step(simplex: Simplex, i: usize, length: f64) -> Step {
        let mu = Mu {
            simplex,
            index: i,
            length,
        };
        Step::Dual(mu)
    }
}

enum Step {
    Primal(Mu),
    Dual(Mu),
}

enum Status {
    Optimal(Simplex),
    Suboptimal(Step),
}

fn pivot(data: &mut [f64], delta: &[f64], index: usize, step_length: f64) {
    data.iter_mut()
        .zip(delta.iter())
        .enumerate()
        .for_each(|(i, (v_i, delta_i))| {
            if i == index {
                *v_i = step_length;
            } else {
                *v_i -= step_length * delta_i;
            }
        });
}

fn find_first_pivot(index_lookup: &[usize], y: &[f64], y_bar: &[f64]) -> Option<usize> {
    assert_eq!(index_lookup.len(), y.len());
    assert_eq!(index_lookup.len(), y_bar.len());
    index_lookup
        .iter()
        .zip(y)
        .zip(y_bar)
        .filter(|((_, _), y_bar_k)| **y_bar_k > 0.0)
        .map(|((&k, &y_k), &y_bar_k)| (k, -y_k / y_bar_k))
        .reduce(|(max_k, max_ratio), (k, ratio)| match ratio > max_ratio {
            true => (k, ratio),
            false => (max_k, max_ratio),
        })
        .map(|(k, _)| k)
}

fn find_second_pivot(
    mu: f64,
    y: &[f64],
    y_bar: &[f64],
    dy: &[f64],
    index_lookup: &[usize],
) -> Option<usize> {
    assert_eq!(y.len(), y_bar.len());
    assert_eq!(y.len(), dy.len());
    assert_eq!(y.len(), index_lookup.len());
    y.iter()
        .zip(y_bar)
        .map(|(y_k, y_bar_k)| y_k + mu * y_bar_k)
        .zip(dy)
        .map(|(denominator, dy_k)| *dy_k / denominator)
        .enumerate()
        .filter(|(_, ratio)| *ratio > 0.0)
        .reduce(|(max_k, max_ratio), (k, ratio)| match ratio > max_ratio {
            true => (k, ratio),
            false => (max_k, max_ratio),
        })
        .map(|(k, _)| index_lookup[k])
}

/// Compute `x / y`, with `0 / 0 = 0`.
fn safe_divide(x: f64, y: f64) -> f64 {
    let div = if x == 0.0 && y == 0.0 { 0.0 } else { x / y };
    assert!(!div.is_infinite() && !div.is_nan(), "safe divide {x} / {y}");
    div
}

#[cfg(test)]
mod tests {
    use crate::error::Error;
    use crate::model::{AffExpr, Inequality};
    use crate::pyobjs::Variable;
    use crate::simplex::{Simplex, EPSILON};

    fn assert_approx_eq(result: f64, expected: f64) {
        assert!(
            (result - expected).abs() <= EPSILON,
            "result={result}, expected={expected}"
        )
    }

    #[test]
    fn test_nonneg_1() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(4.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x), (-1.0, &y)], 1.0);
        let c_2 = Inequality::new(&[(2.0, &x), (-1.0, &y)], 3.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 5.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 31.0);
        assert_approx_eq(solution[&x.id], 4.0);
        assert_approx_eq(solution[&y.id], 5.0);
    }

    #[test]
    fn test_nonneg_2() {
        let x_1 = Variable::nonneg();
        let x_2 = Variable::nonneg();
        let x_3 = Variable::nonneg();

        let objective = AffExpr::new(&[(5.0, &x_1), (4.0, &x_2), (3.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(2.0, &x_1), (3.0, &x_2), (1.0, &x_3)], 5.0);
        let c_2 = Inequality::new(&[(4.0, &x_1), (1.0, &x_2), (2.0, &x_3)], 11.0);
        let c_3 = Inequality::new(&[(3.0, &x_1), (4.0, &x_2), (2.0, &x_3)], 8.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 13.0);
        assert_approx_eq(solution[&x_1.id], 2.0);
        assert_approx_eq(solution[&x_2.id], 0.0);
        assert_approx_eq(solution[&x_3.id], 1.0);
    }

    #[test]
    fn test_nonneg_3() {
        // LP relaxation of the problem on page C-10
        // http://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf
        let x_1 = Variable::nonneg();
        let x_2 = Variable::nonneg();
        let x_3 = Variable::nonneg();
        let x_4 = Variable::nonneg();

        let objective = AffExpr::new(
            &[(300.0, &x_1), (90.0, &x_2), (400.0, &x_3), (150.0, &x_4)],
            0.0,
        );
        let c_1 = Inequality::new(
            &[
                (35_000.0, &x_1),
                (10_000.0, &x_2),
                (25_000.0, &x_3),
                (90_000.0, &x_4),
            ],
            120_000.0,
        );
        let c_2 = Inequality::new(&[(4.0, &x_1), (2.0, &x_2), (7.0, &x_3), (3.0, &x_4)], 12.0);
        let c_3 = Inequality::new(&[(1.0, &x_1), (1.0, &x_2)], 1.0);
        let c_4 = Inequality::new(&[(1.0, &x_1)], 1.0);
        let c_5 = Inequality::new(&[(1.0, &x_2)], 1.0);
        let c_6 = Inequality::new(&[(1.0, &x_3)], 1.0);
        let c_7 = Inequality::new(&[(1.0, &x_4)], 1.0);
        let constraints = vec![c_1, c_2, c_3, c_4, c_5, c_6, c_7];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 750.0);
        assert_approx_eq(solution[&x_1.id], 1.0);
        assert_approx_eq(solution[&x_2.id], 0.0);
        assert_approx_eq(solution[&x_3.id], 1.0);
        assert_approx_eq(solution[&x_4.id], 1.0 / 3.0);
    }

    #[test]
    fn test_nonneg_4() {
        let x_1 = Variable::nonneg();
        let x_2 = Variable::nonneg();
        let x_3 = Variable::nonneg();

        let objective = AffExpr::new(&[(10.0, &x_1), (12.0, &x_2), (12.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x_1), (2.0, &x_2), (2.0, &x_3)], 20.0);
        let c_2 = Inequality::new(&[(2.0, &x_1), (1.0, &x_2), (2.0, &x_3)], 20.0);
        let c_3 = Inequality::new(&[(2.0, &x_1), (2.0, &x_2), (1.0, &x_3)], 20.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 136.0);
        assert_approx_eq(solution[&x_1.id], 4.0);
        assert_approx_eq(solution[&x_2.id], 4.0);
        assert_approx_eq(solution[&x_3.id], 4.0);
    }

    #[test]
    fn test_nonneg_5() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(-1.0, &x), (-1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
        let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
        let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), -7.0);
        assert_approx_eq(solution[&x.id], 7.0);
        assert_approx_eq(solution[&y.id], 0.0);
    }

    #[test]
    fn test_nonneg_6() {
        let x_1 = Variable::nonneg();
        let x_2 = Variable::nonneg();
        let x_3 = Variable::nonneg();

        let objective = AffExpr::new(&[(-10.0, &x_1), (-12.0, &x_2), (-12.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(-1.0, &x_1), (-2.0, &x_2), (-2.0, &x_3)], -20.0);
        let c_2 = Inequality::new(&[(-2.0, &x_1), (-1.0, &x_2), (-2.0, &x_3)], -20.0);
        let c_3 = Inequality::new(&[(-2.0, &x_1), (-2.0, &x_2), (-1.0, &x_3)], -20.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), -136.0);
        assert_approx_eq(solution[&x_1.id], 4.0);
        assert_approx_eq(solution[&x_2.id], 4.0);
        assert_approx_eq(solution[&x_3.id], 4.0);
    }

    #[test]
    fn test_nonneg_8() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(-2.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-1.0, &x), (1.0, &y)], -1.0);
        let c_2 = Inequality::new(&[(-1.0, &x), (-2.0, &y)], -2.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 1.0);
        let constraints = vec![c_1, c_2, c_3];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), -1.0);
        assert_approx_eq(solution[&x.id], 2.0);
        assert_approx_eq(solution[&y.id], 1.0);
    }

    #[test]
    fn test_nonneg_9() {
        let x_1 = Variable::nonneg();
        let x_2 = Variable::nonneg();
        let x_3 = Variable::nonneg();
        let x_4 = Variable::nonneg();
        let x_5 = Variable::nonneg();
        let x_6 = Variable::nonneg();

        let objective = AffExpr::new(&[(2.0, &x_2), (3.0, &x_5)], 10.0);
        let c_1 = Inequality::new(&[(1.0, &x_1), (-1.0, &x_2), (1.0, &x_4)], 4.0);
        let c_2 = Inequality::new(&[(-1.0, &x_1), (1.0, &x_2), (-1.0, &x_4)], -4.0);
        let c_3 = Inequality::new(&[(3.0, &x_2), (1.0, &x_3), (-1.0, &x_5)], 12.0);
        let c_4 = Inequality::new(&[(-3.0, &x_2), (-1.0, &x_3), (1.0, &x_5)], -12.0);
        let c_5 = Inequality::new(&[(1.0, &x_2), (1.0, &x_4), (2.0, &x_5)], 14.0);
        let c_6 = Inequality::new(&[(-1.0, &x_2), (-1.0, &x_4), (-2.0, &x_5)], -14.0);
        let c_7 = Inequality::new(&[(2.0, &x_2), (1.0, &x_5), (1.0, &x_6)], 13.0);
        let c_8 = Inequality::new(&[(-2.0, &x_2), (-1.0, &x_5), (-1.0, &x_6)], -13.0);
        let constraints = vec![c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 33.0);
        assert_approx_eq(solution[&x_1.id], 8.0);
        assert_approx_eq(solution[&x_2.id], 4.0);
        assert_approx_eq(solution[&x_3.id], 5.0);
        assert_approx_eq(solution[&x_4.id], 0.0);
        assert_approx_eq(solution[&x_5.id], 5.0);
        assert_approx_eq(solution[&x_6.id], 0.0);
    }

    #[test]
    fn test_nonneg_no_constraints() {
        let x = Variable::nonneg();

        let objective = AffExpr::new(&[(-3.0, &x)], 2.0);
        let constraints = vec![];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();

        assert_approx_eq(simplex.objective_value(), 2.0);
        assert_approx_eq(simplex.solution()[&x.id], 0.0);
    }

    #[test]
    fn test_variable_constraints() {
        let x = Variable::new(Some(1.0), Some(1.0));
        let y = Variable::new(Some(-3.0), Some(-1.0));

        let objective = AffExpr::new(&[(1.0, &x), (-1.0, &y)], 5.0);
        let constraints = vec![];

        let simplex = Simplex::new(objective, constraints).solve().unwrap();
        let solution = simplex.solution();

        assert_approx_eq(simplex.objective_value(), 9.0);
        assert_approx_eq(solution[&x.id], 1.0);
        assert_approx_eq(solution[&y.id], -3.0);
    }

    #[test]
    fn test_unbounded_1() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(-1.0, &x), (4.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
        let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
        let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
        let constraints = vec![c_1, c_2, c_3];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => (),
            Error::Infeasible => panic!("problem should be unbounded"),
        }
    }

    #[test]
    fn test_unbounded_2() {
        let x = Variable::nonneg();

        let objective = AffExpr::new(&[(1.0, &x)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x)], -4.0);
        let constraints = vec![c_1];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => (),
            Error::Infeasible => panic!("problem should be unbounded"),
        }
    }

    #[test]
    fn test_unbounded_no_constraints() {
        let x = Variable::nonneg();

        let objective = AffExpr::new(&[(1.0, &x)], 10.0);
        let constraints = vec![];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => (),
            Error::Infeasible => panic!("problem should be unbounded"),
        }
    }

    #[test]
    fn test_infeasible_1() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(1.0, &x), (1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x)], -1.0);
        let c_2 = Inequality::new(&[(5.0, &y)], 0.5);
        let constraints = vec![c_1, c_2];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => panic!("problem should be infeasible"),
            Error::Infeasible => (),
        }
    }

    #[test]
    fn test_infeasible_2() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(1.0, &x), (-1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x), (1.0, &y)], -1.0);
        let constraints = vec![c_1];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => panic!("problem should be infeasible"),
            Error::Infeasible => (),
        }
    }

    #[test]
    fn test_infeasible_3() {
        let x = Variable::nonneg();
        let y = Variable::nonneg();

        let objective = AffExpr::new(&[(1.0, &x), (1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x), (1.0, &y)], 1.0);
        let c_2 = Inequality::new(&[(-1.0, &x), (-1.0, &y)], -1.0);
        let c_3 = Inequality::new(&[(1.0, &x), (1.0, &y)], 2.0);
        let c_4 = Inequality::new(&[(-1.0, &x), (-1.0, &y)], -2.0);
        let constraints = vec![c_1, c_2, c_3, c_4];

        match Simplex::new(objective, constraints).solve().unwrap_err() {
            Error::Unbounded => panic!("problem should be infeasible"),
            Error::Infeasible => (),
        }
    }
}
