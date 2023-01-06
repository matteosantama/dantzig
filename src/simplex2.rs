use crate::linalg2::{lu_solve, CscMatrix, Matrix};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct Variable {
    id: usize,
}

impl Variable {
    fn new() -> Self {
        Self {
            id: COUNTER.fetch_add(1, Ordering::Relaxed),
        }
    }
}

struct LinExpr {
    coefs: Vec<f64>,
    vars: Vec<Variable>,
}

impl From<&[(f64, &Variable)]> for LinExpr {
    fn from(value: &[(f64, &Variable)]) -> Self {
        let coefs = value.iter().map(|v| v.0).collect();
        let vars = value.iter().map(|v| v.1).cloned().collect();
        Self { coefs, vars }
    }
}

struct AffExpr {
    linexpr: LinExpr,
    constant: f64,
}

impl AffExpr {
    fn new(linexpr: &[(f64, &Variable)], constant: f64) -> Self {
        Self {
            linexpr: LinExpr::from(linexpr),
            constant,
        }
    }
}

struct Inequality {
    linexpr: LinExpr,
    b: f64,
}

impl Inequality {
    fn slacken(mut self) -> Equality {
        let slack = Variable::new();
        let slack_id = slack.id;

        self.linexpr.vars.push(slack);
        self.linexpr.coefs.push(1.0);

        Equality {
            linexpr: self.linexpr,
            b: self.b,
            slack_id,
        }
    }

    fn new(linexpr: &[(f64, &Variable)], b: f64) -> Self {
        Self {
            linexpr: LinExpr::from(linexpr),
            b,
        }
    }
}

struct Equality {
    linexpr: LinExpr,
    b: f64,
    // Variable ID of the injected slack variable
    slack_id: usize,
}

#[derive(Debug)]
enum Error {
    Unbounded,
    Infeasible,
}

fn align(linexpr: &LinExpr, indices: &HashMap<usize, usize>) -> Vec<f64> {
    let mut aligned = vec![0.0; indices.len()];
    for (i, var) in linexpr.vars.iter().enumerate() {
        aligned[indices[&var.id]] = linexpr.coefs[i];
    }
    aligned
}

/// https://dl.icdst.org/pdfs/files3/faa54c1f53965a11b03f9a13b023f9b2.pdf
struct Simplex {
    obj_coefs: Vec<f64>,
    obj_const: f64,
    constraints: CscMatrix,

    // Indices of the basic and non-basic variables, respectively.
    b: Vec<usize>,
    n: Vec<usize>,
    // If `positions[i] = k`, then variable `i` (identified by position, not ID) is in
    // index `k` in _either_ `b` or `n`. The caller is responsible for knowing whether
    // variable `i` is basic or non-basic.
    positions: Vec<usize>,

    // Store the IDs of all basic variables
    basic_set: HashSet<usize>,
    // Store the IDs of all the slack variables
    slack_set: HashSet<usize>,

    // If `ids[i] = k`, then variable `i` (identified by position) has ID `k`.
    ids: Vec<usize>,
    // Map a variable ID to its index in the problem. In particular, the following must
    // always be true: `ids[indices[i]] == i`.
    indices: HashMap<usize, usize>,

    // These are the "solution vectors" corresponding to the primal and dual problems.
    x: Vec<f64>,
    z: Vec<f64>,

    x_bar: Vec<f64>,
    z_bar: Vec<f64>,
}

impl Debug for Simplex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Simplex {{ objective_value: {} }}",
            self.objective_value()
        )
    }
}

impl Simplex {
    fn prepare(objective: AffExpr, constraints: Vec<Inequality>) -> Self {
        // TODO: Refactor this function to make it simpler.
        // STEP 1: Insert slack variables into each constraint, and convert the inequalities
        // to equalities.
        let mut slack_set = HashSet::new();
        let equality_constraints = constraints
            .into_iter()
            .map(|x| {
                let slack = x.slacken();
                slack_set.insert(slack.slack_id);
                slack
            })
            .collect::<Vec<_>>();

        // STEP 2: Iterate through all variables in the problem, and assign a column index
        // to each.
        let mut ids = Vec::new();
        let mut indices = HashMap::new();
        objective
            .linexpr
            .vars
            .iter()
            .map(|var| var.id)
            .chain(
                equality_constraints
                    .iter()
                    .flat_map(|c| c.linexpr.vars.iter().map(|var| var.id)),
            )
            .for_each(|id| {
                if let Entry::Vacant(e) = indices.entry(id) {
                    e.insert(ids.len());
                    ids.push(id);
                }
            });

        // STEP 3: Align the objective to these new indices
        let obj_const = objective.constant;
        let obj_coefs = align(&objective.linexpr, &indices);

        // STEP 4: Prepare remaining initialization parameters
        let x = equality_constraints.iter().map(|c| c.b).collect::<Vec<_>>();
        let z = ids
            .iter()
            .filter(|id| !slack_set.contains(id))
            .map(|id| -obj_coefs[indices[id]])
            .collect::<Vec<_>>();

        // STEP 5: Align the constraints and store as CSC matrix.
        let m = equality_constraints.len();
        let n = obj_coefs.len();
        let coords = equality_constraints
            .into_iter()
            .map(|c| c.linexpr)
            .enumerate()
            .flat_map(|(i, linexpr)| {
                linexpr
                    .vars
                    .into_iter()
                    .map(|var| indices[&var.id])
                    .zip(linexpr.coefs)
                    .map(move |(j, coef)| (i, j, coef))
            })
            .collect::<Vec<(usize, usize, f64)>>();

        let x_bar = vec![1.0; x.len()];
        let z_bar = vec![1.0; z.len()];

        Self {
            obj_coefs,
            obj_const,
            constraints: Matrix::coords(m, n, &coords).to_sparse(),
            b: (n - m..n).collect(),
            n: (0..n - m).collect(),
            positions: (0..n - m).chain(0..m).collect(),
            basic_set: ids[n - m..].iter().cloned().collect(),
            slack_set,
            ids,
            indices,
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
        e[self.positions[i]] = 1.0;
        let v = lu_solve(basis_matrix.clone().to_dense().t(), e);
        self.constraints.collect_columns(&self.n).neg_t_dot(v)
    }

    fn swap(&mut self, i: usize, j: usize) {
        assert!(self.basic_set.remove(&self.ids[i]));
        assert!(self.basic_set.insert(self.ids[j]));
        self.b[self.positions[i]] = j;
        self.n[self.positions[j]] = i;
        self.positions.swap(i, j);
    }

    fn pivot(&mut self, i: usize, j: usize, dx: &[f64], dz: &[f64]) {
        let t = self.x[self.positions[i]] / dx[self.positions[i]];
        let t_bar = self.x_bar[self.positions[i]] / dx[self.positions[i]];
        let s = self.z[self.positions[j]] / dz[self.positions[j]];
        let s_bar = self.z_bar[self.positions[j]] / dz[self.positions[j]];

        assert!(!t.is_infinite() && !t.is_nan());
        assert!(!t_bar.is_infinite() && !t_bar.is_nan());
        assert!(!s.is_infinite() && !s.is_nan());
        assert!(!s_bar.is_infinite() && !s_bar.is_nan());

        self.x.iter_mut().zip(&self.b).for_each(|(x, k)| {
            if *k == j {
                *x = t;
            } else {
                *x -= t * dx[self.positions[*k]];
            }
        });

        self.x_bar.iter_mut().zip(&self.b).for_each(|(x, k)| {
            if *k == j {
                *x = t_bar;
            } else {
                *x -= t_bar * dx[self.positions[*k]];
            }
        });

        self.z.iter_mut().zip(&self.n).for_each(|(z, k)| {
            if *k == i {
                *z = s;
            } else {
                *z -= s * dz[self.positions[*k]];
            }
        });

        self.z_bar.iter_mut().zip(&self.n).for_each(|(z, k)| {
            if *k == i {
                *z = s_bar;
            } else {
                *z -= s_bar * dz[self.positions[*k]];
            }
        });
        self.swap(i, j);
    }

    fn basis_matrix(&self) -> CscMatrix {
        self.constraints.collect_columns(&self.b)
    }

    fn solve_for_mu(&self) -> Option<Mu> {
        assert_eq!(self.x.len(), self.x_bar.len());
        assert_eq!(self.z.len(), self.z_bar.len());

        let (i, primal) = self
            .b
            .iter()
            .zip(&self.x)
            .zip(&self.x_bar)
            .filter(|((_, _), x_bar_i)| **x_bar_i > 0.0)
            .map(|((i, x_i), x_bar_i)| (i, -*x_i / *x_bar_i))
            .reduce(|(max_i, max_ratio), (i, ratio)| match ratio > max_ratio {
                true => (i, ratio),
                false => (max_i, max_ratio),
            })
            .unwrap();
            // .unwrap_or((&0, f64::NEG_INFINITY));

        let (j, dual) = self
            .n
            .iter()
            .zip(&self.z)
            .zip(&self.z_bar)
            .filter(|((_, _), z_bar_j)| **z_bar_j > 0.0)
            .map(|((j, z_j), z_bar_j)| (j, -*z_j / *z_bar_j))
            .reduce(|(max_j, max_ratio), (j, ratio)| match ratio > max_ratio {
                true => (j, ratio),
                false => (max_j, max_ratio),
            })
            // .unwrap()
            .unwrap_or((&0, f64::NEG_INFINITY));

        if primal <= 0.0 && dual <= 0.0 {
            return None;
        }

        let mu = match dual > primal {
            true => Mu {
                star: 1.0 / dual,
                step: Step::Primal(*j),
            },
            false => Mu {
                star: 1.0 / primal,
                step: Step::Dual(*i),
            },
        };
        Some(mu)
    }

    fn primal_step(&mut self, j: usize, mu_star: f64, basis_matrix: &CscMatrix) {
        let dx = self.solve_for_dx(j, basis_matrix);
        let i = self
            .x
            .iter()
            .zip(&self.x_bar)
            .map(|(x_i, x_bar_i)| x_i + mu_star * x_bar_i)
            .zip(&dx)
            .map(|(denominator, dx_i)| *dx_i / denominator)
            .enumerate()
            .reduce(|(max_i, max_ratio), (i, ratio)| match ratio > max_ratio {
                true => (i, ratio),
                false => (max_i, max_ratio),
            })
            .map(|(i, _)| self.b[i])
            .unwrap();
        dbg!(mu_star, &dx, &self.x, &self.x_bar, &self.b);
        dbg!(i, j);
        let dz = self.solve_for_dz(i, basis_matrix);

        self.pivot(i, j, &dx, &dz);
    }

    fn dual_step(&mut self, i: usize, mu_star: f64, basis_matrix: &CscMatrix) {
        todo!();
        let dz = self.solve_for_dz(i, basis_matrix);
        let j = self
            .z
            .iter()
            .zip(&self.z_bar)
            .map(|(z_j, z_bar_j)| z_j + mu_star * *z_bar_j)
            .zip(&dz)
            .map(|(denominator, dz_j)| *dz_j / denominator)
            .enumerate()
            .reduce(|(max_j, max_ratio), (j, ratio)| match ratio > max_ratio {
                true => (j, ratio),
                false => (max_j, max_ratio),
            })
            .map(|(j, _)| self.n[j])
            .unwrap();
        let dx = self.solve_for_dx(j, basis_matrix);

        self.pivot(i, j, &dx, &dz);
    }

    fn optimize(mut self) -> Result<Self, Error> {
        while let Some(mu) = self.solve_for_mu() {
            let basis_matrix = self.basis_matrix();
            match mu.step {
                Step::Primal(j) => self.primal_step(j, mu.star, &basis_matrix),
                Step::Dual(i) => self.dual_step(i, mu.star, &basis_matrix),
            }
        }
        Ok(self)
    }

    fn objective_value(&self) -> f64 {
        self.obj_const
            + self
                .b
                .iter()
                .map(|&i| self.obj_coefs[i] * self.x[self.positions[i]])
                .sum::<f64>()
    }

    fn solution(&self, var: &Variable) -> f64 {
        match self.basic_set.contains(&var.id) {
            true => self.x[self.positions[self.indices[&var.id]]],
            false => 0.0,
        }
    }
}

#[derive(Debug)]
struct Mu {
    star: f64,
    step: Step,
}

#[derive(Debug)]
enum Step {
    Primal(usize),
    Dual(usize),
}

#[cfg(test)]
mod tests {
    use crate::simplex2::*;

    #[test]
    fn test_primal_simplex_1() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(4.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x), (-1.0, &y)], 1.0);
        let c_2 = Inequality::new(&[(2.0, &x), (-1.0, &y)], 3.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 5.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(result.objective_value(), 31.0);
        assert_eq!(result.solution(&x), 4.0);
        assert_eq!(result.solution(&y), 5.0);
    }

    #[test]
    fn test_primal_simplex_2() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();

        let objective = AffExpr::new(&[(5.0, &x_1), (4.0, &x_2), (3.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(2.0, &x_1), (3.0, &x_2), (1.0, &x_3)], 5.0);
        let c_2 = Inequality::new(&[(4.0, &x_1), (1.0, &x_2), (2.0, &x_3)], 11.0);
        let c_3 = Inequality::new(&[(3.0, &x_1), (4.0, &x_2), (2.0, &x_3)], 8.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(result.objective_value(), 13.0);
        assert_eq!(result.solution(&x_1), 2.0);
        assert_eq!(result.solution(&x_2), 0.0);
        assert_eq!(result.solution(&x_3), 1.0);
    }

    #[test]
    fn test_primal_simplex_3() {
        // LP relaxation of the problem on page C-10
        // http://web.tecnico.ulisboa.pt/mcasquilho/compute/_linpro/TaylorB_module_c.pdf
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();
        let x_4 = Variable::new();

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

        let result = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(result.objective_value(), 750.0);
        assert_eq!(result.solution(&x_1), 1.0);
        assert_eq!(result.solution(&x_2), 0.0);
        assert_eq!(result.solution(&x_3), 1.0);
        assert_eq!(result.solution(&x_4), 1.0 / 3.0);
    }

    #[test]
    fn test_dual_simplex() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(-1.0, &x), (-1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
        let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
        let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(result.objective_value(), -7.0);
        assert_eq!(result.solution(&x), 7.0);
        assert_eq!(result.solution(&y), 0.0);
    }

    #[test]
    fn test_two_phase_simplex_1() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(-1.0, &x), (4.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
        let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
        let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
        let constraints = vec![c_1, c_2, c_3];

        match Simplex::prepare(objective, constraints)
            .optimize()
            .unwrap_err()
        {
            Error::Unbounded => (),
            Error::Infeasible => panic!("problem should be unbounded"),
        }
    }

    #[test]
    fn test_two_phase_simplex_2() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(-2.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-1.0, &x), (1.0, &y)], -1.0);
        let c_2 = Inequality::new(&[(-1.0, &x), (-2.0, &y)], -2.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 1.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(result.objective_value(), -1.0);
        assert_eq!(result.solution(&x), 2.0);
        assert_eq!(result.solution(&y), 1.0);
    }
}
