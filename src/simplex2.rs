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

fn iter_all_ids<'a>(
    objective: &'a AffExpr,
    equalities: &'a [Equality],
) -> impl Iterator<Item = usize> + 'a {
    objective.linexpr.vars.iter().map(|var| var.id).chain(
        equalities
            .iter()
            .flat_map(|equality| equality.linexpr.vars.iter().map(|var| var.id)),
    )
}

fn align(objective: AffExpr, id_to_index: &HashMap<usize, usize>) -> Vec<f64> {
    let mut result = vec![0.0; id_to_index.len()];
    for (i, var) in objective.linexpr.vars.iter().enumerate() {
        result[id_to_index[&var.id]] = objective.linexpr.coefs[i];
    }
    result
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
struct Simplex {
    obj_coefs: Vec<f64>,
    obj_const: f64,
    constraints: CscMatrix,

    // Indices of the basic and non-basic variables, respectively.
    b: Vec<usize>,
    n: Vec<usize>,

    // Map a variable index to the index within `b` or `n`.
    b_key: HashMap<usize, usize>,
    n_key: HashMap<usize, usize>,

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
        write!(
            f,
            "Simplex {{ objective_value: {} }}",
            self.objective_value()
        )
    }
}

impl Simplex {
    fn presolve(objective: AffExpr, constraints: Vec<Inequality>) -> Self {
        let m = constraints.len();

        // STEP 1: Add slack variables to all constraints to transform inequalities to
        // equalities. We store the ID of the slack variable, along with the index of
        // its associated constraint.
        let mut slack_ids = HashMap::with_capacity(m);
        let equalities = constraints
            .into_iter()
            .map(|inequality| {
                let equality = inequality.slacken();
                slack_ids.insert(equality.slack_id, equality.b);
                equality
            })
            .collect::<Vec<_>>();

        // STEP 2: Iterate through all IDs of the problem, and reduce them to a single unique
        // list. To each ID, we also assign an index.
        let mut index_to_id = Vec::new();
        let mut id_to_index = HashMap::new();

        for id in iter_all_ids(&objective, &equalities) {
            if let Entry::Vacant(e) = id_to_index.entry(id) {
                e.insert(index_to_id.len());
                index_to_id.push(id);
            }
        }

        // STEP 3: Align the objective to the indices.
        let obj_const = objective.constant;
        let obj_coefs = align(objective, &id_to_index);

        // STEP 4: Iterate through the unique variables, and construct our basic feasible
        // solution.
        let mut b_key = HashMap::with_capacity(m);
        let mut b = Vec::with_capacity(m);
        let mut x = Vec::with_capacity(m);

        let mut n_key = HashMap::with_capacity(index_to_id.len() - m);
        let mut n = Vec::with_capacity(index_to_id.len() - m);
        let mut z = Vec::with_capacity(index_to_id.len() - m);

        for id in &index_to_id {
            let index = id_to_index[id];
            if let Entry::Occupied(entry) = slack_ids.entry(*id) {
                b_key.insert(index, b.len());
                b.push(index);
                x.push(*entry.get())
            } else {
                n_key.insert(index, n.len());
                n.push(index);
                z.push(-obj_coefs[index]);
            }
        }

        // STEP 5: Prepare the remaining initialization parameters.
        let x_bar = vec![1.0; x.len()];
        let z_bar = vec![1.0; z.len()];

        let constraints = sparsify(equalities, &id_to_index);

        Self {
            obj_coefs,
            obj_const,
            constraints,
            b,
            n,
            b_key,
            n_key,
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
    fn swap(mut self, i: usize, j: usize) -> Result<Self, Error> {
        assert!(self.b_key.contains_key(&i) && !self.b_key.contains_key(&j));
        assert!(self.n_key.contains_key(&j) && !self.n_key.contains_key(&i));

        self.b[self.b_key[&i]] = j;
        self.n[self.n_key[&j]] = i;

        self.b_key.insert(j, self.b_key[&i]);
        self.n_key.insert(i, self.n_key[&j]);

        assert!(self.b_key.remove(&i).is_some());
        assert!(self.n_key.remove(&j).is_some());

        Ok(self)
    }

    fn pivot(mut self, i: usize, j: usize, dx: &[f64], dz: &[f64]) -> Result<Self, Error> {
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

    fn solve_for_mu(&self) -> Option<Mu> {
        assert_eq!(self.x.len(), self.x_bar.len());
        assert_eq!(self.z.len(), self.z_bar.len());

        let (i, primal) = try_pick_enter(&self.b, &self.x, &self.x_bar);
        let (j, dual) = try_pick_enter(&self.n, &self.z, &self.z_bar);

        if primal <= 0.0 && dual <= 0.0 {
            return None;
        }
        let mu = match dual > primal {
            true => Mu {
                star: dual,
                step: Step::Primal(j),
            },
            false => Mu {
                star: primal,
                step: Step::Dual(i),
            },
        };
        Some(mu)
    }

    fn primal_step(
        mut self,
        j: usize,
        mu_star: f64,
        basis_matrix: &CscMatrix,
    ) -> Result<Self, Error> {
        let dx = self.solve_for_dx(j, basis_matrix);
        let i = pick_exit(mu_star, &self.x, &self.x_bar, &dx, &self.b);
        let dz = self.solve_for_dz(i, basis_matrix);

        self.pivot(i, j, &dx, &dz)
    }

    fn dual_step(
        mut self,
        i: usize,
        mu_star: f64,
        basis_matrix: &CscMatrix,
    ) -> Result<Self, Error> {
        let dz = self.solve_for_dz(i, basis_matrix);
        let j = pick_exit(mu_star, &self.z, &self.z_bar, &dz, &self.n);
        let dx = self.solve_for_dx(j, basis_matrix);

        self.pivot(i, j, &dx, &dz)
    }

    fn solve(self) -> Result<Self, Error> {
        if let Some(mu) = self.solve_for_mu() {
            let basis_matrix = self.basis_matrix();
            let result = match mu.step {
                Step::Primal(j) => self.primal_step(j, mu.star, &basis_matrix)?,
                Step::Dual(i) => self.dual_step(i, mu.star, &basis_matrix)?,
            };
            return result.solve();
        }
        Ok(self)
    }

    fn objective_value(&self) -> f64 {
        self.obj_const
            + self
                .b_key
                .iter()
                .map(|(x, y)| self.obj_coefs[*x] * self.x[*y])
                .sum::<f64>()
    }

    fn solution(&self, var: &Variable) -> f64 {
        self.b_key
            .get(&self.id_to_index[&var.id])
            .map(|t| self.x[*t])
            .unwrap_or(0.0)
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

fn try_pick_enter(bn: &[usize], xz: &[f64], xz_bar: &[f64]) -> (usize, f64) {
    bn.iter()
        .zip(xz)
        .zip(xz_bar)
        .filter(|((_, _), xz_bar_k)| **xz_bar_k > 0.0)
        .map(|((&k, &xz_k), &xz_bar_k)| (k, -xz_k / xz_bar_k))
        .reduce(|(max_k, max_ratio), (k, ratio)| match ratio > max_ratio {
            true => (k, ratio),
            false => (max_k, max_ratio),
        })
        .unwrap_or((0, f64::NEG_INFINITY))
}

fn pick_exit(mu_star: f64, xz: &[f64], xz_bar: &[f64], dxz: &[f64], bn: &[usize]) -> usize {
    xz.iter()
        .zip(xz_bar)
        .map(|(xz_k, xz_bar_k)| xz_k + mu_star * xz_bar_k)
        .zip(dxz)
        .map(|(denominator, dxz_k)| *dxz_k / denominator)
        .enumerate()
        .reduce(|(max_k, max_ratio), (k, ratio)| match ratio > max_ratio {
            true => (k, ratio),
            false => (max_k, max_ratio),
        })
        .map(|(k, _)| bn[k])
        .unwrap()
}

/// Compute `x / y`, with `0 / 0 = 0`.
fn safe_divide(x: f64, y: f64) -> f64 {
    let result = if x == 0.0 && y == 0.0 { 0.0 } else { x / y };
    assert!(
        !result.is_infinite() && !result.is_nan(),
        "safe divide {} / {}",
        x,
        y
    );
    result
}

#[cfg(test)]
mod tests {
    use crate::simplex2::*;

    #[test]
    fn test_1() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(4.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x), (-1.0, &y)], 1.0);
        let c_2 = Inequality::new(&[(2.0, &x), (-1.0, &y)], 3.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 5.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 31.0);
        assert_eq!(result.solution(&x), 4.0);
        assert_eq!(result.solution(&y), 5.0);
    }

    #[test]
    fn test_2() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();

        let objective = AffExpr::new(&[(5.0, &x_1), (4.0, &x_2), (3.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(2.0, &x_1), (3.0, &x_2), (1.0, &x_3)], 5.0);
        let c_2 = Inequality::new(&[(4.0, &x_1), (1.0, &x_2), (2.0, &x_3)], 11.0);
        let c_3 = Inequality::new(&[(3.0, &x_1), (4.0, &x_2), (2.0, &x_3)], 8.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 13.0);
        assert_eq!(result.solution(&x_1), 2.0);
        assert_eq!(result.solution(&x_2), 0.0);
        assert_eq!(result.solution(&x_3), 0.9999999999999998);
    }

    #[test]
    fn test_3() {
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

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 750.0);
        assert_eq!(result.solution(&x_1), 1.0);
        assert_eq!(result.solution(&x_2), 0.0);
        assert_eq!(result.solution(&x_3), 1.0000000000000009);
        assert_eq!(result.solution(&x_4), 0.3333333333333314);
    }

    #[test]
    fn test_4() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();

        let objective = AffExpr::new(&[(10.0, &x_1), (12.0, &x_2), (12.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(1.0, &x_1), (2.0, &x_2), (2.0, &x_3)], 20.0);
        let c_2 = Inequality::new(&[(2.0, &x_1), (1.0, &x_2), (2.0, &x_3)], 20.0);
        let c_3 = Inequality::new(&[(2.0, &x_1), (2.0, &x_2), (1.0, &x_3)], 20.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 136.0);
        assert_eq!(result.solution(&x_1), 4.0);
        assert_eq!(result.solution(&x_2), 4.0);
        assert_eq!(result.solution(&x_3), 4.0);
    }

    #[test]
    fn test_5() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(-1.0, &x), (-1.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
        let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
        let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), -7.0);
        assert_eq!(result.solution(&x), 7.0);
        assert_eq!(result.solution(&y), 0.0);
    }

    #[test]
    fn test_6() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();

        let objective = AffExpr::new(&[(-10.0, &x_1), (-12.0, &x_2), (-12.0, &x_3)], 0.0);
        let c_1 = Inequality::new(&[(-1.0, &x_1), (-2.0, &x_2), (-2.0, &x_3)], -20.0);
        let c_2 = Inequality::new(&[(-2.0, &x_1), (-1.0, &x_2), (-2.0, &x_3)], -20.0);
        let c_3 = Inequality::new(&[(-2.0, &x_1), (-2.0, &x_2), (-1.0, &x_3)], -20.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), -136.0);
        assert_eq!(result.solution(&x_1), 4.0);
        assert_eq!(result.solution(&x_2), 4.0);
        assert_eq!(result.solution(&x_3), 4.0);
    }

    // #[test]
    // fn test_7() {
    //     let x = Variable::new();
    //     let y = Variable::new();
    //
    //     let objective = AffExpr::new(&[(-1.0, &x), (4.0, &y)], 0.0);
    //     let c_1 = Inequality::new(&[(-2.0, &x), (-1.0, &y)], 4.0);
    //     let c_2 = Inequality::new(&[(-2.0, &x), (4.0, &y)], -8.0);
    //     let c_3 = Inequality::new(&[(-1.0, &x), (3.0, &y)], -7.0);
    //     let constraints = vec![c_1, c_2, c_3];
    //
    //     match Simplex::prepare(objective, constraints)
    //         .optimize()
    //         .unwrap_err()
    //     {
    //         Error::Unbounded => (),
    //         Error::Infeasible => panic!("problem should be unbounded"),
    //     }
    // }

    #[test]
    fn test_8() {
        let x = Variable::new();
        let y = Variable::new();

        let objective = AffExpr::new(&[(-2.0, &x), (3.0, &y)], 0.0);
        let c_1 = Inequality::new(&[(-1.0, &x), (1.0, &y)], -1.0);
        let c_2 = Inequality::new(&[(-1.0, &x), (-2.0, &y)], -2.0);
        let c_3 = Inequality::new(&[(1.0, &y)], 1.0);
        let constraints = vec![c_1, c_2, c_3];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), -1.0);
        assert_eq!(result.solution(&x), 2.0);
        assert_eq!(result.solution(&y), 1.0);
    }

    #[test]
    fn test_9() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();
        let x_4 = Variable::new();
        let x_5 = Variable::new();
        let x_6 = Variable::new();
        let x_7 = Variable::new();
        let x_8 = Variable::new();

        let objective = AffExpr::new(
            &[
                (-3.0, &x_1),
                (-1.0, &x_2),
                (1.0, &x_3),
                (2.0, &x_4),
                (-1.0, &x_5),
                (1.0, &x_6),
                (-1.0, &x_7),
                (-4.0, &x_8),
            ],
            0.0,
        );
        let c_1 = Inequality::new(
            &[
                (1.0, &x_1),
                (4.0, &x_3),
                (1.0, &x_4),
                (-5.0, &x_5),
                (-2.0, &x_6),
                (3.0, &x_7),
                (-6.0, &x_8),
            ],
            7.0,
        );
        let c_2 = Inequality::new(
            &[
                (-1.0, &x_1),
                (-4.0, &x_3),
                (-1.0, &x_4),
                (5.0, &x_5),
                (2.0, &x_6),
                (-3.0, &x_7),
                (6.0, &x_8),
            ],
            -7.0,
        );
        let c_3 = Inequality::new(
            &[
                (1.0, &x_2),
                (-3.0, &x_3),
                (-1.0, &x_4),
                (4.0, &x_5),
                (1.0, &x_6),
                (-2.0, &x_7),
                (5.0, &x_8),
            ],
            -3.0,
        );
        let c_4 = Inequality::new(
            &[
                (-1.0, &x_2),
                (3.0, &x_3),
                (1.0, &x_4),
                (-4.0, &x_5),
                (-1.0, &x_6),
                (2.0, &x_7),
                (-5.0, &x_8),
            ],
            3.0,
        );
        let constraints = vec![c_1, c_2, c_3, c_4];

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 24.0);
        assert_eq!(result.solution(&x_1), 0.0);
        assert_eq!(result.solution(&x_2), 6.0);
        assert_eq!(result.solution(&x_3), 1.0);
        assert_eq!(result.solution(&x_4), 15.0);
        assert_eq!(result.solution(&x_5), 2.0);
        assert_eq!(result.solution(&x_6), 1.0);
        assert_eq!(result.solution(&x_7), 0.0);
        assert_eq!(result.solution(&x_8), 0.0);
    }

    #[test]
    fn test_10() {
        let x_1 = Variable::new();
        let x_2 = Variable::new();
        let x_3 = Variable::new();
        let x_4 = Variable::new();
        let x_5 = Variable::new();
        let x_6 = Variable::new();

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

        let result = Simplex::presolve(objective, constraints).solve().unwrap();
        assert_eq!(result.objective_value(), 33.0);
        assert_eq!(result.solution(&x_1), 3.0);
        assert_eq!(result.solution(&x_2), 4.0);
        assert_eq!(result.solution(&x_3), 5.0);
        assert_eq!(result.solution(&x_4), 0.0);
        assert_eq!(result.solution(&x_5), 5.0);
        assert_eq!(result.solution(&x_6), 0.0);
    }
}
