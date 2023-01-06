use crate::linalg2::{lu_solve, CscMatrix, Matrix};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
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
        let var = Variable::new();
        self.linexpr.vars.push(var);
        self.linexpr.coefs.push(1.0);
        Equality {
            linexpr: self.linexpr,
            b: self.b,
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
}

#[derive(Debug)]
enum Error {
    Unbounded,
    Infeasible,
}

struct Solution {
    objective_value: f64,
    solution: HashMap<usize, f64>,
}

impl Solution {
    fn __getitem__(&self, id: usize) -> f64 {
        self.solution.get(&id).cloned().unwrap_or(0.0)
    }
}

impl From<&mut Simplex> for Solution {
    fn from(simplex: &mut Simplex) -> Self {
        Self {
            objective_value: simplex.objective_value(),
            solution: simplex.solution(),
        }
    }
}

fn align(linexpr: &LinExpr, indexer: &HashMap<usize, usize>) -> Vec<f64> {
    let mut aligned = vec![0.0; indexer.len()];
    for (i, var) in linexpr.vars.iter().enumerate() {
        aligned[indexer[&var.id]] = linexpr.coefs[i];
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

    // If `ids[i] = k`, then variable `i` (identified by position) has ID `k`.
    ids: Vec<usize>,
    // Map a variable ID to its index in the problem. In particular, the following must
    // always be true: `ids[indexer[i]] == i`.
    #[allow(dead_code)]
    indexer: HashMap<usize, usize>,

    x: Vec<f64>,
    z: Vec<f64>,
}

impl Simplex {
    fn prepare(objective: AffExpr, constraints: Vec<Inequality>) -> Self {
        // STEP 0: Collect and identify all original variables, before we add any slack
        let orig_vars = objective
            .linexpr
            .vars
            .iter()
            .map(|var| var.id)
            .chain(
                constraints
                    .iter()
                    .flat_map(|c| c.linexpr.vars.iter().map(|var| var.id)),
            )
            .collect::<HashSet<usize>>();

        // STEP 1: Insert slack variables into each constraint, and convert the inequalities
        // to equalities.
        let equality_constraints = constraints
            .into_iter()
            .map(|x| x.slacken())
            .collect::<Vec<_>>();

        // STEP 2: Iterate through all variables in the problem, and assign a column index
        // to each.
        let mut ids = Vec::new();
        let mut indexer = HashMap::new();
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
                if let Entry::Vacant(e) = indexer.entry(id) {
                    e.insert(ids.len());
                    ids.push(id);
                }
            });

        // STEP 3: Align the objective to these new indices
        let obj_const = objective.constant;
        let obj_coefs = align(&objective.linexpr, &indexer);

        // STEP 4: Prepare remaining initialization parameters
        let x = equality_constraints.iter().map(|c| c.b).collect();
        let z = ids
            .iter()
            .filter(|id| orig_vars.contains(id))
            .map(|id| -obj_coefs[indexer[id]])
            .collect();

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
                    .map(|var| indexer[&var.id])
                    .zip(linexpr.coefs)
                    .map(move |(j, coef)| (i, j, coef))
            })
            .collect::<Vec<(usize, usize, f64)>>();

        Self {
            obj_coefs,
            obj_const,
            constraints: Matrix::coords(m, n, &coords).to_sparse(),
            b: (n - m..n).collect(),
            n: (0..n - m).collect(),
            positions: (0..n - m).chain(0..m).collect(),
            ids,
            indexer,
            x,
            z,
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
        self.b[self.positions[i]] = j;
        self.n[self.positions[j]] = i;

        self.positions.swap(i, j);
    }

    fn run_primal_simplex(&mut self) -> Result<Solution, Error> {
        while let Some(j) = try_pick_enter(&self.n, &self.z) {
            let basis_matrix = self.constraints.collect_columns(&self.b);

            let dx = self.solve_for_dx(j, &basis_matrix);
            let i = pick_exit(&self.b, &dx, &self.x);
            let dz = self.solve_for_dz(i, &basis_matrix);

            let t = self.x[self.positions[i]] / dx[self.positions[i]];
            let s = self.z[self.positions[j]] / dz[self.positions[j]];
            assert!(!t.is_nan() && !t.is_infinite());
            assert!(!s.is_nan() && !s.is_infinite());
            if t < 0.0 {
                return Err(Error::Unbounded);
            }
            self.pivot(i, j, t, s, &dx, &dz);
        }
        Ok(Solution::from(self))
    }

    fn run_dual_simplex(&mut self) -> Result<Solution, Error> {
        while let Some(i) = try_pick_enter(&self.b, &self.x) {
            let basis_matrix = self.constraints.collect_columns(&self.b);

            let dz = self.solve_for_dz(i, &basis_matrix);
            let j = pick_exit(&self.n, &dz, &self.z);
            let dx = self.solve_for_dx(j, &basis_matrix);

            let s = self.z[self.positions[j]] / dz[self.positions[j]];
            let t = self.x[self.positions[i]] / dx[self.positions[i]];
            assert!(!s.is_nan() && !s.is_infinite());
            assert!(!t.is_nan() && !t.is_infinite());
            if s < 0.0 {
                return Err(Error::Unbounded);
            }
            self.pivot(i, j, t, s, &dx, &dz);
        }
        Ok(Solution::from(self))
    }

    fn pivot(&mut self, i: usize, j: usize, t: f64, s: f64, dx: &[f64], dz: &[f64]) {
        self.x.iter_mut().zip(&self.b).for_each(|(x, k)| {
            if *k == i {
                *x = t;
            } else {
                *x -= t * dx[self.positions[*k]];
            }
        });
        self.z.iter_mut().zip(&self.n).for_each(|(z, k)| {
            if *k == j {
                *z = s;
            } else {
                *z -= s * dz[self.positions[*k]];
            }
        });
        self.swap(i, j);
    }

    fn optimize(&mut self) -> Result<Solution, Error> {
        let is_primal_feasible = self.x.iter().all(|k| *k >= 0.0);
        let is_dual_feasible = self.z.iter().all(|k| *k >= 0.0);

        match (is_primal_feasible, is_dual_feasible) {
            (true, true) => Ok(Solution::from(self)),
            (true, false) => self.run_primal_simplex(),
            (false, true) => self.run_dual_simplex().map_err(|err| match err {
                Error::Unbounded => Error::Infeasible,
                Error::Infeasible => panic!("dual problem was assumed feasible"),
            }),
            (false, false) => todo!(),
        }
    }

    fn objective_value(&self) -> f64 {
        self.b
            .iter()
            .map(|&i| self.obj_coefs[i] * self.x[self.positions[i]])
            .sum::<f64>()
            + self.obj_const
    }

    fn solution(&self) -> HashMap<usize, f64> {
        self.b
            .iter()
            .map(|&i| (self.ids[i], self.x[self.positions[i]]))
            .collect()
    }
}

fn try_pick_enter(set: &[usize], coefs: &[f64]) -> Option<usize> {
    debug_assert!(!set.is_empty());
    debug_assert_eq!(set.len(), coefs.len());
    coefs
        .iter()
        .enumerate()
        .find(|(_, e)| **e < 0.0)
        .map(|(j, _)| set[j])
}

fn pick_exit(set: &[usize], n: &[f64], d: &[f64]) -> usize {
    debug_assert_eq!(n.len(), d.len());

    let mut max_ratio_i = 0;
    let mut max_ratio = n[0] / d[0];

    n.iter().zip(d).enumerate().for_each(|(i, (n_i, d_i))| {
        let ratio = if *n_i == 0.0 && *d_i == 0.0 {
            0.0
        } else {
            *n_i / *d_i
        };
        assert!(!ratio.is_infinite() && !ratio.is_nan());
        if ratio > max_ratio {
            max_ratio = ratio;
            max_ratio_i = i;
        }
    });
    set[max_ratio_i]
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

        let soln = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(soln.objective_value, 31.0);
        assert_eq!(soln.__getitem__(x.id), 4.0);
        assert_eq!(soln.__getitem__(y.id), 5.0);
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

        let soln = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(soln.objective_value, 13.0);
        assert_eq!(soln.__getitem__(x_1.id), 2.0);
        assert_eq!(soln.__getitem__(x_2.id), 0.0);
        assert_eq!(soln.__getitem__(x_3.id), 1.0);
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

        let soln = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(soln.objective_value, 750.0);
        assert_eq!(soln.__getitem__(x_1.id), 1.0);
        assert_eq!(soln.__getitem__(x_2.id), 0.0);
        assert_eq!(soln.__getitem__(x_3.id), 1.0);
        assert_eq!(soln.__getitem__(x_4.id), 1.0 / 3.0);
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

        let soln = Simplex::prepare(objective, constraints).optimize().unwrap();
        assert_eq!(soln.objective_value, -7.0);
        assert_eq!(soln.__getitem__(x.id), 7.0);
        assert_eq!(soln.__getitem__(y.id), 0.0);
    }
}
