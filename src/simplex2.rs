use crate::linalg2::{lu_solve, lu_solve_t, CscMatrix};
use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy)]
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

struct AffExpr {
    linexpr: LinExpr,
    constant: f64,
}

struct Inequality {
    linexpr: LinExpr,
    upper_bound: f64,
}

enum Error {
    Unbounded,
    Infeasible,
}

/// https://dl.icdst.org/pdfs/files3/faa54c1f53965a11b03f9a13b023f9b2.pdf
struct Simplex {
    objective: Vec<f64>,
    constraints: CscMatrix,

    // Indices of the basic and non-basic variables, respectively.
    b: Vec<usize>,
    n: Vec<usize>,
    // TODO:
    positions: Vec<usize>,

    x: Vec<f64>,
    z: Vec<f64>,
}

impl Simplex {
    fn new(objective: &AffExpr, constraints: &[Inequality]) -> Self {
        todo!()
    }

    fn solve_for_dx(&self, j: usize, basis_matrix: &CscMatrix) -> Vec<f64> {
        let b = self.constraints.column(j);
        lu_solve(basis_matrix, b)
    }

    fn solve_for_dz(&self, i: usize, basis_matrix: &CscMatrix) -> Vec<f64> {
        let e = (0..basis_matrix.nrows())
            .map(|k| (k == i) as u8 as f64)
            .collect::<Vec<_>>();
        let v = lu_solve_t(basis_matrix, e);
        self.constraints
            .collect_columns(&self.n)
            .neg_t_dot(v)
    }

    fn swap(&mut self, enter: usize, exit: usize) {
        self.positions.swap(enter, exit);

        self.b[self.positions[enter]] = enter;
        self.n[self.positions[exit]] = exit;
    }

    fn run_primal_simplex(&mut self) -> Result<(), Error> {
        while let Some(j) = try_pick_enter(&self.n, &self.z) {
            let basis_matrix = self.constraints.collect_columns(&self.b);

            let dx = self.solve_for_dx(j, &basis_matrix);
            let i = pick_exit(&self.b, &dx, &self.x);
            let dz = self.solve_for_dz(i, &basis_matrix);

            let t = self.x[i] / dx[i];
            assert!(!t.is_nan(), "t is NaN");
            assert!(!t.is_infinite(), "t is infinite");
            if t < 0.0 {
                return Err(Error::Unbounded);
            }
            let s = self.z[j] / dz[j];
            assert!(!s.is_nan(), "s is NaN");
            assert!(!s.is_infinite(), "s is infinite");

            for (k, x) in self.x.iter_mut().enumerate() {
                if k == j {
                    *x = t;
                } else {
                    *x -= t * dx[k]
                }
            }
            for (k, z) in self.z.iter_mut().enumerate() {
                if k == i {
                    *z = s;
                } else {
                    *z -= s * dz[k]
                }
            }
            self.swap(j, i);
        }
        Ok(())
    }

    fn run_dual_simplex(&mut self) -> Result<(), Error> {
        todo!()
    }

    fn optimize(&mut self) -> Result<(), Error> {
        let is_primal_feasible = self.x.iter().all(|&x| x >= 0.0);
        let is_dual_feasible = self.z.iter().all(|&x| x >= 0.0);

        match (is_primal_feasible, is_dual_feasible) {
            (true, true) => Ok(()),
            (true, false) => self.run_primal_simplex(),
            (false, true) => self.run_dual_simplex().map_err(|err| match err {
                Error::Unbounded => Error::Infeasible,
                Error::Infeasible => panic!("dual problem was assumed feasible"),
            }),
            (false, false) => todo!(),
        }
    }
}

fn try_pick_enter(set: &[usize], coefs: &[f64]) -> Option<usize> {
    debug_assert!(!set.is_empty());
    debug_assert_eq!(set.len(), coefs.len());
    return set
        .iter()
        .zip(coefs)
        .find(|(_, e)| **e < 0.0)
        .map(|(&j, _)| j);
}

fn pick_exit(set: &[usize], n: &[f64], d: &[f64]) -> usize {
    debug_assert!(!set.is_empty());
    debug_assert_eq!(set.len(), n.len());
    debug_assert_eq!(set.len(), d.len());

    let mut index = 0;
    let mut max_ratio = n[0] / d[0];

    for (ratio, i) in n.iter().zip(d).map(|(n, d)| n / d).zip(set) {
        if ratio > max_ratio {
            max_ratio = ratio;
            index = *i;
        }
    }
    set[index]
}

#[cfg(test)]
mod tests {
    use crate::simplex2::*;
}
