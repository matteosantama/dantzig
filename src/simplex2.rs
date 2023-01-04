use crate::linalg2::{CooMatrix, lu_solve};
use std::collections::HashMap;
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
    constraints: CooMatrix,

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

    fn basis_matrix(&self) -> CooMatrix {
        self.constraints.collect_columns(&self.b)
    }

    fn solve_for_dx(&self, j: usize) -> Vec<f64> {
        let a = self.basis_matrix();
        let b = self.constraints.column(j);
        lu_solve(&a, b)
    }

    fn solve_for_dz(&self, i: usize) -> Vec<f64> {
        todo!()
    }

    fn swap(&mut self, enter: usize, exit: usize) {
        self.positions.swap(enter, exit);

        self.b[self.positions[enter]] = enter;
        self.n[self.positions[exit]] = exit;
    }

    fn run_primal_simplex(&mut self) -> Result<(), Error> {
        while let Some(j) = try_pick_enter(&self.n, &self.z) {
            let dx = self.solve_for_dx(j);
            let i = pick_exit(&self.b, &dx, &self.x);
            let dz = self.solve_for_dz(i);

            self.swap(j, i);

            let t = self.x[i] / dx[i];
            assert!(!t.is_nan(), "t is NaN");
            assert!(!t.is_infinite(), "t is infinite");
            if t < 0.0 {
                return Err(Error::Unbounded)
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

    for (ratio, i) in n
        .iter()
        .zip(d)
        .map(|(n, d)| n / d)
        .zip(set)
    {
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
