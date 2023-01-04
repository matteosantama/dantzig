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
    upper_bound: Option<f64>,
}

enum Error {
    Unbounded,
    Infeasible,
}

/// https://dl.icdst.org/pdfs/files3/faa54c1f53965a11b03f9a13b023f9b2.pdf
#[allow(non_snake_case)]
struct Simplex {
    obj_coefs: Vec<f64>,

    B: Vec<usize>,
    N: Vec<usize>,
    // These keys map an index in the overall problem to an index within the set
    B_key: HashMap<usize, usize>,
    N_key: HashMap<usize, usize>,

    x_star: Vec<f64>,
    z_star: Vec<f64>,

    is_primal_feasible: bool,
    is_dual_feasible: bool,
}

impl Simplex {
    fn new(objective: &AffExpr, constraints: &[Inequality]) -> Self {
        todo!()
    }

    fn primal_simplex_method(&mut self) -> Result<(), Error> {
        todo!()
        // debug_assert!(self.x_star.iter().all(|&x| x >= 0.0));
        //
        // while let Some(j) = try_enter_variable(&self.N, &self.z_star) {
        //     let delta_x = lu_solve().unwrap();
        //     let i = exit_variable(&self.B, &delta_x, &self.x_star);
        //     let t = self.x_star[i] / delta_x[i];
        //
        //     let position = self.B_key.remove(&i).unwrap();
        //     self.B_key.insert(j, position);
        //     self.B[position] = j;
        //
        //     for (k, x_star) in self.x_star.iter_mut().enumerate() {
        //         if k == j {
        //             *x_star = t;
        //         } else {
        //             *x_star -= t * delta_x[k]
        //         }
        //     }
        //
        //     let delta_z = lu_solve().unwrap();
        //     let s = self.z_star[j] / delta_z[j];
        //
        //     let position = self.N_key.remove(&j).unwrap();
        //     self.N_key.insert(i, position);
        //     self.N[position] = i;
        //
        //     for (k, z_star) in self.z_star.iter_mut().enumerate() {
        //         if k == i {
        //             *z_star = s;
        //         } else {
        //             *z_star -= s * delta_z[k]
        //         }
        //     }
        // }
        // Ok(())
    }

    fn dual_simplex_method(&mut self) -> Result<(), Error> {
        todo!()
    }

    fn optimize(&mut self) -> Result<(), Error> {
        match (self.is_primal_feasible, self.is_dual_feasible) {
            (true, true) => Ok(()),
            (true, false) => self.primal_simplex_method(),
            (false, true) => self.dual_simplex_method().map_err(|err| match err {
                Error::Unbounded => Error::Infeasible,
                Error::Infeasible => panic!("dual problem was assumed feasible"),
            }),
            (false, false) => todo!(),
        }
    }
}

fn try_enter_variable(indices: &[usize], coefs: &[f64]) -> Option<usize> {
    debug_assert!(!indices.is_empty());
    debug_assert_eq!(indices.len(), coefs.len());

    return indices
        .iter()
        .zip(coefs)
        .find(|(_, &e)| e < 0.0)
        .map(|(&j, _)| j);
}

fn exit_variable(indices: &[usize], numerator: &[f64], denominator: &[f64]) -> usize {
    debug_assert!(!indices.is_empty());
    debug_assert_eq!(indices.len(), numerator.len());
    debug_assert_eq!(indices.len(), denominator.len());

    let mut index = 0;
    let mut max_ratio = numerator[0] / denominator[0];

    for (ratio, i) in numerator
        .iter()
        .zip(denominator)
        .map(|(n, d)| n / d)
        .zip(indices)
    {
        if ratio > max_ratio {
            max_ratio = ratio;
            index = *i;
        }
    }
    indices[index]
}

#[cfg(test)]
mod tests {
    use crate::simplex2::*;
}
