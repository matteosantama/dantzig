use crate::error::Error;
use crate::linalg::{dot, Matrix};

fn check_feasibility(objective_value: f64) -> Result<(), Error> {
    if objective_value > f64::EPSILON {
        return Err(Error::Infeasible);
    }
    Ok(())
}

pub fn solve(
    objective: Vec<f64>,
    constraints: Vec<Vec<f64>>,
    rhs: Vec<f64>,
) -> Result<(f64, Vec<f64>), Error> {
    let constraint_matrix = Matrix::new_unchecked(constraints);

    if constraint_matrix.is_empty() {
        assert!(rhs.is_empty());
        return match objective.iter().all(|&x| x > 0.0) {
            true => Ok((0.0, vec![0.0; objective.len()])),
            false => Err(Error::Unbounded),
        };
    }
    assert_eq!(objective.len(), constraint_matrix.n());
    assert_eq!(rhs.len(), constraint_matrix.m());

    let mut aux = Simplex::auxiliary(constraint_matrix, &rhs);

    // NOTE: The auxiliary problem is trivially bounded and feasible, so we
    // should not encounter an error here.
    let (objective_value, _) = aux
        .solve()
        .expect("Failed solving auxiliary problem; please report as a bug");

    check_feasibility(objective_value)?;

    aux.prepare_for_phase_two(objective).solve()
}

struct Simplex {
    is_auxiliary: bool,
    objective: Vec<f64>,
    constraints: Matrix,
    basis_inverse: Matrix,
    basis_vars: Vec<usize>,
    basis_coefs: Vec<f64>,
    is_in_basis: Vec<bool>,
}

/// Algorithm described in Introduction to Linear Optimization
/// by Dimitris Bertsimas & John N. Tsitsiklis
impl Simplex {
    fn auxiliary(constraint_matrix: Matrix, rhs: &[f64]) -> Self {
        let m = constraint_matrix.m();
        let n = constraint_matrix.n();

        let mut aux = Self {
            is_auxiliary: true,
            objective: Vec::with_capacity(m + n),
            constraints: Matrix::with_capacity(m),
            basis_inverse: Matrix::identity(m),
            basis_vars: Vec::with_capacity(m),
            basis_coefs: Vec::with_capacity(m),
            is_in_basis: Vec::with_capacity(m + n),
        };

        for _ in 0..n {
            aux.objective.push(0.0);
            aux.is_in_basis.push(false)
        }
        for i in 0..m {
            aux.objective.push(1.0);
            aux.is_in_basis.push(true);
            aux.basis_vars.push(n + i);
            aux.basis_coefs.push(rhs[i].abs());

            let mut row = if rhs[i] < 0.0 {
                constraint_matrix.row(i).iter().map(|x| -x).collect()
            } else {
                constraint_matrix.row(i)
            };
            row.resize(m + n, 0.0);
            row[n + i] = 1.0;
            aux.constraints.push(row);
        }
        aux
    }

    /// Fast because we are given `u` and do not need to compute it
    fn fast_pivot(&mut self, entering: usize, exiting: usize, u: &[f64]) {
        let theta = self.basis_coefs[exiting] / u[exiting];
        self.is_in_basis[self.basis_vars[exiting]] = false;
        self.is_in_basis[entering] = true;
        self.basis_vars[exiting] = entering;
        self.basis_coefs[exiting] = theta;
        for i in 0..self.basis_vars.len() {
            if i != exiting {
                self.basis_coefs[i] -= theta * u[i];
                self.basis_inverse
                    .row_operation(i, exiting, u[i] / u[exiting]);
            }
        }
        self.basis_inverse.scale_row(exiting, 1.0 / u[exiting]);
    }

    /// `entering` should be index `0..n` and `exiting` should be indexed `0..m`.
    /// In other words, `entering` indexes all variables but `exiting` only
    /// indexes into the basic variables.
    fn pivot(&mut self, entering: usize, exiting: usize) {
        let column = self.constraints.column(entering);
        let u = &self.basis_inverse.right_mul_by(&column);

        self.fast_pivot(entering, exiting, u);
    }

    /// To "expunge" a variable, we mark it as invalid in the mask.
    fn expunge_artificial_vars(&mut self, n: usize, mask: &mut [bool]) {
        if let Some((l, i)) = self
            .basis_vars
            .iter()
            .enumerate()
            .find(|&(_, &i)| i >= n && mask[i - n])
        {
            mask[i - n] = false;
            let row = self.constraints.left_mul_by(&self.basis_inverse.row(l));
            if let Some(j) = &row[..n]
                .iter()
                .enumerate()
                .find(|&(_, x)| x.abs() > f64::EPSILON)
                .map(|(j, _)| j)
            {
                self.pivot(*j, l)
            }
            self.expunge_artificial_vars(n, mask)
        }
    }

    /// Convert a Phase I auxiliary problem back to the original problem for Phase II.
    fn prepare_for_phase_two(mut self, objective: Vec<f64>) -> Self {
        assert!(
            self.is_auxiliary,
            "function should only be called on an auxiliary problem"
        );
        self.is_auxiliary = false;

        let n = objective.len();
        self.objective = objective;

        let mut mask = vec![true; self.constraints.m()];
        self.expunge_artificial_vars(n, &mut mask);

        self.basis_inverse.mask_rows(&mask);
        self.basis_inverse.mask_cols(&mask);

        self.constraints.mask_rows(&mask);
        self.constraints.truncate_and_shrink(n);

        self.is_in_basis.truncate(n);
        self.is_in_basis.shrink_to_fit();

        let mut i = mask.iter();
        self.basis_vars.retain(|_| *i.next().unwrap());
        self.basis_vars.shrink_to_fit();

        let mut j = mask.iter();
        self.basis_coefs.retain(|_| *j.next().unwrap());
        self.basis_coefs.shrink_to_fit();

        assert!(self.basis_vars.iter().all(|&x| x < n));
        self
    }

    fn p(&self) -> Vec<f64> {
        let masked_objective = self
            .basis_vars
            .iter()
            .map(|&i| self.objective[i])
            .collect::<Vec<f64>>();
        self.basis_inverse.left_mul_by(&masked_objective)
    }

    fn reduced_cost(&self, j: usize, p: &[f64], column_j: &[f64]) -> f64 {
        self.objective[j] - dot(p, column_j)
    }

    fn objective_value(&self) -> f64 {
        self.basis_vars
            .iter()
            .enumerate()
            .map(|(i, &j)| self.objective[j] * self.basis_coefs[i])
            .sum()
    }

    fn ratio_test(&self, u: &[f64]) -> Option<usize> {
        u.iter()
            .cloned()
            .zip(&self.basis_coefs)
            .enumerate()
            .map(|(x, (y, z))| (x, y, z))
            .filter(|&(_, y, _)| y > 0.0)
            .map(|(x, y, z)| (x, z / y))
            .min_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
            .map(|(x, _)| x)
    }

    fn solve(&mut self) -> Result<(f64, Vec<f64>), Error> {
        let p = self.p();
        for j in 0..self.constraints.n() {
            if !self.is_in_basis[j] {
                let column_j = self.constraints.column(j);
                if self.reduced_cost(j, &p, &column_j) < 0.0 {
                    let u = &self.basis_inverse.right_mul_by(&column_j);
                    return match self.ratio_test(u) {
                        None => Err(Error::Unbounded),
                        Some(l) => {
                            self.fast_pivot(j, l, u);
                            self.solve()
                        }
                    };
                };
            }
        }
        Ok((self.objective_value(), self.x()))
    }

    fn x(&self) -> Vec<f64> {
        let mut x = vec![0.0; self.constraints.n()];
        for (i, coef) in self.basis_vars.iter().zip(&self.basis_coefs) {
            x[*i] = *coef;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATOL: f64 = 1e-14;

    fn assert_approx_eq(x: &[f64], y: &[f64], atol: f64) {
        assert_eq!(x.len(), y.len());
        for (x_i, y_i) in x.iter().zip(y) {
            let diff = (x_i - y_i).abs();
            assert!(diff < atol)
        }
    }

    #[test]
    fn test_solve_1() {
        // Example 3.5 in Bertsimas & Tsitsiklis
        let objective = vec![-10.0, -12.0, -12.0, 0.0, 0.0, 0.0];
        let constraints = vec![
            vec![1.0, 2.0, 2.0, 1.0, 0.0, 0.0],
            vec![2.0, 1.0, 2.0, 0.0, 1.0, 0.0],
            vec![2.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        ];
        let rhs = vec![20.0, 20.0, 20.0];

        let (objective_value, x) = solve(objective, constraints, rhs).unwrap();
        assert_eq!(objective_value, -136.0);
        assert_approx_eq(&x, &[4.0, 4.0, 4.0, 0.0, 0.0, 0.0], ATOL);
    }

    #[test]
    fn test_solve_2() {
        // Example 3.8 in Bertsimas & Tsitsiklis
        let objective = vec![1.0, 1.0, 1.0, 0.0];
        let constraints = vec![
            vec![1.0, 2.0, 3.0, 0.0],
            vec![-1.0, 2.0, 6.0, 0.0],
            vec![0.0, 4.0, 9.0, 0.0],
            vec![0.0, 0.0, 3.0, 1.0],
        ];
        let rhs = vec![3.0, 2.0, 5.0, 1.0];

        let (objective_value, x) = solve(objective, constraints, rhs).unwrap();
        assert_eq!(objective_value, 1.75);
        assert_approx_eq(&x, &[0.5, 1.25, 0.0, 1.0], ATOL);
    }

    #[test]
    fn test_solve_3() {
        let objective = vec![1.0, 2.0, 3.0];
        let constraints = vec![];
        let rhs = vec![];

        let (objective_value, x) = solve(objective, constraints, rhs).unwrap();
        assert_eq!(objective_value, 0.0);
        assert_approx_eq(&x, &[0.0, 0.0, 0.0], ATOL);
    }

    #[test]
    fn test_solve_4() {
        let objective = vec![1.0, 2.0, -3.0];
        let constraints = vec![];
        let rhs = vec![];

        let result = solve(objective, constraints, rhs);
        assert_eq!(result.unwrap_err(), Error::Unbounded)
    }
}
