pub(crate) fn lu_solve(a: &CooMatrix, b: Vec<f64>) -> Vec<f64> {
    a.to_dense().factorize().solve(b)
}

/// Dense matrix with row-major order
struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn to_sparse(&self) -> CooMatrix {
        CooMatrix::from(self)
    }
}

impl From<&CooMatrix> for Matrix {
    fn from(coo: &CooMatrix) -> Self {
        let nrows = coo.nrows;
        let ncols = coo.ncols;

        let mut data = vec![0.0; nrows * ncols];
        for (i, j, val) in &coo.data {
            data[i * ncols + j] = *val
        }
        Self { nrows, ncols, data }
    }
}

impl Matrix {
    fn column(&self, j: usize) -> impl Iterator<Item = &f64> {
        self.data.iter().skip(j).step_by(self.ncols)
    }

    fn row(&self, i: usize) -> impl Iterator<Item = &f64> {
        self.data.iter().skip(self.ncols * i).take(self.ncols)
    }

    fn column_mut(&mut self, j: usize) -> impl Iterator<Item = &mut f64> {
        self.data.iter_mut().skip(j).step_by(self.ncols)
    }

    fn row_mut(&mut self, i: usize) -> impl Iterator<Item = &mut f64> {
        self.data.iter_mut().skip(self.ncols * i).take(self.ncols)
    }

    fn at(&self, i: usize, j: usize) -> &f64 {
        &self.data[self.raw_index(i, j)]
    }

    fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.ncols + j]
    }

    fn raw_index(&self, i: usize, j: usize) -> usize {
        i * self.ncols + j
    }

    /// Perform in-place LU decomposition.
    ///
    /// Golub, G., & Van Loan, C. (1996). Matrix Computations.
    /// The Johns Hopkins University Press.
    fn factorize(mut self) -> LU {
        assert_eq!(
            self.nrows, self.ncols,
            "non-square Matrix cannot be factorized; nrows={}, ncols={}",
            self.nrows, self.ncols
        );
        let n = self.nrows;
        let mut p = Vec::with_capacity(n - 1);

        for k in 0..n - 1 {
            let mut mu = k;
            let mut magnitude = self.at(k, k).abs();
            for i in k + 1..n {
                if self.at(i, k).abs() > magnitude {
                    mu = i;
                    magnitude = self.at(i, k).abs();
                }
            }

            for j in k..n {
                let x = self.raw_index(mu, j);
                let y = self.raw_index(k, j);
                self.data.swap(x, y);
            }
            p.push(mu);

            let pivot = *self.at(k, k);
            if pivot != 0.0 {
                for i in k + 1..n {
                    *self.at_mut(i, k) /= pivot;
                    for j in k + 1..n {
                        let adjustment = self.at(i, k) * self.at(k, j);
                        *self.at_mut(i, j) -= adjustment;
                    }
                }
            }
        }
        LU { p, matrix: self }
    }
}

/// Coordinate list format
pub(crate) struct CooMatrix {
    nrows: usize,
    ncols: usize,
    data: Vec<(usize, usize, f64)>,
}

impl CooMatrix {
    fn to_dense(&self) -> Matrix {
        Matrix::from(self)
    }
}

impl From<&Matrix> for CooMatrix {
    fn from(dense: &Matrix) -> Self {
        assert!(dense.nrows > 0);
        assert!(dense.ncols > 0);
        let mut data = vec![];
        let mut i = 0;
        let mut j = 0;
        for val in &dense.data {
            if *val != 0.0 {
                data.push((i, j, *val))
            }
            j += 1;
            if j == dense.ncols {
                j = 0;
                i += 1;
            }
        };
        Self {
            nrows: dense.nrows,
            ncols: dense.ncols,
            data,
        }
    }
}

struct LU {
    p: Vec<usize>,
    matrix: Matrix,
}

impl LU {
    /// Solve `self * x = b` for `x`.
    ///
    /// Golub, G., & Van Loan, C. (1996). Matrix Computations.
    /// The Johns Hopkins University Press.
    fn solve(&self, mut b: Vec<f64>) -> Vec<f64> {
        assert_eq!(self.matrix.ncols, b.len());
        let n = b.len();
        for k in 0..n - 1 {
            b.swap(k, self.p[k]);
            for i in k + 1..n {
                b[i] -= b[k] * self.matrix.at(i, k)
            }
        }
        for i in (0..self.matrix.nrows - 1).rev() {
            for j in i + 1..self.matrix.ncols {
                b[i] -= self.matrix.at(i, j) * b[j];
            }
            b[i] /= self.matrix.at(i, i)
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_row_column_access() {
        let matrix = Matrix {
            nrows: 2,
            ncols: 3,
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };

        assert_eq!(matrix.row(0).cloned().collect::<Vec<_>>(), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row(1).cloned().collect::<Vec<_>>(), &[4.0, 5.0, 6.0]);

        assert_eq!(matrix.column(0).cloned().collect::<Vec<_>>(), &[1.0, 4.0]);
        assert_eq!(matrix.column(1).cloned().collect::<Vec<_>>(), &[2.0, 5.0]);
        assert_eq!(matrix.column(2).cloned().collect::<Vec<_>>(), &[3.0, 6.0]);
    }

    #[test]
    fn test_lu_factorization() {
        let matrix = Matrix {
            nrows: 3,
            ncols: 3,
            data: vec![3.0, 17.0, 10.0, 2.0, 4.0, -2.0, 6.0, 18.0, -12.0],
        };
        let lu = matrix.factorize();
        assert_eq!(lu.p, &[2, 2]);
        assert_eq!(
            lu.matrix.data,
            vec![
                6.0,
                18.0,
                -12.0,
                1.0 / 3.0,
                8.0,
                16.0,
                1.0 / 2.0,
                -1.0 / 4.0,
                6.0
            ]
        );
    }

    #[test]
    fn test_matrix_roundtrip() {
        let matrix = Matrix {
            nrows: 2,
            ncols: 2,
            data: vec![0.0, 1.0, 0.0, 2.0],
        };
        let result = matrix.to_sparse().to_dense();
        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 2);
        assert_eq!(result.data, &[0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_lu_solve() {
        let a = Matrix {
            nrows: 3,
            ncols: 3,
            data: vec![6.0, 18.0, 3.0, 2.0, 12.0, 1.0, 4.0, 15.0, 3.0],
        }
        .to_sparse();
        let b = vec![3.0, 19.0, 0.0];
        let result = lu_solve(&a, b);
        assert_eq!(result, &[-3.0, 3.0, -11.0])
    }
}
