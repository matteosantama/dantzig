/// Solve the equation `a * x = b` for `x` with an LU factorization.
pub(crate) fn lu_solve(a: &CscMatrix, b: Vec<f64>) -> Vec<f64> {
    a.to_dense().factorize().solve(b)
}

/// Solve the equation `a.t() * x = b` for `x` with an LU factorization.
pub(crate) fn lu_solve_t(a: &CscMatrix, b: Vec<f64>) -> Vec<f64> {
    a.to_dense().t().factorize().solve(b)
}

/// Dense matrix with row-major order
struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn to_sparse(&self) -> CscMatrix {
        CscMatrix::from(self)
    }

    fn zero(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            data: vec![0.0; nrows * ncols],
        }
    }
}

impl From<&CscMatrix> for Matrix {
    fn from(sparse: &CscMatrix) -> Self {
        let mut dense = Matrix::zero(sparse.nrows, sparse.ncols);

        for (i, j, val) in sparse.iter() {
            *dense.at_mut(i, j) = *val
        }
        dense
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
    /// The resulting dense matrix encodes both the the L (lower) and U (upper) factors.
    ///
    /// See Golub, G., & Van Loan, C. (1996). Matrix Computations. The Johns Hopkins University Press
    /// for further details.
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

    fn t(&self) -> Self {
        let mut data = Vec::with_capacity(self.nrows * self.ncols);
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                data.push(*self.at(i, j));
            }
        }
        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            data,
        }
    }
}

/// Sparse matrix in CSC (Compressed Sparse Column) format
pub(crate) struct CscMatrix {
    nrows: usize,
    ncols: usize,
    row_idx: Vec<usize>,
    col_ptr: Vec<usize>,
    data: Vec<f64>,
}

impl CscMatrix {
    pub(crate) fn column(&self, j: usize) -> Vec<f64> {
        let mut col = vec![0.0; self.nrows];
        for (i, val) in self.column_iter(j) {
            col[i] = *val
        }
        col
    }

    pub(crate) fn collect_columns(&self, cols: &[usize]) -> Self {
        let cols = cols.iter().map(|j| self.column(*j)).collect::<Vec<_>>();
        Self::from_cols(self.nrows, cols.len(), &cols)
    }

    pub(crate) fn nrows(&self) -> usize {
        self.nrows
    }

    pub(crate) fn ncols(&self) -> usize {
        self.ncols
    }

    /// An efficient utility function for computing -self^T *  v.
    ///
    /// A CSC matrix has efficient column access, so rather than compute
    /// -self^T * v, it is easier to compute -v^T *self if we don't care
    /// about the orientation of the resulting vector.
    pub(crate) fn neg_t_dot(&self, v: Vec<f64>) -> Vec<f64> {
        assert_eq!(self.nrows, v.len());
        let mut result = Vec::with_capacity(v.len());
        for j in 0..self.ncols {
            let x = self.column_iter(j).map(|(i, val)| *val * -v[i]).sum();
            result.push(x);
        }
        result
    }

    fn new(nrows: usize, ncols: usize) -> CscMatrix {
        Self {
            nrows,
            ncols,
            row_idx: vec![],
            col_ptr: vec![0],
            data: vec![],
        }
    }

    fn from_cols(nrows: usize, ncols: usize, cols: &[Vec<f64>]) -> Self {
        assert_eq!(ncols, cols.len());
        let mut sparse = Self::new(nrows, ncols);

        for col in cols.iter() {
            assert_eq!(nrows, col.len());
            for (i, val) in col.iter().enumerate() {
                if *val != 0.0 {
                    sparse.row_idx.push(i);
                    sparse.data.push(*val);
                }
            }
            sparse.col_ptr.push(sparse.data.len());
        }
        sparse
    }

    fn to_dense(&self) -> Matrix {
        Matrix::from(self)
    }

    /// Iterate over the non-zero entries of the j'th column, yielding the row index and value.
    fn column_iter(&self, j: usize) -> impl Iterator<Item = (usize, &f64)> {
        self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
            .iter()
            .cloned()
            .zip(self.data[self.col_ptr[j]..self.col_ptr[j + 1]].iter())
    }

    /// Iterate over the non-zero values of the matrix, providing their (row, col) index too.
    fn iter(&self) -> impl Iterator<Item = (usize, usize, &f64)> {
        (0..self.ncols).flat_map(|j| self.column_iter(j).map(move |(i, val)| (i, j, val)))
    }
}

impl From<&Matrix> for CscMatrix {
    fn from(dense: &Matrix) -> Self {
        let mut sparse = CscMatrix::new(dense.nrows, dense.ncols);

        for j in 0..dense.ncols {
            for i in 0..dense.nrows {
                let val = dense.at(i, j);
                if *val != 0.0 {
                    sparse.row_idx.push(i);
                    sparse.data.push(*val);
                }
            }
            sparse.col_ptr.push(sparse.data.len());
        }
        sparse
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

    #[test]
    fn test_csc_from_dense() {
        let sparse = Matrix {
            nrows: 3,
            ncols: 3,
            data: vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0],
        }
        .to_sparse();
        assert_eq!(sparse.row_idx, &[0, 2, 2, 0, 1, 2]);
        assert_eq!(sparse.col_ptr, &[0, 2, 3, 6]);
        assert_eq!(sparse.data, &[1.0, 4.0, 5.0, 2.0, 3.0, 6.0])
    }

    #[test]
    fn test_csc_column() {
        let sparse = Matrix {
            nrows: 3,
            ncols: 3,
            data: vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0],
        }
        .to_sparse();
        assert_eq!(sparse.column(0), &[1.0, 0.0, 4.0]);
        assert_eq!(sparse.column(1), &[0.0, 0.0, 5.0]);
        assert_eq!(sparse.column(2), &[2.0, 3.0, 6.0]);
    }

    #[test]
    fn test_csc_collect_columns() {
        let sparse = Matrix {
            nrows: 3,
            ncols: 3,
            data: vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0],
        }
        .to_sparse();
        let cols = &[1, 2, 0];
        let result = sparse.collect_columns(cols);
        assert_eq!(sparse.column(0), result.column(2));
        assert_eq!(sparse.column(1), result.column(0));
        assert_eq!(sparse.column(2), result.column(1));
    }

    #[test]
    fn test_dense_transpose() {
        let result = Matrix {
            nrows: 2,
            ncols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        }
        .t();
        assert_eq!(result.column(0).cloned().collect::<Vec<_>>(), &[1.0, 2.0]);
        assert_eq!(result.column(1).cloned().collect::<Vec<_>>(), &[3.0, 4.0]);
    }

    #[test]
    fn test_neg_transpose_dot() {
        let sparse = Matrix {
            nrows: 3,
            ncols: 4,
            data: (0..12).map(|x| x as f64).collect()
        }.to_sparse();
        let v = vec![1.0, 2.0, 3.0];
        let result = sparse.neg_t_dot(v);
        assert_eq!(result, &[-32.0, -38.0, -44.0, -50.0]);
    }
}
