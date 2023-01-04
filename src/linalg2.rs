use std::mem::swap;

pub(crate) fn lu_solve() -> Result<Vec<f64>, ()> {
    todo!()
}

/// Dense matrix with row-major order
struct Matrix {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl From<&CscMatrix> for Matrix {
    fn from(csc: &CscMatrix) -> Self {
        let nrows = csc.nrows();
        let ncols = csc.ncols();

        let mut data = vec![0.0; nrows * ncols];
        for j in 0..ncols {
            for (i, val) in csc.column(j) {
                data[i * ncols + j] = *val
            }
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
        &self.data[i * self.ncols + j]
    }

    fn at_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.ncols + j]
    }

    fn partial_row_swap(&mut self, mu: usize, k: usize) {
        todo!()
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
                let x = *self.at(mu, j);
                let y = *self.at(k, j);
                *self.at_mut(mu, j) = y;
                *self.at_mut(k, j) = x;
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

#[derive(Clone)]
pub(crate) struct CscMatrix {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    data: Vec<f64>,
}

impl CscMatrix {
    /// Supply matrix entries by (row, column, value).
    fn from_coordinates(coords: &[(usize, usize, f64)]) -> Self {
        assert!(!coords.is_empty(), "must provide coordinates");

        let mut owned = coords.to_vec();
        owned.sort_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));

        let capacity = owned.len();

        let mut col_ptr = Vec::with_capacity(capacity);
        let mut row_idx = Vec::with_capacity(capacity);
        let mut data = Vec::with_capacity(capacity);

        let mut opt_prev_col = None;
        for (row, col, val) in owned {
            if let Some(prev) = opt_prev_col {
                if col != prev {
                    col_ptr.push(data.len());
                    opt_prev_col = Some(col)
                }
            } else {
                col_ptr.push(data.len());
                opt_prev_col = Some(0);
            }
            row_idx.push(row);
            data.push(val);
        }
        col_ptr.push(data.len());

        row_idx.shrink_to_fit();
        col_ptr.shrink_to_fit();

        Self {
            col_ptr,
            row_idx,
            data,
        }
    }

    fn nrows(&self) -> usize {
        self.row_idx.iter().cloned().max().unwrap_or(0)
    }

    fn ncols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.column(j)
            .find(|(row, _)| *row == i)
            .map(|(_, &e)| e)
            .unwrap_or(0.0)
    }

    fn row_slice(&self, j: usize) -> &[usize] {
        &self.row_idx[self.col_ptr[j]..self.col_ptr[j + 1]]
    }

    fn data_slice(&self, j: usize) -> &[f64] {
        &self.data[self.col_ptr[j]..self.col_ptr[j + 1]]
    }

    fn column(&self, j: usize) -> impl Iterator<Item = (usize, &f64)> {
        self.row_slice(j).iter().cloned().zip(self.data_slice(j))
    }

    // fn column_mut(&self, j: usize) -> impl Iterator<Item = (usize, &mut f64)> {
    //     self.row_slice(j)
    //         .iter()
    //         .cloned()
    //         .zip(self.data_slice(j).iter_mut())
    // }
}

struct LU {
    p: Vec<usize>,
    matrix: Matrix,
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

    // #[test]
    // fn test_csc_from_coordinates() {
    //     // Example taken from https://scipy-lectures.org/advanced/scipy_sparse/csc_matrix.html
    //     let csc = CscMatrix::from_coordinates(&[
    //         (0, 0, 1.0),
    //         (0, 2, 2.0),
    //         (1, 2, 3.0),
    //         (2, 0, 4.0),
    //         (2, 1, 5.0),
    //         (2, 2, 6.0),
    //     ]);
    //     assert_eq!(csc.data, &[1.0, 4.0, 5.0, 2.0, 3.0, 6.0]);
    //     assert_eq!(csc.row_idx, &[0, 2, 2, 0, 1, 2]);
    //     assert_eq!(csc.col_ptr, &[0, 2, 3, 6]);
    //     assert_eq!(csc.nrows(), 3);
    //     assert_eq!(csc.ncols(), 3);
    //     assert_eq!(csc.nnz(), 6);
    // }
}
