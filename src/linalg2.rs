use std::collections::HashSet;

pub(crate) fn lu_solve() -> Result<Vec<f64>, ()> {
    todo!()
}

// struct CscVec<T> {
//     data: Vec<T>,
//     index: Vec<usize>,
// }
//
// impl<T: Clone> CscVec<T> where f64: From<T> {
//     fn new(values: &[T]) -> Self {
//         let mut data = vec![];
//         let mut index = vec![];
//
//         for (i, &x) in values.iter().enumerate() {
//             if f64::from(x) != 0.0 {
//                 data.push(x);
//                 index.push(i);
//             }
//         }
//
//         Self { data, index }
//     }
// }

pub(crate) struct CscMat<T> {
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    data: Vec<T>,
}

impl<T: Clone> CscMat<T> {
    /// Supply matrix entries by (row, column, value).
    fn from_coordinate_list(coords: &[(usize, usize, T)]) -> Self {
        debug_assert!(!coords.is_empty());
        debug_assert!(row_cols_unique(coords));

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

    fn factorize(&self) -> LU {
        // let permutation = vec![];
        // let lower = vec![];
        // let upper = vec![];

        todo!()
    }
}

fn row_cols_unique<T>(coords: &[(usize, usize, T)]) -> bool {
    let mut unique = HashSet::new();
    for (row, col, _) in coords {
        if unique.contains(&(row, col)) {
            return false;
        }
        unique.insert((row, col));
    }
    true
}

impl<T> CscMat<T> {
    fn nrows(&self) -> usize {
        self.row_idx.last().map(|i| i + 1).unwrap_or(0)
    }

    fn ncols(&self) -> usize {
        self.col_ptr.len() - 1
    }

    fn nnz(&self) -> usize {
        self.data.len()
    }
}

struct LU {
    permutation: CscMat<i8>,
    lower: CscMat<f64>,
    upper: CscMat<f64>,
}

// impl LU {
//     fn from_coordinate_lists()
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_coordinate_list() {
        // Example taken from https://scipy-lectures.org/advanced/scipy_sparse/csc_matrix.html
        let csc = CscMat::from_coordinate_list(&[
            (0, 0, 1),
            (0, 2, 2),
            (1, 2, 3),
            (2, 0, 4),
            (2, 1, 5),
            (2, 2, 6),
        ]);
        assert_eq!(csc.data, &[1, 4, 5, 2, 3, 6]);
        assert_eq!(csc.row_idx, &[0, 2, 2, 0, 1, 2]);
        assert_eq!(csc.col_ptr, &[0, 2, 3, 6]);
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 6);
    }
}
