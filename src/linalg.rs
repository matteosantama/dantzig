pub struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn new_unchecked(data: Vec<Vec<f64>>) -> Self {
        Self { data }
    }

    pub(crate) fn identity(size: usize) -> Self {
        let mut data = Vec::with_capacity(size);
        for i in 0..size {
            let mut row = vec![0.0; size];
            row[i] = 1.0;
            data.push(row)
        }
        Self { data }
    }

    pub fn m(&self) -> usize {
        self.data.len()
    }

    pub fn n(&self) -> usize {
        self.data[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub(crate) fn truncate_and_shrink(&mut self, len: usize) {
        for i in 0..self.data.len() {
            self.data[i].truncate(len);
            self.data[i].shrink_to_fit();
        }
    }

    pub(crate) fn push(&mut self, row: Vec<f64>) {
        self.data.push(row)
    }

    pub(crate) fn right_mul_by(&self, vector: &[f64]) -> Vec<f64> {
        assert_eq!(vector.len(), self.n());
        self.data.iter().map(|x| dot(x, vector)).collect()
    }

    pub(crate) fn left_mul_by(&self, vector: &[f64]) -> Vec<f64> {
        assert_eq!(vector.len(), self.m());
        let mut result = Vec::with_capacity(self.n());
        for j in 0..self.n() {
            let column = self.data.iter().map(|row| row[j]).collect::<Vec<f64>>();
            let element = dot(vector, &column);
            result.push(element);
        }
        result
    }

    pub(crate) fn column(&self, j: usize) -> Vec<f64> {
        self.data.iter().map(|x| x[j]).collect()
    }

    pub(crate) fn row(&self, i: usize) -> Vec<f64> {
        self.data[i].clone()
    }

    pub(crate) fn scale_row(&mut self, i: usize, scalar: f64) {
        self.row_operation(i, i, 1.0 - scalar)
    }

    pub(crate) fn row_operation(&mut self, target: usize, row: usize, scalar: f64) {
        assert!(!scalar.is_infinite() && !scalar.is_nan());
        let result = self.data[target]
            .iter()
            .zip(&self.data[row])
            .map(|(x, y)| x - scalar * y)
            .collect();
        self.data[target] = result;
    }

    pub(crate) fn mask_cols(&mut self, mask: &[bool]) {
        assert_eq!(mask.len(), self.n());
        for row in &mut self.data {
            let mut i = mask.iter();
            row.retain(|_| *i.next().unwrap());
            row.shrink_to_fit()
        }
    }

    pub(crate) fn mask_rows(&mut self, mask: &[bool]) {
        assert_eq!(mask.len(), self.m());
        let mut i = mask.iter();
        self.data.retain(|_| *i.next().unwrap());
        self.data.shrink_to_fit()
    }
}

#[cfg(feature = "simd")]
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    use std::simd::{f64x4, SimdFloat, StdFloat};

    assert_eq!(x.len(), y.len());

    let mut sum = x
        .array_chunks::<4>()
        .map(|&a| f64x4::from_array(a))
        .zip(y.array_chunks::<4>().map(|&b| f64x4::from_array(b)))
        .fold(f64x4::splat(0.), |acc, (a, b)| a.mul_add(b, acc))
        .reduce_sum();

    let remain = x.len() - (x.len() % 4);
    sum += x[remain..]
        .iter()
        .zip(&y[remain..])
        .map(|(a, b)| a * b)
        .sum::<f64>();
    sum
}

#[cfg(not(feature = "simd"))]
pub(crate) fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let x = &[1.0, 2.0, 3.0, 0.0, 1.0];
        let y = &[2.0, 0.0, 1.0, 0.0, 1.0];

        assert_eq!(dot(x, x), 15.0);
        assert_eq!(dot(x, y), 6.0);
        assert_eq!(dot(y, x), 6.0);
        assert_eq!(dot(y, y), 6.0);
    }

    #[test]
    fn test_column() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Matrix::new_unchecked(data);

        assert_eq!(matrix.column(0), &[1.0, 4.0]);
        assert_eq!(matrix.column(1), &[2.0, 5.0]);
        assert_eq!(matrix.column(2), &[3.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let matrix = Matrix::new_unchecked(data);

        assert_eq!(matrix.right_mul_by(&[2.0, 2.0, 3.0]), &[15.0, 36.0]);
        assert_eq!(matrix.left_mul_by(&[-2.0, 2.0]), &[6.0, 6.0, 6.0]);
    }

    #[test]
    fn test_scale_row() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut matrix = Matrix::new_unchecked(data);

        matrix.scale_row(1, 2.0);
        assert_eq!(matrix.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row(1), &[8.0, 10.0, 12.0])
    }

    #[test]
    fn test_row_operation() {
        let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut matrix = Matrix::new_unchecked(data);

        matrix.row_operation(1, 0, 3.0);
        assert_eq!(matrix.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(matrix.row(1), &[1.0, -1.0, -3.0]);
    }
}
