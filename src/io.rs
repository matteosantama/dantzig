use std::cmp::{min, Ordering};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::io::Lines;
use std::io::{BufRead, BufReader, Read};
use std::iter::Peekable;
use std::ops::Index;
use std::path::Path;

/// Refer to http://users.clas.ufl.edu/hager/coap/mps_format for details.
struct MPS {
    name: String,
    rows: Rows,
    columns: Columns,
    rhs: RHS,
    bounds: Bounds,
}

struct Solution {
    objective_value: f64,
    variables: HashMap<String, f64>,
}

impl Index<&str> for Solution {
    type Output = f64;

    fn index(&self, index: &str) -> &Self::Output {
        &self.variables[index]
    }
}

struct Rows {
    objective: String,
    equations: HashMap<String, Ordering>,
}

impl Rows {
    fn new() -> Self {
        Self {
            objective: "".to_string(),
            equations: HashMap::new(),
        }
    }

    fn store_ordering(&mut self, name: String, ordering: Ordering) {
        self.equations.insert(name, ordering);
    }

    fn store_objective(&mut self, name: String) {
        self.objective = name
    }
}

struct Columns {
    data: HashMap<(String, String), f64>,
}

impl Columns {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn store(&mut self, column_name: String, row_name: String, value: f64) {
        self.data.insert((row_name, column_name), value);
    }
}

struct RHS {
    data: HashMap<(String, String), f64>,
}

impl RHS {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn store(&mut self, rhs_name: String, row_name: String, value: f64) {
        self.data.insert((rhs_name, row_name), value);
    }
}

enum Bound {
    LO(f64),
    UP(f64),
    FR,
    FX(f64),
}

struct Bounds {
    // COLUMN_NAME -> BOUND_NAME -> BOUND
    data: HashMap<String, HashMap<String, Vec<Bound>>>,
}

impl Bounds {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn store(&mut self, bound_type: Bound, bound_name: String, column_name: String) {
        match self.data.entry(column_name) {
            Entry::Occupied(mut column_entry) => match column_entry.get_mut().entry(bound_name) {
                Entry::Occupied(mut bound_entry) => bound_entry.get_mut().push(bound_type),
                Entry::Vacant(bound_entry) => {
                    bound_entry.insert(vec![bound_type]);
                }
            },
            Entry::Vacant(column_entry) => {
                column_entry.insert(HashMap::from([(bound_name, vec![bound_type])]));
            }
        }
    }
}

fn read_rows_section<B: BufRead>(rows: &mut Rows, lines: &mut Peekable<Lines<B>>) {
    while let Some(peek) = lines.peek().map(|line| line.as_ref().unwrap()) {
        if !peek.starts_with(' ') {
            return;
        }

        let line = lines.next().unwrap().unwrap();

        let name = line[4..min(12, line.len())].to_string();

        match &line[1..2] {
            "E" => rows.store_ordering(name, Ordering::Equal),
            "L" => rows.store_ordering(name, Ordering::Less),
            "G" => rows.store_ordering(name, Ordering::Greater),
            "N" => rows.store_objective(name),
            _ => unimplemented!(),
        };
    }
}

fn read_columns_section<B: BufRead>(columns: &mut Columns, lines: &mut Peekable<Lines<B>>) {
    while let Some(peek) = lines.peek().map(|line| line.as_ref().unwrap()) {
        if !peek.starts_with(' ') {
            return;
        }

        let line = lines.next().unwrap().unwrap();
        let column_name = line[4..12].to_string();
        let row_name = line[14..22].to_string();
        let value = line[24..36].trim().parse::<f64>().unwrap();

        columns.store(column_name, row_name, value);

        if line.len() > 40 {
            let column_name = line[4..12].to_string();
            let row_name = line[39..47].to_string();
            let value = line[49..61].trim().parse::<f64>().unwrap();

            columns.store(column_name, row_name, value);
        }
    }
}

fn read_rhs_section<B: BufRead>(rhs: &mut RHS, lines: &mut Peekable<Lines<B>>) {
    while let Some(peek) = lines.peek().map(|line| line.as_ref().unwrap()) {
        if !peek.starts_with(' ') {
            return;
        }

        let line = lines.next().unwrap().unwrap();
        let rhs_name = line[4..12].to_string();
        let row_name = line[14..22].to_string();
        let value = line[24..36].trim().parse::<f64>().unwrap();

        rhs.store(rhs_name, row_name, value);

        if line.len() > 40 {
            let rhs_name = line[4..12].to_string();
            let row_name = line[39..47].to_string();
            let value = line[59..61].trim().parse::<f64>().unwrap();

            rhs.store(rhs_name, row_name, value);
        }
    }
}

fn read_bounds_section<B: BufRead>(bounds: &mut Bounds, lines: &mut Peekable<Lines<B>>) {
    while let Some(peek) = lines.peek().map(|line| line.as_ref().unwrap()) {
        if !peek.starts_with(' ') {
            return;
        }

        let line = lines.next().unwrap().unwrap();

        let bound = || line[24..36].trim().parse::<f64>().unwrap();

        let bound_type = match &line[1..3] {
            "LO" => Bound::LO(bound()),
            "UP" => Bound::UP(bound()),
            "FR" => Bound::FR,
            "FX" => Bound::FX(bound()),
            _ => unimplemented!("{} is not supported", &line[1..3]),
        };
        let bound_name = line[4..12].to_string();
        let column_name = line[14..22].to_string();

        bounds.store(bound_type, bound_name, column_name);
    }
}

impl MPS {
    fn new(name: String, rows: Rows, columns: Columns, rhs: RHS, bounds: Bounds) -> Self {
        Self {
            name,
            rows,
            columns,
            rhs,
            bounds,
        }
    }

    fn solve(self) -> Solution {
        todo!()
    }

    pub fn read<R: Read>(src: R) -> Self {
        let reader = BufReader::new(src);

        let mut lines = reader.lines().peekable();

        let mut name = String::new();
        let mut rows = Rows::new();
        let mut columns = Columns::new();
        let mut rhs = RHS::new();
        let mut bounds = Bounds::new();

        while let Some(result) = lines.next() {
            if let Ok(line) = result {
                if !line.is_empty() && !line.starts_with('*') {
                    match line.as_str() {
                        "ROWS" => read_rows_section(&mut rows, &mut lines),
                        "COLUMNS" => read_columns_section(&mut columns, &mut lines),
                        "RHS" => read_rhs_section(&mut rhs, &mut lines),
                        "BOUNDS" => read_bounds_section(&mut bounds, &mut lines),
                        "ENDATA" => return Self::new(name, rows, columns, rhs, bounds),
                        _ if line.starts_with("NAME") => {
                            let end = min(line.len(), 25);
                            name = line[15..end].to_string()
                        }
                        _ => unimplemented!("section {line} is not supported"),
                    }
                }
            }
        }
        panic!("no ENDATA section encountered");
    }

    fn write<P: AsRef<Path>>(&self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::MPS;

    #[test]
    fn test_read() {
        let raw = r#"
NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1
    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1
    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA
"#;
        let solution = MPS::read(raw.as_bytes()).solve();
        assert_eq!(solution.objective_value, 54.0);
        assert_eq!(solution["XONE"], 4.0);
        assert_eq!(solution["YTWO"], -1.0);
        assert_eq!(solution["ZTHREE"], 6.0);
    }
}
