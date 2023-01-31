use crate::error::Error;
use crate::model::{AffExpr, Inequality, LinExpr};
use crate::pyobjs::Variable;
use crate::simplex::Simplex;
use std::cmp::{min, Ordering};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
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

impl Solution {
    fn new(simplex: Simplex, variables: HashMap<String, Variable>) -> Self {
        Self {
            objective_value: simplex.objective_value(),
            variables: variables
                .into_iter()
                .map(|(name, variable)| (name, simplex.solution()[&variable.id]))
                .collect(),
        }
    }
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
    // ROW_NAME -> COLUMN_NAME -> VALUE
    data: HashMap<String, HashMap<String, f64>>,
    columns: HashSet<String>,
}

impl Columns {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
            columns: HashSet::new(),
        }
    }

    fn store(&mut self, column: String, row: String, value: f64) {
        match self.data.entry(row) {
            Entry::Occupied(mut row_entry) => match row_entry.get_mut().entry(column.clone()) {
                Entry::Occupied(_) => panic!("duplicate coefficient specification detected"),
                Entry::Vacant(column_entry) => {
                    column_entry.insert(value);
                }
            },
            Entry::Vacant(row_entry) => {
                row_entry.insert(HashMap::from([(column.clone(), value)]));
            }
        }
        self.columns.insert(column);
    }

    fn fetch(&self, column: &str, row: &str) -> f64 {
        self.data[row].get(column).cloned().unwrap_or_default()
    }
}

struct RHS {
    data: HashMap<String, f64>,
}

impl RHS {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn store(&mut self, row: String, value: f64) {
        match self.data.entry(row) {
            Entry::Occupied(_) => panic!("duplicate RHS specifications"),
            Entry::Vacant(entry) => entry.insert(value),
        };
    }

    fn fetch(&self, row: &str) -> f64 {
        self.data[row]
    }
}

#[derive(Debug)]
enum Bound {
    LO(f64),
    UP(f64),
    FR,
    FX(f64),
}

struct Bounds {
    data: HashMap<String, Vec<Bound>>,
}

impl Bounds {
    fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    fn store(&mut self, bound_type: Bound, column_name: String) {
        self.data.entry(column_name).or_default().push(bound_type)
    }

    fn fetch(&self, column: &str) -> Option<&Vec<Bound>> {
        self.data.get(column)
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
        let column_name = line[4..12].trim().to_string();
        let row_name = line[14..22].trim().to_string();
        let value = line[24..36].trim().parse::<f64>().unwrap();

        columns.store(column_name, row_name, value);

        if line.len() > 40 {
            let column_name = line[4..12].trim().to_string();
            let row_name = line[39..47].trim().to_string();
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
        let row = line[14..22].trim().to_string();
        let value = line[24..36].trim().parse::<f64>().unwrap();

        rhs.store(row, value);

        if line.len() > 40 {
            let row = line[39..47].trim().to_string();
            let value = line[59..61].trim().parse::<f64>().unwrap();

            rhs.store(row, value);
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
        let column_name = line[14..22].trim().to_string();

        bounds.store(bound_type, column_name);
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

    /// Create a map of variable *names* to variable *objects*.
    ///
    /// Note that multiple bounds are allowed for a single variable, but
    /// the most restrictive will be used (the lowest upper bound, for example).
    /// If no bounds are provided, the variable is assumed to be free.
    fn initialize_variables(&self) -> HashMap<String, Variable> {
        self.columns
            .columns
            .iter()
            .map(|column| {
                let variable = match self.bounds.fetch(column) {
                    None => Variable::free(),
                    Some(bounds) => {
                        let mut lb = None;
                        let mut ub = None;

                        let mut set_lb = |x| {
                            let bound = match lb {
                                None => x,
                                Some(z) => f64::max(z, x),
                            };
                            lb = Some(bound)
                        };

                        let mut set_ub = |x| {
                            let bound = match ub {
                                None => x,
                                Some(z) => f64::min(z, x),
                            };
                            ub = Some(bound)
                        };

                        for bound in bounds {
                            match bound {
                                Bound::LO(x) => set_lb(*x),
                                Bound::UP(x) => set_ub(*x),
                                Bound::FR => {}
                                Bound::FX(x) => {
                                    set_lb(*x);
                                    set_ub(*x);
                                }
                            }
                        }
                        Variable::new(lb, ub)
                    }
                };
                (column.to_owned(), variable)
            })
            .collect()
    }

    fn initialize_objective(
        &self,
        variables: &HashMap<String, Variable>,
        order: &[&String],
    ) -> AffExpr {
        let linexpr = self.linexpr(&self.rows.objective, variables, order);
        AffExpr::from(linexpr)
    }

    fn initialize_constraints(
        &self,
        variables: &HashMap<String, Variable>,
        order: &[&String],
    ) -> Vec<Inequality> {
        let mut constraints = Vec::with_capacity(self.rows.equations.len());
        for (row, ordering) in &self.rows.equations {
            let linexpr = self.linexpr(&row, variables, order);
            let rhs = self.rhs.fetch(row);
            match ordering {
                Ordering::Less => constraints.push(Inequality::less_than_eq(linexpr, rhs)),
                Ordering::Equal => {
                    constraints.push(Inequality::less_than_eq(linexpr.clone(), rhs));
                    constraints.push(Inequality::greater_than_eq(linexpr, rhs));
                }
                Ordering::Greater => constraints.push(Inequality::greater_than_eq(linexpr, rhs)),
            }
        }
        constraints
    }

    fn linexpr(
        &self,
        row: &str,
        variables: &HashMap<String, Variable>,
        order: &[&String],
    ) -> LinExpr {
        let terms = order
            .iter()
            .map(|column| {
                let variable = variables.get(*column).unwrap();
                let coef = self.columns.fetch(column, row);
                (coef, variable)
            })
            .collect::<Vec<_>>();
        LinExpr::from(terms.as_slice())
    }

    fn solve(self) -> Result<Solution, Error> {
        let variables = self.initialize_variables();
        let order = variables.keys().collect::<Vec<_>>();
        let objective = self.initialize_objective(&variables, &order);
        let constraints = self.initialize_constraints(&variables, &order);
        Simplex::new(objective, constraints)
            .solve()
            .map(|simplex| Solution::new(simplex, variables))
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
    fn test_mps_read() {
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
        let solution = MPS::read(raw.as_bytes()).solve().unwrap();
        assert_eq!(solution.objective_value, 80.0);
        assert_eq!(solution["XONE"], 4.0);
        assert_eq!(solution["YTWO"], 1.0);
        assert_eq!(solution["ZTHREE"], 8.0);
    }
}
