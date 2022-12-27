use std::fmt;

#[derive(Debug, PartialEq)]
pub enum Error {
    Unbounded,
    Infeasible,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unbounded => write!(f, "Error::UNBOUNDED"),
            Error::Infeasible => write!(f, "Error::INFEASIBLE"),
        }
    }
}
