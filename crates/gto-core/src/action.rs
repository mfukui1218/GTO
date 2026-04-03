use std::fmt;

/// Betting actions a player can take.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Action {
    Fold,
    Check,
    Call,
    Bet(u32),
    Raise(u32),
    AllIn,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Fold => write!(f, "f"),
            Action::Check => write!(f, "x"),
            Action::Call => write!(f, "c"),
            Action::Bet(amt) => write!(f, "b{}", amt),
            Action::Raise(amt) => write!(f, "r{}", amt),
            Action::AllIn => write!(f, "a"),
        }
    }
}

/// The current street (betting round) of a hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}
