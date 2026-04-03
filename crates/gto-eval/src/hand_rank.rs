use crate::evaluator::{self, HandStrength};
use std::fmt;

/// A poker hand rank. Higher internal value = stronger hand.
/// Use comparison operators directly: stronger > weaker.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandRank(pub HandStrength);

impl HandRank {
    /// Get the hand category (0-8).
    pub fn category(self) -> u32 {
        self.0 >> 20
    }

    /// Get the category name (e.g., "Full House").
    pub fn category_name(self) -> &'static str {
        evaluator::category_name(self.0)
    }
}

impl fmt::Debug for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HandRank({}, 0x{:06X})", self.category_name(), self.0)
    }
}

impl fmt::Display for HandRank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.category_name())
    }
}
