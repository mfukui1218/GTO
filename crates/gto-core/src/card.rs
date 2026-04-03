use rand::seq::SliceRandom;
use std::fmt;

/// A card represented as rank * 4 + suit (0..52).
/// Ranks: 0=2, 1=3, ..., 8=T, 9=J, 10=Q, 11=K, 12=A
/// Suits: 0=clubs, 1=diamonds, 2=hearts, 3=spades
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Card(pub u8);

impl Card {
    pub const COUNT: usize = 52;

    pub fn new(rank: u8, suit: u8) -> Self {
        debug_assert!(rank < 13 && suit < 4);
        Card(rank * 4 + suit)
    }

    pub fn rank(self) -> u8 {
        self.0 / 4
    }

    pub fn suit(self) -> u8 {
        self.0 % 4
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const RANKS: &[u8] = b"23456789TJQKA";
        const SUITS: &[u8] = b"cdhs";
        let r = RANKS[self.rank() as usize] as char;
        let s = SUITS[self.suit() as usize] as char;
        write!(f, "{}{}", r, s)
    }
}

/// A set of cards represented as a 64-bit bitmask for O(1) set operations.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct CardSet(pub u64);

impl CardSet {
    pub fn empty() -> Self {
        CardSet(0)
    }

    pub fn insert(&mut self, card: Card) {
        self.0 |= 1u64 << card.0;
    }

    pub fn remove(&mut self, card: Card) {
        self.0 &= !(1u64 << card.0);
    }

    pub fn contains(self, card: Card) -> bool {
        (self.0 >> card.0) & 1 != 0
    }

    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    pub fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }

    pub fn intersects(self, other: CardSet) -> bool {
        (self.0 & other.0) != 0
    }

    pub fn iter(self) -> CardSetIter {
        CardSetIter(self.0)
    }
}

pub struct CardSetIter(u64);

impl Iterator for CardSetIter {
    type Item = Card;

    fn next(&mut self) -> Option<Card> {
        if self.0 == 0 {
            None
        } else {
            let bit = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1; // clear lowest set bit
            Some(Card(bit))
        }
    }
}

/// A deck of cards that can be shuffled and dealt from.
pub struct Deck {
    cards: Vec<Card>,
    pos: usize,
}

impl Deck {
    pub fn full() -> Self {
        let cards = (0..52).map(Card).collect();
        Deck { cards, pos: 0 }
    }

    pub fn shuffle(&mut self, rng: &mut impl rand::Rng) {
        self.cards.shuffle(rng);
        self.pos = 0;
    }

    pub fn deal(&mut self) -> Card {
        let card = self.cards[self.pos];
        self.pos += 1;
        card
    }

    pub fn remaining(&self) -> usize {
        self.cards.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_rank_suit() {
        let c = Card::new(12, 3); // Ace of spades
        assert_eq!(c.rank(), 12);
        assert_eq!(c.suit(), 3);
        assert_eq!(format!("{}", c), "As");
    }

    #[test]
    fn card_display() {
        assert_eq!(format!("{}", Card::new(0, 0)), "2c");
        assert_eq!(format!("{}", Card::new(9, 2)), "Jh");
        assert_eq!(format!("{}", Card::new(10, 1)), "Qd");
    }

    #[test]
    fn cardset_operations() {
        let mut set = CardSet::empty();
        let c1 = Card::new(0, 0);
        let c2 = Card::new(12, 3);

        set.insert(c1);
        set.insert(c2);
        assert!(set.contains(c1));
        assert!(set.contains(c2));
        assert_eq!(set.count(), 2);

        set.remove(c1);
        assert!(!set.contains(c1));
        assert_eq!(set.count(), 1);
    }

    #[test]
    fn cardset_iter() {
        let mut set = CardSet::empty();
        set.insert(Card(0));
        set.insert(Card(5));
        set.insert(Card(51));

        let cards: Vec<Card> = set.iter().collect();
        assert_eq!(cards.len(), 3);
        assert_eq!(cards[0], Card(0));
        assert_eq!(cards[1], Card(5));
        assert_eq!(cards[2], Card(51));
    }
}
