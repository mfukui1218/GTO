//! Suit isomorphism (canonicalization).
//!
//! In poker, boards/hands that differ only by suit relabeling are strategically
//! identical. For example, `A♠ K♥ 2♣` and `A♦ K♣ 2♥` yield the same optimal
//! strategy — the "names" of the suits don't matter.
//!
//! This module provides canonicalization functions that remap suits to a unique
//! representative per equivalence class. The canonical form assigns suit IDs in
//! the order they first appear when cards are listed in their given order, so:
//!
//! - `A♠ K♥ 2♣` → suits remapped as ♠→0, ♥→1, ♣→2  → canonical `A♣ K♦ 2♥`
//!   (using the convention: new suit 0 = clubs, 1 = diamonds, 2 = hearts, 3 = spades)
//! - `A♦ K♣ 2♥` → suits remapped as ♦→0, ♣→1, ♥→2  → canonical `A♣ K♦ 2♥`
//!
//! Both iso flops collapse to the same canonical form.
//!
//! Callers that need to canonicalize a board together with private hole cards
//! must use [`canonicalize_with_hand`] so the same suit permutation is applied
//! to the hand.

use crate::card::Card;

/// A suit permutation: `perm[old_suit] = new_suit`.
///
/// `255` means "not yet assigned". After canonicalization all 4 suits should
/// have a valid mapping (we fill unassigned suits with the remaining IDs in
/// ascending order).
pub type SuitPermutation = [u8; 4];

/// Apply a suit permutation to a card, producing a new card with the mapped suit.
#[inline]
pub fn apply_perm(card: Card, perm: &SuitPermutation) -> Card {
    let new_suit = perm[card.suit() as usize];
    debug_assert!(new_suit < 4, "unmapped suit in permutation");
    Card::new(card.rank(), new_suit)
}

/// Compute the suit permutation that canonicalizes a sequence of cards.
///
/// Suits are assigned new IDs in the order of first appearance. Suits that do
/// not appear in `cards` are filled with the lowest unused IDs in ascending
/// order of their original index, so the result is always a complete 4-element
/// permutation.
///
/// # Example
/// ```
/// use gto_core::Card;
/// use gto_core::iso::build_perm;
/// // A♠ K♥ 2♣ — suits ♠=3, ♥=2, ♣=0 appear in order ♠,♥,♣.
/// let cards = [Card::new(12, 3), Card::new(11, 2), Card::new(0, 0)];
/// let perm = build_perm(&cards);
/// assert_eq!(perm[3], 0); // ♠ → new 0
/// assert_eq!(perm[2], 1); // ♥ → new 1
/// assert_eq!(perm[0], 2); // ♣ → new 2
/// assert_eq!(perm[1], 3); // ♦ → new 3 (unused, gets remainder)
/// ```
pub fn build_perm(cards: &[Card]) -> SuitPermutation {
    let mut perm: SuitPermutation = [255; 4];
    let mut next_id: u8 = 0;

    for &c in cards {
        let s = c.suit() as usize;
        if perm[s] == 255 {
            perm[s] = next_id;
            next_id += 1;
        }
    }

    // Fill remaining suits with the next available IDs in ascending order
    // of original suit index, so the permutation is deterministic and complete.
    for s in 0..4 {
        if perm[s] == 255 {
            perm[s] = next_id;
            next_id += 1;
        }
    }

    perm
}

/// Canonicalize a sequence of cards by remapping their suits.
/// Returns the canonical cards and the permutation used.
///
/// The returned cards have the same ranks as the input; only suits are remapped.
/// Applying the same permutation to any related cards (e.g. hole cards) via
/// [`apply_perm`] preserves the shared structure.
pub fn canonicalize(cards: &[Card]) -> (Vec<Card>, SuitPermutation) {
    let perm = build_perm(cards);
    let canon: Vec<Card> = cards.iter().map(|&c| apply_perm(c, &perm)).collect();
    (canon, perm)
}

/// Canonicalize a 3-card flop.
#[inline]
pub fn canonicalize_flop(flop: [Card; 3]) -> ([Card; 3], SuitPermutation) {
    let perm = build_perm(&flop);
    (
        [
            apply_perm(flop[0], &perm),
            apply_perm(flop[1], &perm),
            apply_perm(flop[2], &perm),
        ],
        perm,
    )
}

/// Canonicalize a board **jointly** with a 2-card hand so both are remapped by
/// the same suit permutation.
///
/// The board's suit structure drives the canonicalization — the hand is
/// whatever the board says. This is the correct transform for converting a
/// "hand vs. board" situation into a canonical information set: the hand
/// relative to the board is preserved.
pub fn canonicalize_with_hand(board: &[Card], hand: &[Card; 2]) -> (Vec<Card>, [Card; 2], SuitPermutation) {
    let perm = build_perm(board);
    let canon_board: Vec<Card> = board.iter().map(|&c| apply_perm(c, &perm)).collect();
    let canon_hand = [apply_perm(hand[0], &perm), apply_perm(hand[1], &perm)];
    (canon_board, canon_hand, perm)
}

/// Pack up to 5 cards into a single `u64` key for use as a cache key.
/// Cards are sorted ascending before packing, so the key is independent of
/// input order. Bytes 5-7 are 0.
///
/// Intended for storing bucketing caches keyed by (canonical) board state.
pub fn pack_cards_u64(cards: &[Card]) -> u64 {
    debug_assert!(cards.len() <= 5, "pack_cards_u64 supports up to 5 cards");
    let mut bytes = [0u8; 8];
    for (i, &c) in cards.iter().enumerate() {
        bytes[i] = c.0;
    }
    // Sort the populated prefix so the key is order-independent.
    let len = cards.len();
    bytes[..len].sort_unstable();
    u64::from_le_bytes(bytes)
}

/// Canonicalize a board and pack it into a cache key.
/// Iso boards collapse to the same key.
pub fn canonical_cache_key(board: &[Card]) -> u64 {
    let (canon, _) = canonicalize(board);
    pack_cards_u64(&canon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card::new(rank, suit)
    }

    #[test]
    fn perm_assigns_in_order() {
        // ♠=3 first, then ♥=2, then ♣=0
        let cards = [c(12, 3), c(11, 2), c(0, 0)];
        let perm = build_perm(&cards);
        assert_eq!(perm[3], 0);
        assert_eq!(perm[2], 1);
        assert_eq!(perm[0], 2);
        assert_eq!(perm[1], 3); // unused ♦ → last
    }

    #[test]
    fn canonicalize_is_idempotent() {
        let cards = [c(12, 3), c(11, 2), c(0, 0), c(5, 0)];
        let (canon1, _) = canonicalize(&cards);
        let (canon2, _) = canonicalize(&canon1);
        assert_eq!(canon1, canon2);
    }

    #[test]
    fn iso_flops_collapse() {
        // A♠ K♥ 2♣
        let flop1 = [c(12, 3), c(11, 2), c(0, 0)];
        // A♦ K♣ 2♥
        let flop2 = [c(12, 1), c(11, 0), c(0, 2)];
        let (canon1, _) = canonicalize_flop(flop1);
        let (canon2, _) = canonicalize_flop(flop2);
        assert_eq!(canon1, canon2);
    }

    #[test]
    fn non_iso_flops_do_not_collapse() {
        // A♠ K♠ 2♣ (two-tone) vs A♠ K♥ 2♣ (rainbow) — different structures
        let flop1 = [c(12, 3), c(11, 3), c(0, 0)];
        let flop2 = [c(12, 3), c(11, 2), c(0, 0)];
        let (canon1, _) = canonicalize_flop(flop1);
        let (canon2, _) = canonicalize_flop(flop2);
        assert_ne!(canon1, canon2);
    }

    #[test]
    fn monotone_flops_collapse() {
        // A♠ K♠ 2♠ and A♥ K♥ 2♥ — both monotone, iso under suit swap
        let flop1 = [c(12, 3), c(11, 3), c(0, 3)];
        let flop2 = [c(12, 2), c(11, 2), c(0, 2)];
        let (canon1, _) = canonicalize_flop(flop1);
        let (canon2, _) = canonicalize_flop(flop2);
        assert_eq!(canon1, canon2);
    }

    #[test]
    fn hand_preserved_relative_to_board() {
        // Board A♠ K♥ 2♣, hand 7♦ 7♠ (one blocker on ♠ A)
        let board1 = [c(12, 3), c(11, 2), c(0, 0)];
        let hand1 = [c(5, 1), c(5, 3)];

        // Iso board A♦ K♣ 2♥, iso hand 7♥ 7♦ (same structure: one non-flop-suit pair
        // member, one ♦-blocker... wait need to map properly)
        // Let's compute directly: hand must map such that its "role" relative to
        // the board is preserved.
        let (cb1, ch1, _) = canonicalize_with_hand(&board1, &hand1);

        // Now start with the same canonical board and hand, apply a different
        // permutation to get iso input, then canonicalize again — should match.
        let board2 = [c(12, 1), c(11, 0), c(0, 2)]; // A♦ K♣ 2♥
        // Applying the inverse of board1's iso perm, the suits ♠→?, ♥→?, ♣→?
        // Board1 perm: ♠→0, ♥→1, ♣→2, ♦→3
        // Board2 perm: ♦→0, ♣→1, ♥→2, ♠→3
        // So the map board1 → board2 is: ♠(0)→♠(new 3), ♥(1)→♥(new 2), ♣(2)→♣(new 1)
        // For hand1 = [7♦, 7♠] (1, 3), under board1 perm it becomes [7(3→?), 7(0)]:
        //   perm1 = [2, 3, 1, 0], so 7♦(s=1)→new 3, 7♠(s=3)→new 0
        //   ch1 = [Card(rank=5, suit=3), Card(rank=5, suit=0)]
        // Under board2: ♦→0, ♣→1, ♥→2, ♠→3
        //   perm2 = [1, 0, 2, 3], so 7♦(s=1)→new 0, 7♠(s=3)→new 3
        //   we want the hand to produce the same canonical form as ch1, which
        //   is (5, suit=3) and (5, suit=0). Under perm2, hand must have 7 with
        //   old-suit 3 (to get new 0) and 7 with old-suit 2 (to get new... wait)
        //
        // The pairing that survives the relabeling: in the canonical hand we had
        // one 7 paired with the first-appearing board suit (the "A-suit") and
        // one 7 on the fourth (unused) suit. For board2, the A-suit is ♦, so
        // the equivalent hand2 is 7♦ paired with... hmm, the fourth unused suit
        // of board2 is ♠ (since ♦♣♥ are used and ♠ is free). So hand2 = [7♦ pair, 7 of fourth (♠)]
        //   = [c(5,1), c(5,3)]  — wait that's the same hand because ♠ and ♦ are
        // both present in both boards just at different positions.
        //
        // Let's just verify the *other* direction: canonicalize both board+hand
        // pairs that should be iso and check equality.
        let hand2 = [c(5, 1), c(5, 3)]; // 7♦ 7♠ — this is iso to hand1 under the board iso
        let (cb2, ch2, _) = canonicalize_with_hand(&board2, &hand2);
        assert_eq!(cb1, cb2, "iso boards should canonicalize identically");
        // Hand cards are unordered — compare as sorted pairs.
        let mut sorted1 = ch1;
        sorted1.sort();
        let mut sorted2 = ch2;
        sorted2.sort();
        assert_eq!(sorted1, sorted2, "iso hands on iso boards should canonicalize identically");
    }

    #[test]
    fn pack_cards_order_independent() {
        let a = [c(12, 3), c(11, 2), c(0, 0)];
        let b = [c(0, 0), c(12, 3), c(11, 2)];
        assert_eq!(pack_cards_u64(&a), pack_cards_u64(&b));
    }

    #[test]
    fn canonical_cache_key_matches_iso_flops() {
        let flop1 = [c(12, 3), c(11, 2), c(0, 0)];
        let flop2 = [c(12, 1), c(11, 0), c(0, 2)];
        assert_eq!(canonical_cache_key(&flop1), canonical_cache_key(&flop2));
    }
}
