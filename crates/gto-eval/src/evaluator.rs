use gto_core::Card;
use std::sync::LazyLock;

/// Hand strength as a u32. Higher value = stronger hand.
///
/// Encoding:
/// - Bits 20-23: category (0=high card, 1=pair, ..., 8=straight flush)
/// - Bits 16-19: primary rank component
/// - Bits 12-15: secondary
/// - Bits 8-11: tertiary
/// - Bits 4-7: quaternary
/// - Bits 0-3: quinary
pub type HandStrength = u32;

pub const CAT_HIGH_CARD: u32 = 0;
pub const CAT_ONE_PAIR: u32 = 1;
pub const CAT_TWO_PAIR: u32 = 2;
pub const CAT_THREE_KIND: u32 = 3;
pub const CAT_STRAIGHT: u32 = 4;
pub const CAT_FLUSH: u32 = 5;
pub const CAT_FULL_HOUSE: u32 = 6;
pub const CAT_FOUR_KIND: u32 = 7;
pub const CAT_STRAIGHT_FLUSH: u32 = 8;

/// Get the category name from a hand strength value.
pub fn category_name(strength: HandStrength) -> &'static str {
    match strength >> 20 {
        CAT_STRAIGHT_FLUSH => "Straight Flush",
        CAT_FOUR_KIND => "Four of a Kind",
        CAT_FULL_HOUSE => "Full House",
        CAT_FLUSH => "Flush",
        CAT_STRAIGHT => "Straight",
        CAT_THREE_KIND => "Three of a Kind",
        CAT_TWO_PAIR => "Two Pair",
        CAT_ONE_PAIR => "One Pair",
        CAT_HIGH_CARD => "High Card",
        _ => "Unknown",
    }
}

#[inline(always)]
fn encode(cat: u32, r0: u8, r1: u8, r2: u8, r3: u8, r4: u8) -> HandStrength {
    (cat << 20)
        | ((r0 as u32) << 16)
        | ((r1 as u32) << 12)
        | ((r2 as u32) << 8)
        | ((r3 as u32) << 4)
        | (r4 as u32)
}

/// Detect highest straight from a 13-bit rank bitmask (bit 0 = rank 0 = '2', bit 12 = rank 12 = 'A').
/// Returns the top rank of the straight if found.
#[inline(always)]
fn highest_straight(rank_bits: u16) -> Option<u8> {
    // Check from highest possible top (A-high = rank 12) down to 6-high (rank 4)
    for top in (4u8..=12).rev() {
        let mask = 0x1Fu16 << (top - 4);
        if rank_bits & mask == mask {
            return Some(top);
        }
    }
    // Wheel: A-5-4-3-2 → bits 12,3,2,1,0 → 0x100F
    if rank_bits & 0x100F == 0x100F {
        return Some(3); // 5-high straight
    }
    None
}

/// Extract the top `n` set bit positions from a 13-bit rank bitmask, in descending order.
/// Returns an array of 5 u8 values; only the first `n` are meaningful.
#[inline(always)]
fn top_n_bits(mut bits: u16, n: usize) -> [u8; 5] {
    let mut result = [0u8; 5];
    for i in 0..n {
        // Find highest set bit. bits is at most 13 bits (0..8191).
        // We use 16-bit leading zeros: the highest bit in a 16-bit number at position k
        // has leading_zeros = 15 - k.
        let pos = 15 - bits.leading_zeros() as u8;
        result[i] = pos;
        bits ^= 1u16 << pos;
    }
    result
}

/// Evaluate a 5-card hand and return its strength.
/// No heap allocation; everything on the stack.
pub fn evaluate_5(cards: &[Card; 5]) -> HandStrength {
    let mut rank_counts = [0u8; 13];
    let mut rank_bits: u16 = 0;
    let suit0 = cards[0].suit();
    let mut is_flush = true;

    for c in cards {
        let r = c.rank() as usize;
        rank_counts[r] += 1;
        rank_bits |= 1u16 << r;
        if c.suit() != suit0 {
            is_flush = false;
        }
    }

    // Classify ranks by scanning high to low
    let mut quads: u8 = 0;
    let mut trips: u8 = 0;
    let mut pairs = [0u8; 2];
    let mut pair_count: usize = 0;
    let mut kickers_bits: u16 = 0;

    for r in (0..13u8).rev() {
        match rank_counts[r as usize] {
            4 => quads = r,
            3 => trips = r,
            2 => {
                if pair_count < 2 {
                    pairs[pair_count] = r;
                    pair_count += 1;
                }
            }
            1 => kickers_bits |= 1u16 << r,
            _ => {}
        }
    }

    let straight_top = highest_straight(rank_bits);
    let num_distinct = rank_bits.count_ones();

    // Straight flush
    if let Some(top) = straight_top {
        if is_flush {
            return encode(CAT_STRAIGHT_FLUSH, top, 0, 0, 0, 0);
        }
    }

    // Four of a kind (5 cards: 4+1 → 2 distinct ranks)
    if num_distinct == 2 && quads > 0 || rank_counts[quads as usize] == 4 {
        // quads > 0 check: if quads==0 (rank '2'), check count
        if rank_counts[quads as usize] == 4 {
            let k = top_n_bits(rank_bits ^ (1u16 << quads), 1);
            return encode(CAT_FOUR_KIND, quads, k[0], 0, 0, 0);
        }
    }

    // Full house (5 cards: 3+2 → 2 distinct)
    if trips > 0 || rank_counts[0] == 3 {
        let has_trips = if trips > 0 {
            true
        } else if rank_counts[0] == 3 {
            trips = 0;
            true
        } else {
            false
        };
        if has_trips && pair_count == 1 {
            return encode(CAT_FULL_HOUSE, trips, pairs[0], 0, 0, 0);
        }
    }

    // Flush
    if is_flush {
        let k = top_n_bits(rank_bits, 5);
        return encode(CAT_FLUSH, k[0], k[1], k[2], k[3], k[4]);
    }

    // Straight
    if let Some(top) = straight_top {
        return encode(CAT_STRAIGHT, top, 0, 0, 0, 0);
    }

    // Three of a kind (5 cards: 3+1+1 → 3 distinct)
    if rank_counts[trips as usize] == 3 && pair_count == 0 {
        let k = top_n_bits(kickers_bits, 2);
        return encode(CAT_THREE_KIND, trips, k[0], k[1], 0, 0);
    }

    // Two pair (5 cards: 2+2+1 → 3 distinct)
    if pair_count == 2 {
        let k = top_n_bits(kickers_bits, 1);
        return encode(CAT_TWO_PAIR, pairs[0], pairs[1], k[0], 0, 0);
    }

    // One pair (5 cards: 2+1+1+1 → 4 distinct)
    if pair_count == 1 {
        let k = top_n_bits(kickers_bits, 3);
        return encode(CAT_ONE_PAIR, pairs[0], k[0], k[1], k[2], 0);
    }

    // High card
    let k = top_n_bits(rank_bits, 5);
    encode(CAT_HIGH_CARD, k[0], k[1], k[2], k[3], k[4])
}

/// Evaluate rank-based (non-flush) strength from 7 cards.
/// Uses rank_counts and rank_bits already computed.
#[inline(always)]
fn evaluate_7_no_flush(rank_counts: &[u8; 13], rank_bits: u16) -> HandStrength {
    // Classify: scan high to low
    let mut quads: i8 = -1;
    let mut trips = [-1i8; 2];
    let mut trip_count: usize = 0;
    let mut pairs = [-1i8; 3];
    let mut pair_count: usize = 0;
    let mut kickers_bits: u16 = 0;

    for r in (0..13i8).rev() {
        match rank_counts[r as usize] {
            4 => quads = r,
            3 => {
                if trip_count < 2 {
                    trips[trip_count] = r;
                    trip_count += 1;
                }
            }
            2 => {
                if pair_count < 3 {
                    pairs[pair_count] = r;
                    pair_count += 1;
                }
            }
            1 => kickers_bits |= 1u16 << r,
            _ => {}
        }
    }

    // Four of a kind
    if quads >= 0 {
        let q = quads as u8;
        // Best kicker: highest rank among remaining 3 cards
        let remaining = rank_bits ^ (1u16 << q);
        let k = top_n_bits(remaining, 1);
        return encode(CAT_FOUR_KIND, q, k[0], 0, 0, 0);
    }

    // Full house variations
    if trip_count >= 1 {
        let t = trips[0] as u8; // highest trips
        if trip_count == 2 {
            // 3+3+1: second trips acts as pair
            return encode(CAT_FULL_HOUSE, t, trips[1] as u8, 0, 0, 0);
        }
        if pair_count >= 1 {
            // 3+2+... : use highest pair
            return encode(CAT_FULL_HOUSE, t, pairs[0] as u8, 0, 0, 0);
        }
    }

    // Straight
    if let Some(top) = highest_straight(rank_bits) {
        return encode(CAT_STRAIGHT, top, 0, 0, 0, 0);
    }

    // Three of a kind (no pairs, single trips, 7 cards: 3+1+1+1+1)
    if trip_count == 1 && pair_count == 0 {
        let t = trips[0] as u8;
        let k = top_n_bits(kickers_bits, 2);
        return encode(CAT_THREE_KIND, t, k[0], k[1], 0, 0);
    }

    // Two pair: pick best 2 pairs + best kicker
    if pair_count >= 2 {
        let p1 = pairs[0] as u8;
        let p2 = pairs[1] as u8;
        // Kicker: best from remaining singles + any third pair's rank
        let mut kicker_pool = kickers_bits;
        if pair_count >= 3 {
            kicker_pool |= 1u16 << pairs[2];
        }
        let k = top_n_bits(kicker_pool, 1);
        return encode(CAT_TWO_PAIR, p1, p2, k[0], 0, 0);
    }

    // One pair
    if pair_count == 1 {
        let p = pairs[0] as u8;
        let k = top_n_bits(kickers_bits, 3);
        return encode(CAT_ONE_PAIR, p, k[0], k[1], k[2], 0);
    }

    // High card
    let k = top_n_bits(rank_bits, 5);
    encode(CAT_HIGH_CARD, k[0], k[1], k[2], k[3], k[4])
}

/// Total number of 7-card rank distributions: C(19, 7) = 50,388.
const NUM_RANK_DISTRIBUTIONS: usize = 50_388;

/// Precomputed binomial coefficient table for C(n, k), n ≤ 19, k ≤ 7.
const BINOM: [[usize; 8]; 20] = build_binom_table();

const fn build_binom_table() -> [[usize; 8]; 20] {
    let mut table = [[0usize; 8]; 20];
    let mut n = 0;
    while n < 20 {
        table[n][0] = 1;
        let mut k = 1;
        while k < 8 && k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}

/// Compute combinatorial index for a 7-card rank distribution.
/// Maps rank_counts to a unique index in [0, 50387].
#[inline(always)]
fn rank_distribution_index(rank_counts: &[u8; 13]) -> usize {
    // Convert rank counts to a sorted sequence of 7 ranks, then to strictly increasing
    // x[i] = sorted_rank[i] + i, giving a unique combinatorial index.
    let mut index = 0usize;
    let mut pos = 0usize; // position in the 7-element sequence
    for r in 0..13u8 {
        let count = rank_counts[r as usize];
        for c in 0..count {
            let x = r as usize + pos;
            index += BINOM[x][pos + 1];
            pos += 1;
            let _ = c; // suppress warning
        }
    }
    index
}

/// Precomputed lookup table for non-flush 7-card evaluation (~200KB).
static RANK_LUT: LazyLock<Vec<HandStrength>> = LazyLock::new(|| build_rank_lut());

fn build_rank_lut() -> Vec<HandStrength> {
    let mut lut = vec![0u32; NUM_RANK_DISTRIBUTIONS];
    let mut counts = [0u8; 13];
    fill_rank_lut(&mut counts, 0, 7, &mut lut);
    lut
}

fn fill_rank_lut(
    counts: &mut [u8; 13],
    rank: usize,
    remaining: u8,
    lut: &mut Vec<HandStrength>,
) {
    if rank == 13 {
        if remaining == 0 {
            let idx = rank_distribution_index(counts);
            let mut rank_bits: u16 = 0;
            for r in 0..13 {
                if counts[r] > 0 {
                    rank_bits |= 1u16 << r;
                }
            }
            lut[idx] = evaluate_7_no_flush(counts, rank_bits);
        }
        return;
    }
    let max_count = remaining.min(4);
    for c in 0..=max_count {
        counts[rank] = c;
        fill_rank_lut(counts, rank + 1, remaining - c, lut);
    }
    counts[rank] = 0;
}

/// Evaluate the best 5-card hand from 7 cards.
/// Uses a precomputed lookup table for non-flush hands (~200KB, O(1) access).
pub fn evaluate_7(cards: &[Card; 7]) -> HandStrength {
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    let mut suit_rank_bits = [0u16; 4];

    for c in cards {
        let r = c.rank();
        let s = c.suit() as usize;
        rank_counts[r as usize] += 1;
        suit_counts[s] += 1;
        suit_rank_bits[s] |= 1u16 << r;
    }

    // Check for flush (5+ cards of same suit)
    let mut flush_strength: HandStrength = 0;
    for s in 0..4 {
        if suit_counts[s] >= 5 {
            let sbits = suit_rank_bits[s];
            // Check straight flush
            if let Some(top) = highest_straight(sbits) {
                flush_strength = encode(CAT_STRAIGHT_FLUSH, top, 0, 0, 0, 0);
            } else {
                // Regular flush: top 5 cards of this suit
                let k = top_n_bits(sbits, 5);
                flush_strength = encode(CAT_FLUSH, k[0], k[1], k[2], k[3], k[4]);
            }
            break; // At most one flush suit in 7 cards
        }
    }

    // Non-flush: O(1) lookup from precomputed table
    let rank_strength = RANK_LUT[rank_distribution_index(&rank_counts)];

    // Return the better of flush and rank-based hands
    if flush_strength > rank_strength {
        flush_strength
    } else {
        rank_strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: &str) -> Card {
        let bytes = s.as_bytes();
        let rank = match bytes[0] {
            b'2' => 0,
            b'3' => 1,
            b'4' => 2,
            b'5' => 3,
            b'6' => 4,
            b'7' => 5,
            b'8' => 6,
            b'9' => 7,
            b'T' => 8,
            b'J' => 9,
            b'Q' => 10,
            b'K' => 11,
            b'A' => 12,
            _ => panic!("bad rank"),
        };
        let suit = match bytes[1] {
            b'c' => 0,
            b'd' => 1,
            b'h' => 2,
            b's' => 3,
            _ => panic!("bad suit"),
        };
        Card::new(rank, suit)
    }

    // --- Reference implementations for exhaustive testing ---

    fn detect_straight_ref(sorted: &[u8; 5]) -> Option<u8> {
        if sorted[0] - sorted[4] == 4
            && sorted[1] == sorted[0] - 1
            && sorted[2] == sorted[0] - 2
            && sorted[3] == sorted[0] - 3
        {
            return Some(sorted[0]);
        }
        if sorted[0] == 12 && sorted[1] == 3 && sorted[2] == 2 && sorted[3] == 1 && sorted[4] == 0
        {
            return Some(3);
        }
        None
    }

    fn evaluate_5_reference(cards: &[Card; 5]) -> HandStrength {
        let mut ranks = [0u8; 5];
        let mut suits = [0u8; 5];
        for (i, c) in cards.iter().enumerate() {
            ranks[i] = c.rank();
            suits[i] = c.suit();
        }

        let is_flush = suits[0] == suits[1]
            && suits[1] == suits[2]
            && suits[2] == suits[3]
            && suits[3] == suits[4];

        let mut counts = [0u8; 13];
        for &r in &ranks {
            counts[r as usize] += 1;
        }

        let mut freq: Vec<(u8, u8)> = Vec::with_capacity(5);
        for (r, &c) in counts.iter().enumerate() {
            if c > 0 {
                freq.push((c, r as u8));
            }
        }
        freq.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

        ranks.sort_unstable();
        ranks.reverse();

        let straight_top = detect_straight_ref(&ranks);
        let max_count = freq[0].0;
        let num_groups = freq.len();

        if let Some(top) = straight_top {
            if is_flush {
                return encode(CAT_STRAIGHT_FLUSH, top, 0, 0, 0, 0);
            }
        }

        if max_count == 4 {
            return encode(CAT_FOUR_KIND, freq[0].1, freq[1].1, 0, 0, 0);
        }

        if max_count == 3 && num_groups == 2 {
            return encode(CAT_FULL_HOUSE, freq[0].1, freq[1].1, 0, 0, 0);
        }

        if is_flush {
            return encode(
                CAT_FLUSH, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4],
            );
        }

        if let Some(top) = straight_top {
            return encode(CAT_STRAIGHT, top, 0, 0, 0, 0);
        }

        if max_count == 3 {
            return encode(CAT_THREE_KIND, freq[0].1, freq[1].1, freq[2].1, 0, 0);
        }

        if max_count == 2 && num_groups == 3 {
            return encode(CAT_TWO_PAIR, freq[0].1, freq[1].1, freq[2].1, 0, 0);
        }

        if max_count == 2 {
            return encode(
                CAT_ONE_PAIR,
                freq[0].1,
                freq[1].1,
                freq[2].1,
                freq[3].1,
                0,
            );
        }

        encode(
            CAT_HIGH_CARD,
            ranks[0],
            ranks[1],
            ranks[2],
            ranks[3],
            ranks[4],
        )
    }

    fn evaluate_7_reference(cards: &[Card; 7]) -> HandStrength {
        let mut best: HandStrength = 0;
        for skip1 in 0..7usize {
            for skip2 in (skip1 + 1)..7 {
                let mut hand = [Card(0); 5];
                let mut idx = 0;
                for i in 0..7 {
                    if i != skip1 && i != skip2 {
                        hand[idx] = cards[i];
                        idx += 1;
                    }
                }
                let s = evaluate_5_reference(&hand);
                if s > best {
                    best = s;
                }
            }
        }
        best
    }

    // --- Original tests ---

    #[test]
    fn royal_flush() {
        let hand = [c("As"), c("Ks"), c("Qs"), c("Js"), c("Ts")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_STRAIGHT_FLUSH);
        assert_eq!(category_name(s), "Straight Flush");
    }

    #[test]
    fn wheel_straight_flush() {
        let hand = [c("5h"), c("4h"), c("3h"), c("2h"), c("Ah")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_STRAIGHT_FLUSH);
        let six_high = [c("6h"), c("5h"), c("4h"), c("3h"), c("2h")];
        assert!(evaluate_5(&six_high) > s);
    }

    #[test]
    fn four_of_a_kind() {
        let hand = [c("9s"), c("9h"), c("9d"), c("9c"), c("Ac")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_FOUR_KIND);
        let aces = [c("As"), c("Ah"), c("Ad"), c("Ac"), c("Ks")];
        assert!(evaluate_5(&aces) > s);
    }

    #[test]
    fn full_house() {
        let hand = [c("Ks"), c("Kh"), c("Kd"), c("7s"), c("7h")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_FULL_HOUSE);
        let qqq_aa = [c("Qs"), c("Qh"), c("Qd"), c("As"), c("Ah")];
        assert!(s > evaluate_5(&qqq_aa));
    }

    #[test]
    fn flush() {
        let hand = [c("Ad"), c("Kd"), c("Qd"), c("Jd"), c("9d")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_FLUSH);
        let lower = [c("Ad"), c("Kd"), c("Qd"), c("Jd"), c("8d")];
        assert!(s > evaluate_5(&lower));
    }

    #[test]
    fn straight() {
        let hand = [c("Ts"), c("9h"), c("8d"), c("7c"), c("6s")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_STRAIGHT);
    }

    #[test]
    fn wheel_straight() {
        let hand = [c("Ac"), c("2d"), c("3h"), c("4s"), c("5c")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_STRAIGHT);
        let six_high = [c("6c"), c("2d"), c("3h"), c("4s"), c("5c")];
        assert!(evaluate_5(&six_high) > s);
    }

    #[test]
    fn three_of_a_kind() {
        let hand = [c("8s"), c("8h"), c("8d"), c("Ac"), c("Kc")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_THREE_KIND);
    }

    #[test]
    fn two_pair() {
        let hand = [c("Ks"), c("Kh"), c("7d"), c("7c"), c("As")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_TWO_PAIR);
        let lower = [c("Ks"), c("Kh"), c("6d"), c("6c"), c("As")];
        assert!(s > evaluate_5(&lower));
    }

    #[test]
    fn one_pair() {
        let hand = [c("Qs"), c("Qh"), c("Ad"), c("Kc"), c("Js")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_ONE_PAIR);
    }

    #[test]
    fn high_card() {
        let hand = [c("As"), c("Kh"), c("Qd"), c("Jc"), c("9s")];
        let s = evaluate_5(&hand);
        assert_eq!(s >> 20, CAT_HIGH_CARD);
    }

    #[test]
    fn category_ordering() {
        let sf = evaluate_5(&[c("As"), c("Ks"), c("Qs"), c("Js"), c("Ts")]);
        let fk = evaluate_5(&[c("As"), c("Ah"), c("Ad"), c("Ac"), c("Ks")]);
        let fh = evaluate_5(&[c("As"), c("Ah"), c("Ad"), c("Ks"), c("Kh")]);
        let fl = evaluate_5(&[c("As"), c("Ks"), c("Qs"), c("Js"), c("9s")]);
        let st = evaluate_5(&[c("As"), c("Kh"), c("Qd"), c("Jc"), c("Ts")]);
        let tk = evaluate_5(&[c("As"), c("Ah"), c("Ad"), c("Ks"), c("Qh")]);
        let tp = evaluate_5(&[c("As"), c("Ah"), c("Ks"), c("Kh"), c("Qd")]);
        let op = evaluate_5(&[c("As"), c("Ah"), c("Ks"), c("Qh"), c("Jd")]);
        let hc = evaluate_5(&[c("As"), c("Kh"), c("Qd"), c("Jc"), c("9h")]);

        assert!(sf > fk);
        assert!(fk > fh);
        assert!(fh > fl);
        assert!(fl > st);
        assert!(st > tk);
        assert!(tk > tp);
        assert!(tp > op);
        assert!(op > hc);
    }

    #[test]
    fn evaluate_7_finds_best() {
        let cards = [
            c("As"),
            c("Ks"),
            c("Qs"),
            c("Js"),
            c("9s"),
            c("2d"),
            c("3c"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FLUSH);

        let cards2 = [
            c("Ts"),
            c("9h"),
            c("8d"),
            c("7c"),
            c("6s"),
            c("2d"),
            c("3c"),
        ];
        let s2 = evaluate_7(&cards2);
        assert_eq!(s2 >> 20, CAT_STRAIGHT);
    }

    #[test]
    fn evaluate_7_hidden_full_house() {
        let cards = [
            c("Ks"),
            c("7h"),
            c("Kh"),
            c("Kd"),
            c("7d"),
            c("7c"),
            c("2s"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FULL_HOUSE);
    }

    // --- New edge case tests for 7-card evaluation ---

    #[test]
    fn evaluate_7_quads_plus_trips() {
        // 4+3 pattern: KKKK + 777
        let cards = [
            c("Ks"),
            c("Kh"),
            c("Kd"),
            c("Kc"),
            c("7s"),
            c("7h"),
            c("7d"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FOUR_KIND);
        // Kicker should be 7 (the trip rank)
        let expected = encode(CAT_FOUR_KIND, 11, 5, 0, 0, 0);
        assert_eq!(s, expected);
    }

    #[test]
    fn evaluate_7_two_trips() {
        // 3+3+1 pattern: KKK + 777 + A → FH KKK-77
        let cards = [
            c("Ks"),
            c("Kh"),
            c("Kd"),
            c("7s"),
            c("7h"),
            c("7d"),
            c("As"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FULL_HOUSE);
        let expected = encode(CAT_FULL_HOUSE, 11, 5, 0, 0, 0);
        assert_eq!(s, expected);
    }

    #[test]
    fn evaluate_7_trips_two_pairs() {
        // 3+2+2 pattern: KKK + 88 + 55 → FH KKK-88
        let cards = [
            c("Ks"),
            c("Kh"),
            c("Kd"),
            c("8s"),
            c("8h"),
            c("5s"),
            c("5h"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FULL_HOUSE);
        let expected = encode(CAT_FULL_HOUSE, 11, 6, 0, 0, 0);
        assert_eq!(s, expected);
    }

    #[test]
    fn evaluate_7_three_pairs() {
        // 2+2+2+1 pattern: AA + KK + QQ + J → 2P AA-KK, kicker Q
        let cards = [
            c("As"),
            c("Ah"),
            c("Ks"),
            c("Kh"),
            c("Qs"),
            c("Qh"),
            c("Jd"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_TWO_PAIR);
        // Top 2 pairs: A,K. Kicker: Q (from third pair, higher than J)
        let expected = encode(CAT_TWO_PAIR, 12, 11, 10, 0, 0);
        assert_eq!(s, expected);
    }

    #[test]
    fn evaluate_7_flush_vs_full_house() {
        // 7 cards where both flush and full house are possible
        // FH should win over flush
        let cards = [
            c("As"),
            c("Ah"),
            c("Ad"),
            c("Ks"),
            c("Kh"),
            c("Qs"),
            c("Js"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FULL_HOUSE);
    }

    #[test]
    fn evaluate_7_straight_flush_in_7() {
        // SF within 7 cards
        let cards = [
            c("9s"),
            c("8s"),
            c("7s"),
            c("6s"),
            c("5s"),
            c("Ad"),
            c("Kd"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_STRAIGHT_FLUSH);
    }

    #[test]
    fn evaluate_7_wheel_sf() {
        // Wheel straight flush in 7 cards
        let cards = [
            c("Ah"),
            c("2h"),
            c("3h"),
            c("4h"),
            c("5h"),
            c("Kd"),
            c("Qd"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_STRAIGHT_FLUSH);
        let expected = encode(CAT_STRAIGHT_FLUSH, 3, 0, 0, 0, 0);
        assert_eq!(s, expected);
    }

    #[test]
    fn evaluate_7_six_flush_cards() {
        // 6 cards of same suit → best 5 flush
        let cards = [
            c("As"),
            c("Ks"),
            c("Qs"),
            c("Js"),
            c("9s"),
            c("8s"),
            c("2d"),
        ];
        let s = evaluate_7(&cards);
        assert_eq!(s >> 20, CAT_FLUSH);
        let expected = encode(CAT_FLUSH, 12, 11, 10, 9, 7);
        assert_eq!(s, expected);
    }

    // --- Exhaustive 5-card verification ---

    #[test]
    fn exhaustive_evaluate_5() {
        // Test all C(52,5) = 2,598,960 hands
        let mut count = 0u64;
        let mut mismatches = 0u64;
        for a in 0..52u8 {
            for b in (a + 1)..52 {
                for cc in (b + 1)..52 {
                    for d in (cc + 1)..52 {
                        for e in (d + 1)..52 {
                            let cards = [Card(a), Card(b), Card(cc), Card(d), Card(e)];
                            let new_val = evaluate_5(&cards);
                            let ref_val = evaluate_5_reference(&cards);
                            if new_val != ref_val {
                                mismatches += 1;
                                if mismatches <= 5 {
                                    eprintln!(
                                        "MISMATCH: {:?} → new=0x{:08X} ref=0x{:08X}",
                                        cards, new_val, ref_val
                                    );
                                }
                            }
                            count += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(count, 2_598_960);
        assert_eq!(
            mismatches, 0,
            "{} mismatches in {} hands",
            mismatches, count
        );
    }

    // --- Random sample 7-card verification ---

    #[test]
    fn sample_evaluate_7() {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(12345);
        let mut deck: Vec<Card> = (0..52).map(Card).collect();

        let num_samples = 1_000_000;
        let mut mismatches = 0u64;

        for _ in 0..num_samples {
            deck.shuffle(&mut rng);
            let cards: [Card; 7] = [deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6]];
            let new_val = evaluate_7(&cards);
            let ref_val = evaluate_7_reference(&cards);
            if new_val != ref_val {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "7-CARD MISMATCH: {:?} → new=0x{:08X} ref=0x{:08X}",
                        cards, new_val, ref_val
                    );
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{} mismatches in {} 7-card hands",
            mismatches, num_samples
        );
    }
}
