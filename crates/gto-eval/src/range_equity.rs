use crate::equity::equity_exact;
use crate::evaluator::{category_name, evaluate_7};
use gto_core::{Card, CardSet};
use rand::seq::SliceRandom;
use rayon::prelude::*;

pub const NUM_CLASSES: usize = 169;
const RANKS: &[u8] = b"23456789TJQKA";

/// Result of range vs range equity calculation.
pub struct RangeEquityResult {
    /// Hero's overall equity 0.0-1.0
    pub equity: f64,
    /// Number of combo pairs evaluated
    pub total_matchups: u64,
    /// Per-class equity for hero (indexed 0-168)
    pub class_equity: [f64; NUM_CLASSES],
    /// Per-class total weight (available combos considered)
    pub class_weight: [f64; NUM_CLASSES],
}

/// Board texture analysis.
pub struct BoardTexture {
    pub paired: bool,
    pub two_tone: bool,
    pub monotone: bool,
    pub straight_possible: bool,
    pub high_card: String,
    pub hand_category: String,
}

/// Expand a hand class index (0-168) to all concrete card combos,
/// removing dead cards that overlap with the board or other dead set.
pub fn expand_class_combos(
    class: usize,
    frequency: f64,
    dead: CardSet,
) -> Vec<([Card; 2], f64)> {
    if frequency <= 0.0 {
        return Vec::new();
    }

    let (rank_a, rank_b, suited) = class_to_ranks(class);
    let mut combos = Vec::new();

    if rank_a == rank_b {
        // Pair: C(4,2) = 6 combos
        for s1 in 0u8..4 {
            let c1 = Card::new(rank_a, s1);
            if dead.contains(c1) {
                continue;
            }
            for s2 in (s1 + 1)..4 {
                let c2 = Card::new(rank_a, s2);
                if dead.contains(c2) {
                    continue;
                }
                combos.push(([c1, c2], frequency));
            }
        }
    } else if suited {
        // Suited: 4 combos (one per suit)
        for s in 0u8..4 {
            let c1 = Card::new(rank_a, s);
            let c2 = Card::new(rank_b, s);
            if dead.contains(c1) || dead.contains(c2) {
                continue;
            }
            combos.push(([c1, c2], frequency));
        }
    } else {
        // Offsuit: 4*3 = 12 combos
        for s1 in 0u8..4 {
            let c1 = Card::new(rank_a, s1);
            if dead.contains(c1) {
                continue;
            }
            for s2 in 0u8..4 {
                if s1 == s2 {
                    continue;
                }
                let c2 = Card::new(rank_b, s2);
                if dead.contains(c2) {
                    continue;
                }
                combos.push(([c1, c2], frequency));
            }
        }
    }

    combos
}

/// Convert class index to (high_rank, low_rank, suited).
fn class_to_ranks(class: usize) -> (u8, u8, bool) {
    if class < 13 {
        // Pair
        (class as u8, class as u8, false)
    } else if class < 91 {
        // Suited
        let idx = class - 13;
        let (low, high) = index_to_ranks(idx);
        (high, low, true)
    } else {
        // Offsuit
        let idx = class - 91;
        let (low, high) = index_to_ranks(idx);
        (high, low, false)
    }
}

/// Reverse the triangle index to get (low_rank, high_rank).
fn index_to_ranks(index: usize) -> (u8, u8) {
    let mut high = 1u8;
    while ((high + 1) as usize) * (high as usize) / 2 <= index {
        high += 1;
    }
    let low = index - (high as usize) * (high as usize - 1) / 2;
    (low as u8, high)
}

/// Get the name for a hand class index.
pub fn class_index_to_name(class: usize) -> String {
    let (r_a, r_b, suited) = class_to_ranks(class);
    if r_a == r_b {
        format!("{}{}", RANKS[r_a as usize] as char, RANKS[r_b as usize] as char)
    } else if suited {
        format!(
            "{}{}s",
            RANKS[r_a as usize] as char,
            RANKS[r_b as usize] as char
        )
    } else {
        format!(
            "{}{}o",
            RANKS[r_a as usize] as char,
            RANKS[r_b as usize] as char
        )
    }
}

/// Compute range vs range equity with a given board (3, 4, or 5 cards).
/// Uses rayon to parallelize across hero hand classes.
pub fn range_vs_range_equity(
    range1: &[f64; NUM_CLASSES],
    range2: &[f64; NUM_CLASSES],
    board: &[Card],
) -> RangeEquityResult {
    let mut board_dead = CardSet::empty();
    for &c in board {
        board_dead.insert(c);
    }

    // Collect active hero classes
    let active_classes: Vec<usize> = (0..NUM_CLASSES)
        .filter(|&i| range1[i] > 0.0)
        .collect();

    // Parallel computation per hero class
    let per_class_results: Vec<(f64, f64, u64, usize, f64, f64)> = active_classes
        .par_iter()
        .flat_map(|&i| {
            let combos1 = expand_class_combos(i, range1[i], board_dead);
            combos1
                .into_iter()
                .flat_map(move |(hand1, w1)| {
                    let mut hand1_dead = board_dead;
                    hand1_dead.insert(hand1[0]);
                    hand1_dead.insert(hand1[1]);

                    let mut results = Vec::new();
                    for j in 0..NUM_CLASSES {
                        if range2[j] <= 0.0 {
                            continue;
                        }
                        let combos2 = expand_class_combos(j, range2[j], hand1_dead);
                        for (hand2, w2) in combos2 {
                            let eq = equity_exact(hand1, hand2, board);
                            let e = eq.equity();
                            let w = w1 * w2;
                            // (eq_weighted, weight, matchups, class_idx, class_eq, class_wt)
                            results.push((e * w, w, 1u64, i, e * w2, w2));
                        }
                    }
                    results
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Aggregate results
    let mut total_eq_weighted = 0.0f64;
    let mut total_weight = 0.0f64;
    let mut total_matchups = 0u64;
    let mut class_eq_sum = [0.0f64; NUM_CLASSES];
    let mut class_weight = [0.0f64; NUM_CLASSES];

    for &(ew, w, m, ci, ce, cw) in &per_class_results {
        total_eq_weighted += ew;
        total_weight += w;
        total_matchups += m;
        class_eq_sum[ci] += ce;
        class_weight[ci] += cw;
    }

    let equity = if total_weight > 0.0 {
        total_eq_weighted / total_weight
    } else {
        0.0
    };

    let mut class_equity = [0.0f64; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        if class_weight[i] > 0.0 {
            class_equity[i] = class_eq_sum[i] / class_weight[i];
        }
    }

    RangeEquityResult {
        equity,
        total_matchups,
        class_equity,
        class_weight,
    }
}

/// Compute range vs range equity using Monte Carlo (for preflop).
pub fn range_vs_range_monte_carlo(
    range1: &[f64; NUM_CLASSES],
    range2: &[f64; NUM_CLASSES],
    num_samples: usize,
    rng: &mut impl rand::Rng,
) -> RangeEquityResult {
    // Build weighted combo lists for both ranges
    let combos1 = build_all_combos(range1, CardSet::empty());
    let combos2_all = build_all_combos(range2, CardSet::empty());

    if combos1.is_empty() || combos2_all.is_empty() {
        return RangeEquityResult {
            equity: 0.0,
            total_matchups: 0,
            class_equity: [0.0; NUM_CLASSES],
            class_weight: [0.0; NUM_CLASSES],
        };
    }

    // Build cumulative weight distribution for range1
    let total_w1: f64 = combos1.iter().map(|c| c.2).sum();
    let mut cum_w1: Vec<f64> = Vec::with_capacity(combos1.len());
    let mut acc = 0.0;
    for c in &combos1 {
        acc += c.2;
        cum_w1.push(acc / total_w1);
    }

    let mut total_eq = 0.0f64;
    let mut total_count = 0u64;
    let mut class_eq_sum = [0.0f64; NUM_CLASSES];
    let mut class_count = [0.0f64; NUM_CLASSES];

    for _ in 0..num_samples {
        // Sample hand1 from range1 weighted by frequency
        let r1: f64 = rng.gen();
        let idx1 = match cum_w1.binary_search_by(|w| w.partial_cmp(&r1).unwrap()) {
            Ok(i) => i,
            Err(i) => i.min(combos1.len() - 1),
        };
        let (hand1, class1, _) = combos1[idx1];

        // Filter combos2 to remove those that conflict with hand1
        let mut hand1_dead = CardSet::empty();
        hand1_dead.insert(hand1[0]);
        hand1_dead.insert(hand1[1]);

        let valid2: Vec<&([Card; 2], usize, f64)> = combos2_all
            .iter()
            .filter(|(h, _, _)| !hand1_dead.contains(h[0]) && !hand1_dead.contains(h[1]))
            .collect();

        if valid2.is_empty() {
            continue;
        }

        let total_w2: f64 = valid2.iter().map(|c| c.2).sum();
        let r2: f64 = rng.gen::<f64>() * total_w2;
        let mut acc2 = 0.0;
        let mut idx2 = valid2.len() - 1;
        for (k, c) in valid2.iter().enumerate() {
            acc2 += c.2;
            if acc2 >= r2 {
                idx2 = k;
                break;
            }
        }
        let (hand2, _, _) = *valid2[idx2];

        // Deal 5 board cards
        let mut board_dead = hand1_dead;
        board_dead.insert(hand2[0]);
        board_dead.insert(hand2[1]);

        let available: Vec<Card> = (0..52u8)
            .map(Card)
            .filter(|c| !board_dead.contains(*c))
            .collect();

        if available.len() < 5 {
            continue;
        }

        // Shuffle and pick 5
        let mut avail_copy = available;
        avail_copy.shuffle(rng);
        let board = &avail_copy[..5];

        let cards1 = [
            hand1[0], hand1[1], board[0], board[1], board[2], board[3], board[4],
        ];
        let cards2 = [
            hand2[0], hand2[1], board[0], board[1], board[2], board[3], board[4],
        ];

        let s1 = evaluate_7(&cards1);
        let s2 = evaluate_7(&cards2);

        let eq = if s1 > s2 {
            1.0
        } else if s1 == s2 {
            0.5
        } else {
            0.0
        };

        total_eq += eq;
        total_count += 1;
        class_eq_sum[class1] += eq;
        class_count[class1] += 1.0;
    }

    let equity = if total_count > 0 {
        total_eq / total_count as f64
    } else {
        0.0
    };

    let mut class_equity = [0.0f64; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        if class_count[i] > 0.0 {
            class_equity[i] = class_eq_sum[i] / class_count[i];
        }
    }

    // Use class_count as weight
    let mut class_weight = [0.0f64; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        class_weight[i] = class_count[i];
    }

    RangeEquityResult {
        equity,
        total_matchups: total_count,
        class_equity,
        class_weight,
    }
}

/// Build all concrete combos from a range, each with (hand, class_index, weight).
fn build_all_combos(
    range: &[f64; NUM_CLASSES],
    dead: CardSet,
) -> Vec<([Card; 2], usize, f64)> {
    let mut all = Vec::new();
    for i in 0..NUM_CLASSES {
        if range[i] <= 0.0 {
            continue;
        }
        let combos = expand_class_combos(i, range[i], dead);
        for (hand, w) in combos {
            all.push((hand, i, w));
        }
    }
    all
}

/// Compute hero's equity against a villain range for a specific hero hand.
pub fn hand_vs_range_equity(
    hero: [Card; 2],
    villain_range: &[f64; NUM_CLASSES],
    board: &[Card],
) -> f64 {
    let mut dead = CardSet::empty();
    for &c in board {
        dead.insert(c);
    }
    dead.insert(hero[0]);
    dead.insert(hero[1]);

    let mut total_eq = 0.0f64;
    let mut total_weight = 0.0f64;

    for j in 0..NUM_CLASSES {
        if villain_range[j] <= 0.0 {
            continue;
        }
        let combos = expand_class_combos(j, villain_range[j], dead);
        for (villain_hand, w) in combos {
            let eq = equity_exact(hero, villain_hand, board);
            total_eq += eq.equity() * w;
            total_weight += w;
        }
    }

    if total_weight > 0.0 {
        total_eq / total_weight
    } else {
        0.0
    }
}

/// Count outs: cards in remaining deck that improve hero from losing/tying to winning.
pub fn count_outs(
    hero: [Card; 2],
    villain_range: &[f64; NUM_CLASSES],
    board: &[Card],
) -> u32 {
    if board.len() < 3 || board.len() > 4 {
        return 0;
    }

    let current_eq = hand_vs_range_equity(hero, villain_range, board);

    let mut dead = CardSet::empty();
    for &c in board {
        dead.insert(c);
    }
    dead.insert(hero[0]);
    dead.insert(hero[1]);

    let mut outs = 0u32;

    for card_id in 0..52u8 {
        let card = Card(card_id);
        if dead.contains(card) {
            continue;
        }

        let mut new_board: Vec<Card> = board.to_vec();
        new_board.push(card);
        let new_eq = hand_vs_range_equity(hero, villain_range, &new_board);

        // Count as an out if equity improves by at least 15 percentage points
        if new_eq > current_eq + 0.15 {
            outs += 1;
        }
    }

    outs
}

/// Analyze board texture.
pub fn board_texture(board: &[Card]) -> BoardTexture {
    if board.is_empty() {
        return BoardTexture {
            paired: false,
            two_tone: false,
            monotone: false,
            straight_possible: false,
            high_card: String::new(),
            hand_category: String::new(),
        };
    }

    // Check paired
    let mut rank_counts = [0u8; 13];
    for &c in board {
        rank_counts[c.rank() as usize] += 1;
    }
    let paired = rank_counts.iter().any(|&c| c >= 2);

    // Check suits
    let mut suit_counts = [0u8; 4];
    for &c in board {
        suit_counts[c.suit() as usize] += 1;
    }
    let max_suit = *suit_counts.iter().max().unwrap();
    let monotone = max_suit >= 3;
    let two_tone = max_suit >= 2 && !monotone;

    // Check straight possible (3+ cards within 4-rank window, or connected)
    let mut ranks_present: Vec<u8> = board.iter().map(|c| c.rank()).collect();
    ranks_present.sort();
    ranks_present.dedup();
    let straight_possible = if ranks_present.len() >= 3 {
        // Check if any 3 consecutive ranks exist within 4-span
        ranks_present
            .windows(3)
            .any(|w| w[2] - w[0] <= 4)
    } else {
        false
    };

    // High card
    let high_rank = board.iter().map(|c| c.rank()).max().unwrap();
    let high_card = format!("{}", RANKS[high_rank as usize] as char);

    // If we have 5+ cards, evaluate hand category
    let hand_category = if board.len() >= 5 {
        let s = crate::evaluator::evaluate_5(&[board[0], board[1], board[2], board[3], board[4]]);
        category_name(s).to_string()
    } else {
        String::new()
    };

    BoardTexture {
        paired,
        two_tone,
        monotone,
        straight_possible,
        high_card,
        hand_category,
    }
}

/// Calculate range statistics: (percentage of all hands, combo count).
pub fn range_stats(range: &[f64; NUM_CLASSES]) -> (f64, f64) {
    let combos = |class: usize| -> f64 {
        if class < 13 {
            6.0
        } else if class < 91 {
            4.0
        } else {
            12.0
        }
    };
    let total_combos: f64 = (0..NUM_CLASSES).map(|c| combos(c) * range[c]).sum();
    let total_hands: f64 = (0..NUM_CLASSES).map(combos).sum(); // 1326
    (total_combos / total_hands * 100.0, total_combos)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(s: &str) -> Card {
        let bytes = s.as_bytes();
        let rank = match bytes[0] {
            b'2' => 0, b'3' => 1, b'4' => 2, b'5' => 3, b'6' => 4,
            b'7' => 5, b'8' => 6, b'9' => 7, b'T' => 8, b'J' => 9,
            b'Q' => 10, b'K' => 11, b'A' => 12,
            _ => panic!("bad rank"),
        };
        let suit = match bytes[1] {
            b'c' => 0, b'd' => 1, b'h' => 2, b's' => 3,
            _ => panic!("bad suit"),
        };
        Card::new(rank, suit)
    }

    #[test]
    fn expand_pair_combos() {
        // AA (class 12) should have 6 combos with no dead cards
        let combos = expand_class_combos(12, 1.0, CardSet::empty());
        assert_eq!(combos.len(), 6);

        // With one dead ace, should have 3 combos
        let mut dead = CardSet::empty();
        dead.insert(c("As"));
        let combos = expand_class_combos(12, 1.0, dead);
        assert_eq!(combos.len(), 3);
    }

    #[test]
    fn expand_suited_combos() {
        // AKs: class index for suited with ranks (12, 11)
        // hand_class_index(11, 12) = 12*11/2 + 11 = 66+11 = 77, so class = 13+77 = 90
        let combos = expand_class_combos(90, 1.0, CardSet::empty());
        assert_eq!(combos.len(), 4);
        // All combos should be suited
        for (hand, _) in &combos {
            assert_eq!(hand[0].suit(), hand[1].suit());
        }
    }

    #[test]
    fn expand_offsuit_combos() {
        // AKo: class index for offsuit with ranks (12, 11)
        // 91 + 77 = 168
        let combos = expand_class_combos(168, 1.0, CardSet::empty());
        assert_eq!(combos.len(), 12);
        // All combos should be offsuit
        for (hand, _) in &combos {
            assert_ne!(hand[0].suit(), hand[1].suit());
        }
    }

    #[test]
    fn class_names() {
        assert_eq!(class_index_to_name(0), "22");
        assert_eq!(class_index_to_name(12), "AA");
        assert_eq!(class_index_to_name(90), "AKs");
        assert_eq!(class_index_to_name(168), "AKo");
    }

    #[test]
    fn range_stats_full() {
        let range = [1.0; NUM_CLASSES];
        let (pct, combos) = range_stats(&range);
        assert!((pct - 100.0).abs() < 0.01);
        assert!((combos - 1326.0).abs() < 0.01);
    }

    #[test]
    fn range_stats_empty() {
        let range = [0.0; NUM_CLASSES];
        let (pct, combos) = range_stats(&range);
        assert!((pct).abs() < 0.01);
        assert!((combos).abs() < 0.01);
    }

    #[test]
    fn hand_vs_range_river() {
        // AA vs 100% range on a given river board
        let hero = [c("As"), c("Ah")];
        let board = [c("2d"), c("5h"), c("9s"), c("Jc"), c("3d")];
        let range = [1.0; NUM_CLASSES];
        let eq = hand_vs_range_equity(hero, &range, &board);
        // AA should win the vast majority on this board
        assert!(eq > 0.8, "AA equity vs 100% range = {:.4}", eq);
    }

    #[test]
    fn range_vs_range_river_simple() {
        // Only AA (class 12) vs only KK (class 11) on a safe board
        let mut range1 = [0.0; NUM_CLASSES];
        let mut range2 = [0.0; NUM_CLASSES];
        range1[12] = 1.0; // AA
        range2[11] = 1.0; // KK
        let board = [c("2d"), c("5h"), c("9s"), c("Jc"), c("3c")];

        let result = range_vs_range_equity(&range1, &range2, &board);
        // AA beats KK on this board always
        assert!(
            result.equity > 0.95,
            "AA vs KK river equity = {:.4}",
            result.equity
        );
        assert!(result.total_matchups > 0);
    }

    #[test]
    fn board_texture_basic() {
        let board = [c("Ks"), c("Qs"), c("Js")];
        let tex = board_texture(&board);
        assert!(!tex.paired);
        assert!(tex.monotone);
        assert!(tex.straight_possible);
        assert_eq!(tex.high_card, "K");
    }

    #[test]
    fn board_texture_paired() {
        let board = [c("Ks"), c("Kh"), c("2d")];
        let tex = board_texture(&board);
        assert!(tex.paired);
        assert!(!tex.monotone);
    }

    #[test]
    fn monte_carlo_basic() {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // AA vs 100% range preflop - should be ~85%
        let mut range1 = [0.0; NUM_CLASSES];
        let range2 = [1.0; NUM_CLASSES];
        range1[12] = 1.0; // AA

        let result = range_vs_range_monte_carlo(&range1, &range2, 50_000, &mut rng);
        assert!(
            (result.equity - 0.85).abs() < 0.03,
            "AA vs 100% MC equity = {:.4}",
            result.equity
        );
    }
}
