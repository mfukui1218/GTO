use gto_core::{Card, CardSet};
use gto_eval::evaluate_7;
use rand::seq::SliceRandom;

/// Hand bucketing: maps each 2-card combo to a bucket index based on hand strength.
#[derive(Clone, Debug)]
pub struct HandBucketing {
    pub num_buckets: usize,
    /// bucket_map[c1 * 52 + c2] = bucket index (255 = invalid).
    /// Only entries where c1 < c2 are valid.
    pub bucket_map: Vec<u8>,
    /// Average equity for each bucket.
    pub bucket_equity: Vec<f64>,
    /// Number of combos in each bucket (weight).
    pub bucket_weight: Vec<f64>,
}

impl HandBucketing {
    /// Look up the bucket for a hand (c1, c2). Cards can be in any order.
    pub fn get_bucket(&self, c1: Card, c2: Card) -> u8 {
        let (lo, hi) = if c1.0 < c2.0 { (c1.0, c2.0) } else { (c2.0, c1.0) };
        self.bucket_map[lo as usize * 52 + hi as usize]
    }
}

/// Compute hand bucketing for a given board and dead cards.
///
/// - **River (5 board cards)**: Uses `evaluate_7` directly for exact ranking.
/// - **Turn (4 board cards)**: Averages `evaluate_7` over all remaining river cards.
/// - **Flop (3 board cards)**: MC sampling to approximate equity.
pub fn compute_bucketing(
    board: &[Card],
    dead_cards: &CardSet,
    num_buckets: usize,
) -> HandBucketing {
    let num_buckets = num_buckets.min(255).max(1);

    // Build set of unavailable cards
    let mut used = *dead_cards;
    for &c in board {
        used.insert(c);
    }

    // Enumerate all valid 2-card combos
    let mut combos: Vec<(u8, u8, f64)> = Vec::new(); // (c1, c2, strength)

    match board.len() {
        5 => {
            // River: exact hand strength via evaluate_7
            for c1 in 0..52u8 {
                if used.contains(Card(c1)) { continue; }
                for c2 in (c1 + 1)..52u8 {
                    if used.contains(Card(c2)) { continue; }
                    let cards = [
                        Card(c1), Card(c2),
                        board[0], board[1], board[2], board[3], board[4],
                    ];
                    let strength = evaluate_7(&cards) as f64;
                    combos.push((c1, c2, strength));
                }
            }
        }
        4 => {
            // Turn: average strength over all possible river cards
            let remaining: Vec<u8> = (0..52u8)
                .filter(|&c| !used.contains(Card(c)))
                .collect();

            for c1 in 0..52u8 {
                if used.contains(Card(c1)) { continue; }
                for c2 in (c1 + 1)..52u8 {
                    if used.contains(Card(c2)) { continue; }
                    let mut total_strength = 0.0;
                    let mut count = 0;
                    for &river in &remaining {
                        if river == c1 || river == c2 { continue; }
                        let cards = [
                            Card(c1), Card(c2),
                            board[0], board[1], board[2], board[3], Card(river),
                        ];
                        total_strength += evaluate_7(&cards) as f64;
                        count += 1;
                    }
                    if count > 0 {
                        combos.push((c1, c2, total_strength / count as f64));
                    }
                }
            }
        }
        3 => {
            // Flop: MC sampling for approximate equity
            compute_flop_bucketing_mc(board, &used, &mut combos);
        }
        _ => {
            // Unsupported board size: just use uniform
            for c1 in 0..52u8 {
                if used.contains(Card(c1)) { continue; }
                for c2 in (c1 + 1)..52u8 {
                    if used.contains(Card(c2)) { continue; }
                    combos.push((c1, c2, 0.5));
                }
            }
        }
    }

    build_bucketing_from_combos(combos, num_buckets)
}

/// Precompute flop bucketing (convenience function).
pub fn precompute_flop_bucketing(flop: [Card; 3], num_buckets: usize) -> HandBucketing {
    compute_bucketing(&flop, &CardSet::empty(), num_buckets)
}

/// MC sampling for flop hand strength approximation.
fn compute_flop_bucketing_mc(
    board: &[Card],
    used: &CardSet,
    combos: &mut Vec<(u8, u8, f64)>,
) {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let remaining: Vec<u8> = (0..52u8)
        .filter(|&c| !used.contains(Card(c)))
        .collect();

    const MC_SAMPLES: usize = 100;

    for c1 in 0..52u8 {
        if used.contains(Card(c1)) { continue; }
        for c2 in (c1 + 1)..52u8 {
            if used.contains(Card(c2)) { continue; }
            // Sample random turn+river runouts and average hand strength
            let available: Vec<u8> = remaining.iter()
                .copied()
                .filter(|&c| c != c1 && c != c2)
                .collect();

            if available.len() < 2 { continue; }

            let mut total_strength = 0.0;
            let mut count = 0;
            let mut pool = available.clone();

            for _ in 0..MC_SAMPLES {
                pool.shuffle(&mut rng);
                let turn = pool[0];
                let river = pool[1];
                let cards = [
                    Card(c1), Card(c2),
                    board[0], board[1], board[2], Card(turn), Card(river),
                ];
                total_strength += evaluate_7(&cards) as f64;
                count += 1;
            }

            if count > 0 {
                combos.push((c1, c2, total_strength / count as f64));
            }
        }
    }
}

/// Given combos with strength, sort by strength and assign to equal-sized buckets.
fn build_bucketing_from_combos(
    mut combos: Vec<(u8, u8, f64)>,
    num_buckets: usize,
) -> HandBucketing {
    // Sort by strength ascending
    combos.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let n = combos.len();
    let bucket_size = if n > 0 { (n + num_buckets - 1) / num_buckets } else { 1 };

    let mut bucket_map = vec![255u8; 52 * 52];
    let mut bucket_strength_sum = vec![0.0f64; num_buckets];
    let mut bucket_count = vec![0usize; num_buckets];

    for (i, &(c1, c2, strength)) in combos.iter().enumerate() {
        let bucket = (i / bucket_size).min(num_buckets - 1);
        bucket_map[c1 as usize * 52 + c2 as usize] = bucket as u8;
        bucket_strength_sum[bucket] += strength;
        bucket_count[bucket] += 1;
    }

    let mut bucket_equity = vec![0.0; num_buckets];
    let mut bucket_weight = vec![0.0; num_buckets];
    let total_combos: usize = bucket_count.iter().sum();

    // Find max strength for normalization
    let max_strength = combos.last().map(|c| c.2).unwrap_or(1.0).max(1.0);

    for b in 0..num_buckets {
        if bucket_count[b] > 0 {
            bucket_equity[b] = bucket_strength_sum[b] / (bucket_count[b] as f64 * max_strength);
            bucket_weight[b] = bucket_count[b] as f64 / total_combos.max(1) as f64;
        }
    }

    HandBucketing {
        num_buckets,
        bucket_map,
        bucket_equity,
        bucket_weight,
    }
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
    fn river_bucketing_strong_hands_in_high_buckets() {
        let board = [c("Ts"), c("7h"), c("2c"), c("Jd"), c("3s")];
        let bucketing = compute_bucketing(&board, &CardSet::empty(), 10);

        // AA should be in a higher bucket than 54o
        let aa_bucket = bucketing.get_bucket(c("As"), c("Ah"));
        let low_bucket = bucketing.get_bucket(c("5d"), c("4h"));
        assert!(
            aa_bucket > low_bucket,
            "AA bucket {} should be > 54o bucket {}",
            aa_bucket, low_bucket
        );
    }

    #[test]
    fn bucketing_covers_all_combos() {
        let board = [c("Ts"), c("7h"), c("2c"), c("Jd"), c("3s")];
        let bucketing = compute_bucketing(&board, &CardSet::empty(), 10);

        let mut used = CardSet::empty();
        for &b in &board { used.insert(b); }

        let mut valid_count = 0;
        for c1 in 0..52u8 {
            if used.contains(Card(c1)) { continue; }
            for c2 in (c1 + 1)..52u8 {
                if used.contains(Card(c2)) { continue; }
                let b = bucketing.bucket_map[c1 as usize * 52 + c2 as usize];
                assert!(b < 10, "bucket should be < 10, got {}", b);
                valid_count += 1;
            }
        }
        // C(47, 2) = 1081
        assert_eq!(valid_count, 1081);
    }

    #[test]
    fn flop_bucketing_works() {
        let flop = [c("Ts"), c("7h"), c("2c")];
        let bucketing = precompute_flop_bucketing(flop, 8);

        assert_eq!(bucketing.num_buckets, 8);
        assert_eq!(bucketing.bucket_equity.len(), 8);

        // Equity should be monotonically non-decreasing
        for i in 1..8 {
            assert!(
                bucketing.bucket_equity[i] >= bucketing.bucket_equity[i - 1] - 0.01,
                "bucket equity not monotonic: {} < {}",
                bucketing.bucket_equity[i],
                bucketing.bucket_equity[i - 1]
            );
        }
    }

    #[test]
    fn turn_bucketing_works() {
        let board = [c("Ts"), c("7h"), c("2c"), c("Jd")];
        let bucketing = compute_bucketing(&board, &CardSet::empty(), 10);

        assert_eq!(bucketing.num_buckets, 10);
        // Weights should sum to ~1.0
        let total_weight: f64 = bucketing.bucket_weight.iter().sum();
        assert!(
            (total_weight - 1.0).abs() < 0.01,
            "weights sum to {}, expected ~1.0",
            total_weight
        );
    }

    #[test]
    fn dead_cards_excluded() {
        let board = [c("Ts"), c("7h"), c("2c"), c("Jd"), c("3s")];
        let mut dead = CardSet::empty();
        dead.insert(c("As"));
        dead.insert(c("Ah"));
        let bucketing = compute_bucketing(&board, &dead, 10);

        // AsAh bucket should be 255 (invalid) since As is dead
        let b = bucketing.bucket_map[c("As").0 as usize * 52 + c("Ah").0 as usize];
        assert_eq!(b, 255, "dead card combo should be 255");
    }
}
