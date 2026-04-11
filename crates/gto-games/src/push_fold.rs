use gto_cfr::Game;
use gto_core::{Action, Card};
use gto_eval::{evaluate_7, hand_class};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Number of canonical hand classes: 13 pairs + 78 suited + 78 offsuit = 169.
pub const NUM_CLASSES: usize = 169;

/// Precomputed data for push/fold analysis.
pub struct PushFoldData {
    /// equity_matrix[i][j] = equity of hand class i vs hand class j.
    pub equity: Vec<Vec<f64>>,
    /// weights[i][j] = probability of dealing class i to SB and class j to BB.
    pub weights: Vec<Vec<f64>>,
    /// Cumulative distribution function for chance sampling (length NUM_CLASSES*NUM_CLASSES).
    pub cumulative_weights: Vec<f64>,
}

impl PushFoldData {
    /// Build PushFoldData from equity and weights, computing the CDF.
    pub fn new(equity: Vec<Vec<f64>>, weights: Vec<Vec<f64>>) -> Self {
        let mut cumulative_weights = Vec::with_capacity(NUM_CLASSES * NUM_CLASSES);
        let mut cumsum = 0.0;
        for i in 0..NUM_CLASSES {
            for j in 0..NUM_CLASSES {
                cumsum += weights[i][j];
                cumulative_weights.push(cumsum);
            }
        }
        // Normalize to exactly 1.0
        if let Some(last) = cumulative_weights.last_mut() {
            *last = 1.0;
        }
        PushFoldData {
            equity,
            weights,
            cumulative_weights,
        }
    }

    /// Compute equity matrix and weights via Monte Carlo sampling.
    /// Uses rayon to parallelize across multiple threads.
    pub fn compute(num_samples: usize) -> Self {
        let num_chunks = rayon::current_num_threads().max(1);
        let chunk_size = num_samples / num_chunks;

        // Each thread computes partial results independently
        let partial_results: Vec<_> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let mut equity_sum = vec![vec![0.0f64; NUM_CLASSES]; NUM_CLASSES];
                let mut eq_counts = vec![vec![0u32; NUM_CLASSES]; NUM_CLASSES];
                let mut weight_counts = vec![vec![0u32; NUM_CLASSES]; NUM_CLASSES];

                // Each thread uses a different seed for independent sampling
                let mut rng = ChaCha8Rng::seed_from_u64(42 + chunk_idx as u64);
                let mut deck: Vec<Card> = (0..52).map(Card).collect();

                let samples = if chunk_idx == num_chunks - 1 {
                    num_samples - chunk_size * (num_chunks - 1)
                } else {
                    chunk_size
                };

                for _ in 0..samples {
                    deck.shuffle(&mut rng);

                    let hand1 = [deck[0], deck[1]];
                    let hand2 = [deck[2], deck[3]];

                    let c1 = hand_class(hand1[0], hand1[1]);
                    let c2 = hand_class(hand2[0], hand2[1]);

                    let cards1 = [
                        hand1[0], hand1[1], deck[4], deck[5], deck[6], deck[7], deck[8],
                    ];
                    let cards2 = [
                        hand2[0], hand2[1], deck[4], deck[5], deck[6], deck[7], deck[8],
                    ];

                    let s1 = evaluate_7(&cards1);
                    let s2 = evaluate_7(&cards2);

                    let eq1 = if s1 > s2 {
                        1.0
                    } else if s1 == s2 {
                        0.5
                    } else {
                        0.0
                    };

                    equity_sum[c1][c2] += eq1;
                    equity_sum[c2][c1] += 1.0 - eq1;
                    eq_counts[c1][c2] += 1;
                    eq_counts[c2][c1] += 1;
                    weight_counts[c1][c2] += 1;
                    weight_counts[c2][c1] += 1;
                }

                (equity_sum, eq_counts, weight_counts)
            })
            .collect();

        // Merge partial results
        let mut equity_sum = vec![vec![0.0f64; NUM_CLASSES]; NUM_CLASSES];
        let mut eq_counts = vec![vec![0u32; NUM_CLASSES]; NUM_CLASSES];
        let mut weight_counts = vec![vec![0u32; NUM_CLASSES]; NUM_CLASSES];

        for (p_eq, p_cnt, p_wt) in partial_results {
            for i in 0..NUM_CLASSES {
                for j in 0..NUM_CLASSES {
                    equity_sum[i][j] += p_eq[i][j];
                    eq_counts[i][j] += p_cnt[i][j];
                    weight_counts[i][j] += p_wt[i][j];
                }
            }
        }

        let mut equity = vec![vec![0.5; NUM_CLASSES]; NUM_CLASSES];
        let total_weight: f64 = weight_counts
            .iter()
            .flat_map(|r| r.iter())
            .map(|&c| c as f64)
            .sum();
        let mut weights = vec![vec![0.0; NUM_CLASSES]; NUM_CLASSES];

        for i in 0..NUM_CLASSES {
            for j in 0..NUM_CLASSES {
                if eq_counts[i][j] > 0 {
                    equity[i][j] = equity_sum[i][j] / eq_counts[i][j] as f64;
                }
                weights[i][j] = weight_counts[i][j] as f64 / total_weight;
            }
        }

        PushFoldData::new(equity, weights)
    }
}

/// Push/Fold game: SB can push all-in or fold, BB can call or fold.
pub struct PushFoldGame {
    /// Effective stack in big blinds.
    pub stack_bb: f64,
    /// Precomputed data.
    pub data: PushFoldData,
}

/// State of a push/fold hand.
#[derive(Clone, Debug)]
pub struct PushFoldState {
    /// SB's hand class index (0-168). -1 if not dealt.
    pub sb_class: i16,
    /// BB's hand class index (0-168). -1 if not dealt.
    pub bb_class: i16,
    /// Action history.
    pub history: Vec<Action>,
}

impl PushFoldGame {
    pub fn new(stack_bb: f64, data: PushFoldData) -> Self {
        PushFoldGame { stack_bb, data }
    }
}

impl Game for PushFoldGame {
    type State = PushFoldState;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> PushFoldState {
        PushFoldState {
            sb_class: -1,
            bb_class: -1,
            history: Vec::new(),
        }
    }

    fn is_terminal(&self, state: &PushFoldState) -> bool {
        match state.history.len() {
            1 => matches!(state.history[0], Action::Fold), // SB folded
            2 => true,                                     // BB acted after push
            _ => false,
        }
    }

    fn is_chance_node(&self, state: &PushFoldState) -> bool {
        state.sb_class < 0
    }

    fn chance_outcomes(&self, state: &PushFoldState) -> Vec<(PushFoldState, f64)> {
        let mut outcomes = Vec::with_capacity(NUM_CLASSES * NUM_CLASSES);
        for i in 0..NUM_CLASSES {
            for j in 0..NUM_CLASSES {
                let w = self.data.weights[i][j];
                if w > 0.0 {
                    outcomes.push((
                        PushFoldState {
                            sb_class: i as i16,
                            bb_class: j as i16,
                            history: state.history.clone(),
                        },
                        w,
                    ));
                }
            }
        }
        outcomes
    }

    fn sample_chance_outcome(
        &self,
        state: &PushFoldState,
        rng: &mut dyn rand::RngCore,
    ) -> (PushFoldState, f64) {
        use rand::Rng;
        let r: f64 = rng.gen();
        let idx = self.data.cumulative_weights
            .partition_point(|&cw| cw <= r)
            .min(NUM_CLASSES * NUM_CLASSES - 1);
        let i = idx / NUM_CLASSES;
        let j = idx % NUM_CLASSES;
        let prob = self.data.weights[i][j];
        (
            PushFoldState {
                sb_class: i as i16,
                bb_class: j as i16,
                history: state.history.clone(),
            },
            prob,
        )
    }

    fn current_player(&self, state: &PushFoldState) -> usize {
        state.history.len() // 0 = SB, 1 = BB
    }

    fn actions(&self, state: &PushFoldState) -> Vec<Action> {
        match state.history.len() {
            0 => vec![Action::Fold, Action::AllIn],     // SB: fold or push
            1 => vec![Action::Fold, Action::Call],       // BB: fold or call
            _ => unreachable!(),
        }
    }

    fn apply_action(&self, state: &PushFoldState, action: Action) -> PushFoldState {
        let mut s = state.clone();
        s.history.push(action);
        s
    }

    fn info_set_key(&self, state: &PushFoldState, player: usize) -> String {
        let class = if player == 0 {
            state.sb_class
        } else {
            state.bb_class
        };
        let name = class_index_to_name(class as usize);

        if player == 1 {
            // BB knows SB pushed
            format!("{}|a", name)
        } else {
            name
        }
    }

    fn payoff(&self, state: &PushFoldState, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));

        match state.history.as_slice() {
            [Action::Fold] => {
                // SB folded → SB loses 0.5bb, BB wins 0.5bb
                if player == 0 {
                    -0.5
                } else {
                    0.5
                }
            }
            [Action::AllIn, Action::Fold] => {
                // SB pushed, BB folded → SB wins 1bb, BB loses 1bb
                if player == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            [Action::AllIn, Action::Call] => {
                // Showdown: SB's equity determines payoff
                let eq = self.data.equity[state.sb_class as usize][state.bb_class as usize];
                let s = self.stack_bb;
                // SB's EV = equity × 2S - S = (2*eq - 1) * S
                let sb_ev = (2.0 * eq - 1.0) * s;
                if player == 0 {
                    sb_ev
                } else {
                    -sb_ev
                }
            }
            _ => unreachable!(),
        }
    }
}

/// Convert a hand class index (0-168) to a name like "AA", "AKs", "72o".
pub fn class_index_to_name(class: usize) -> String {
    const RANKS: &[u8] = b"23456789TJQKA";

    if class < 13 {
        // Pair
        let r = RANKS[class] as char;
        format!("{}{}", r, r)
    } else if class < 91 {
        // Suited
        let (low, high) = index_to_ranks(class - 13);
        format!("{}{}s", RANKS[high as usize] as char, RANKS[low as usize] as char)
    } else {
        // Offsuit
        let (low, high) = index_to_ranks(class - 91);
        format!("{}{}o", RANKS[high as usize] as char, RANKS[low as usize] as char)
    }
}

/// Convert a triangle index to (low_rank, high_rank).
fn index_to_ranks(index: usize) -> (u8, u8) {
    let mut high = 1u8;
    while ((high + 1) as usize) * (high as usize) / 2 <= index {
        high += 1;
    }
    let low = index - (high as usize) * (high as usize - 1) / 2;
    (low as u8, high)
}

/// Display push/fold results as a 13x13 chart.
pub fn display_chart(title: &str, frequencies: &[f64; NUM_CLASSES]) {
    const RANKS: &[u8] = b"23456789TJQKA";

    println!("{}", title);
    println!();

    // Count how many hands are in the range
    let combos = |class: usize| -> f64 {
        if class < 13 {
            6.0
        } else if class < 91 {
            4.0
        } else {
            12.0
        }
    };
    let total_combos: f64 = (0..NUM_CLASSES)
        .map(|c| combos(c) * frequencies[c])
        .sum();
    let total_hands: f64 = (0..NUM_CLASSES).map(combos).sum();
    println!(
        "Range: {:.1}% ({:.0}/{:.0} combos)",
        total_combos / total_hands * 100.0,
        total_combos,
        total_hands
    );
    println!();

    // Header
    print!("     ");
    for r in (0..13).rev() {
        print!(" {:>4}", RANKS[r] as char);
    }
    println!();

    // Grid: rows = high card rank (desc), cols = low card rank (desc)
    // Above diagonal = suited, below = offsuit, diagonal = pairs
    for row in (0..13).rev() {
        print!("  {}  ", RANKS[row] as char);
        for col in (0..13).rev() {
            let class = if row == col {
                row // pair
            } else if col > row {
                // col > row: suited (col is high, row is low)
                13 + (col * (col - 1) / 2 + row)
            } else {
                // row > col: offsuit (row is high, col is low)
                91 + (row * (row - 1) / 2 + col)
            };
            let freq = frequencies[class];
            if freq > 0.995 {
                print!("  100");
            } else if freq < 0.005 {
                print!("    .");
            } else {
                print!(" {:>4.0}", freq * 100.0);
            }
        }
        println!();
    }
    println!();
    println!("  (rows = first rank, cols = second rank)");
    println!("  (above diagonal = suited, below = offsuit)");
}

/// Extract push frequencies from a solved strategy.
pub fn extract_push_range(strategy: &gto_cfr::Strategy) -> [f64; NUM_CLASSES] {
    let mut freqs = [0.0; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        let name = class_index_to_name(i);
        if let Some(probs) = strategy.get(&name) {
            // Actions: [Fold, AllIn]
            freqs[i] = probs[1] as f64; // AllIn frequency
        }
    }
    freqs
}

/// Extract call frequencies from a solved strategy.
pub fn extract_call_range(strategy: &gto_cfr::Strategy) -> [f64; NUM_CLASSES] {
    let mut freqs = [0.0; NUM_CLASSES];
    for i in 0..NUM_CLASSES {
        let name = format!("{}|a", class_index_to_name(i));
        if let Some(probs) = strategy.get(&name) {
            // Actions: [Fold, Call]
            freqs[i] = probs[1] as f64; // Call frequency
        }
    }
    freqs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_name_roundtrip() {
        for i in 0..NUM_CLASSES {
            let name = class_index_to_name(i);
            assert!(
                name.len() >= 2 && name.len() <= 3,
                "class {} = '{}' has bad length",
                i,
                name
            );
        }
        // Spot checks
        assert_eq!(class_index_to_name(0), "22");
        assert_eq!(class_index_to_name(12), "AA");
    }

    #[test]
    fn class_index_consistency() {
        // Verify that hand_class and class_index_to_name are consistent
        // for canonical hands
        for rank in 0..13u8 {
            let c1 = Card::new(rank, 0);
            let c2 = Card::new(rank, 1);
            let class = hand_class(c1, c2);
            assert_eq!(class, rank as usize, "Pair rank {} mismatch", rank);
        }
    }

    #[test]
    fn push_fold_basic_game_flow() {
        // Tiny equity matrix for testing (just use 0.5 everywhere)
        let data = PushFoldData::new(
            vec![vec![0.5; NUM_CLASSES]; NUM_CLASSES],
            {
                let w = 1.0 / (NUM_CLASSES * NUM_CLASSES) as f64;
                vec![vec![w; NUM_CLASSES]; NUM_CLASSES]
            },
        );
        let game = PushFoldGame::new(10.0, data);

        let state = game.initial_state();
        assert!(game.is_chance_node(&state));

        // Deal
        let outcomes = game.chance_outcomes(&state);
        assert_eq!(outcomes.len(), NUM_CLASSES * NUM_CLASSES);

        // Pick a state after dealing
        let dealt = &outcomes[0].0;
        assert!(!game.is_chance_node(dealt));
        assert!(!game.is_terminal(dealt));
        assert_eq!(game.current_player(dealt), 0); // SB acts

        // SB folds
        let folded = game.apply_action(dealt, Action::Fold);
        assert!(game.is_terminal(&folded));
        assert_eq!(game.payoff(&folded, 0), -0.5);
        assert_eq!(game.payoff(&folded, 1), 0.5);

        // SB pushes
        let pushed = game.apply_action(dealt, Action::AllIn);
        assert!(!game.is_terminal(&pushed));
        assert_eq!(game.current_player(&pushed), 1); // BB acts

        // BB folds
        let bb_fold = game.apply_action(&pushed, Action::Fold);
        assert!(game.is_terminal(&bb_fold));
        assert_eq!(game.payoff(&bb_fold, 0), 1.0);

        // BB calls with 50% equity → EV = 0
        let bb_call = game.apply_action(&pushed, Action::Call);
        assert!(game.is_terminal(&bb_call));
        assert!((game.payoff(&bb_call, 0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn push_fold_cfr_converges() {
        use gto_cfr::{train, TrainerConfig};

        // Small MC samples for fast test
        let data = PushFoldData::compute(500_000);
        let game = PushFoldGame::new(10.0, data);

        let config = TrainerConfig {
            iterations: 5_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &config);
        let exploit = solver.exploitability(&game);

        assert!(
            exploit < 0.1,
            "Push/fold exploitability {:.4} should be < 0.1",
            exploit
        );

        // AA should always push
        let strategy = gto_cfr::Strategy::from_solver(&solver);
        let push_range = extract_push_range(&strategy);
        assert!(
            push_range[12] > 0.95,
            "AA push freq = {:.4}, should be > 0.95",
            push_range[12]
        );
    }

    #[test]
    fn push_fold_chance_sampling_converges() {
        use gto_cfr::{train, TrainerConfig};

        let data = PushFoldData::compute(500_000);
        let game = PushFoldGame::new(10.0, data);

        let config = TrainerConfig {
            iterations: 200_000,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &config);
        let exploit = solver.exploitability(&game);

        assert!(
            exploit < 0.1,
            "Push/fold CS-MCCFR exploitability {:.4} should be < 0.1",
            exploit
        );

        let strategy = gto_cfr::Strategy::from_solver(&solver);
        let push_range = extract_push_range(&strategy);
        assert!(
            push_range[12] > 0.95,
            "AA push freq = {:.4}, should be > 0.95",
            push_range[12]
        );
    }

    #[test]
    fn cdf_sampling_correctness() {
        let data = PushFoldData::new(
            vec![vec![0.5; NUM_CLASSES]; NUM_CLASSES],
            {
                let w = 1.0 / (NUM_CLASSES * NUM_CLASSES) as f64;
                vec![vec![w; NUM_CLASSES]; NUM_CLASSES]
            },
        );

        // CDF should be monotonically non-decreasing
        for i in 1..data.cumulative_weights.len() {
            assert!(
                data.cumulative_weights[i] >= data.cumulative_weights[i - 1],
                "CDF not monotonic at index {}: {} < {}",
                i,
                data.cumulative_weights[i],
                data.cumulative_weights[i - 1]
            );
        }

        // Last element should be 1.0
        assert!(
            (data.cumulative_weights.last().unwrap() - 1.0).abs() < 1e-10,
            "CDF last element = {}, should be 1.0",
            data.cumulative_weights.last().unwrap()
        );

        // Length should be NUM_CLASSES * NUM_CLASSES
        assert_eq!(
            data.cumulative_weights.len(),
            NUM_CLASSES * NUM_CLASSES,
            "CDF length mismatch"
        );
    }
}
