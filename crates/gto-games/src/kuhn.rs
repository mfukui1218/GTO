use gto_cfr::Game;
use gto_core::Action;

/// Kuhn Poker: the simplest non-trivial poker game.
///
/// - 3 cards: J(0), Q(1), K(2)
/// - 2 players, each gets 1 card
/// - Each antes 1 chip
/// - Player 0 acts first: check or bet(1)
/// - If check: Player 1 can check or bet(1)
///   - If P1 checks: showdown
///   - If P1 bets: P0 can call or fold
/// - If bet: Player 1 can call or fold
///
/// Known Nash equilibrium game value for P0: -1/18 ≈ -0.0556
pub struct KuhnPoker;

/// State of a Kuhn Poker hand.
#[derive(Clone, Debug)]
pub struct KuhnState {
    /// Cards dealt: cards[0] = player 0's card, cards[1] = player 1's card
    /// -1 means not yet dealt (chance node)
    pub cards: [i8; 2],
    /// Action history (sequence of actions taken by players)
    pub history: Vec<Action>,
    /// Pot contributions per player (starts at 1 each for ante)
    pub pot: [u32; 2],
}

impl KuhnState {
    fn new() -> Self {
        KuhnState {
            cards: [-1, -1],
            history: Vec::new(),
            pot: [0; 2],
        }
    }

    /// Get a short string for the card: J, Q, K.
    fn card_str(card: i8) -> &'static str {
        match card {
            0 => "J",
            1 => "Q",
            2 => "K",
            _ => unreachable!(),
        }
    }
}

impl Game for KuhnPoker {
    type State = KuhnState;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> KuhnState {
        KuhnState::new()
    }

    fn is_terminal(&self, state: &KuhnState) -> bool {
        let h = &state.history;
        match h.len() {
            // After P0 bets, P1 folds or calls → terminal
            2 if matches!(h[0], Action::Bet(_)) => true,
            // After P0 checks, P1 checks → showdown (terminal)
            2 if matches!(h[0], Action::Check) && matches!(h[1], Action::Check) => true,
            // After P0 checks, P1 bets, P0 folds or calls → terminal
            3 => true,
            _ => false,
        }
    }

    fn is_chance_node(&self, state: &KuhnState) -> bool {
        // Chance if either card hasn't been dealt
        state.cards[0] < 0 || state.cards[1] < 0
    }

    fn chance_outcomes(&self, state: &KuhnState) -> Vec<(KuhnState, f64)> {
        let mut outcomes = Vec::new();

        if state.cards[0] < 0 {
            // Deal both cards at once: enumerate all ordered pairs from {0,1,2}
            for c0 in 0..3i8 {
                for c1 in 0..3i8 {
                    if c0 != c1 {
                        let mut new_state = state.clone();
                        new_state.cards[0] = c0;
                        new_state.cards[1] = c1;
                        new_state.pot = [1, 1]; // antes
                        // 3 choose 2 × 2! = 6 equally likely deals
                        outcomes.push((new_state, 1.0 / 6.0));
                    }
                }
            }
        }

        outcomes
    }

    fn current_player(&self, state: &KuhnState) -> usize {
        match state.history.len() {
            0 => 0, // P0 acts first
            1 => 1, // P1 responds
            2 => 0, // P0 responds to P1's bet (after check-bet)
            _ => unreachable!("should be terminal"),
        }
    }

    fn actions(&self, state: &KuhnState) -> Vec<Action> {
        let h = &state.history;
        match h.len() {
            0 => vec![Action::Check, Action::Bet(1)],      // P0: check or bet
            1 => {
                if matches!(h[0], Action::Check) {
                    vec![Action::Check, Action::Bet(1)]     // P1 after check: check or bet
                } else {
                    vec![Action::Fold, Action::Call]         // P1 after bet: fold or call
                }
            }
            2 => {
                // P0 after check-bet: fold or call
                vec![Action::Fold, Action::Call]
            }
            _ => unreachable!(),
        }
    }

    fn apply_action(&self, state: &KuhnState, action: Action) -> KuhnState {
        let mut new_state = state.clone();
        let player = self.current_player(state);

        match action {
            Action::Bet(amt) => {
                new_state.pot[player] += amt;
            }
            Action::Call => {
                // Match the opponent's bet
                let opponent = 1 - player;
                new_state.pot[player] = new_state.pot[opponent];
            }
            Action::Fold | Action::Check => {}
            _ => unreachable!("invalid action for Kuhn Poker"),
        }

        new_state.history.push(action);
        new_state
    }

    fn info_set_key(&self, state: &KuhnState, player: usize) -> String {
        // Format: "card|action_history"
        // e.g., "Q|xb" = holding Queen, P0 checked then P1 bet
        let mut key = String::new();
        key.push_str(KuhnState::card_str(state.cards[player]));
        if !state.history.is_empty() {
            key.push('|');
            for action in &state.history {
                key.push_str(&format!("{}", action));
            }
        }
        key
    }

    fn payoff(&self, state: &KuhnState, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));

        let h = &state.history;
        let opponent = 1 - player;

        // Check if someone folded
        if let Some(last) = h.last() {
            if matches!(last, Action::Fold) {
                // The player who folded is the current_player at the time of fold
                let folder = match h.len() {
                    2 => 1, // P1 folded (after P0 bet, P1 fold)
                    3 => 0, // P0 folded (after P0 check, P1 bet, P0 fold)
                    _ => unreachable!(),
                };
                if folder == player {
                    // We folded: we lose our pot contribution
                    return -(state.pot[player] as f64);
                } else {
                    // Opponent folded: we win their pot contribution
                    return state.pot[opponent] as f64;
                }
            }
        }

        // Showdown: higher card wins
        let our_card = state.cards[player];
        let their_card = state.cards[opponent];

        if our_card > their_card {
            state.pot[opponent] as f64
        } else {
            -(state.pot[player] as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_chance() {
        let game = KuhnPoker;
        let state = game.initial_state();
        assert!(game.is_chance_node(&state));
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn chance_outcomes_count() {
        let game = KuhnPoker;
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        assert_eq!(outcomes.len(), 6); // 3P2 = 6
        let prob_sum: f64 = outcomes.iter().map(|(_, p)| p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn check_check_showdown() {
        let game = KuhnPoker;
        // K vs J, check-check: K wins 1
        let mut state = KuhnState::new();
        state.cards = [2, 0]; // K vs J
        state.pot = [1, 1];
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), 1.0);  // K wins J's ante
        assert_eq!(game.payoff(&state, 1), -1.0); // J loses ante
    }

    #[test]
    fn bet_fold() {
        let game = KuhnPoker;
        // J vs Q, P0 bets, P1 folds
        let mut state = KuhnState::new();
        state.cards = [0, 1]; // J vs Q
        state.pot = [1, 1];
        state = game.apply_action(&state, Action::Bet(1));
        state = game.apply_action(&state, Action::Fold);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), 1.0);  // P0 wins P1's ante
        assert_eq!(game.payoff(&state, 1), -1.0); // P1 loses ante
    }

    #[test]
    fn bet_call_showdown() {
        let game = KuhnPoker;
        // J vs K, P0 bets, P1 calls
        let mut state = KuhnState::new();
        state.cards = [0, 2]; // J vs K
        state.pot = [1, 1];
        state = game.apply_action(&state, Action::Bet(1));
        state = game.apply_action(&state, Action::Call);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), -2.0); // J loses bet+ante
        assert_eq!(game.payoff(&state, 1), 2.0);  // K wins
    }

    #[test]
    fn check_bet_call() {
        let game = KuhnPoker;
        // Q vs K, check-bet-call
        let mut state = KuhnState::new();
        state.cards = [1, 2]; // Q vs K
        state.pot = [1, 1];
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Bet(1));
        state = game.apply_action(&state, Action::Call);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), -2.0); // Q loses
        assert_eq!(game.payoff(&state, 1), 2.0);  // K wins
    }

    #[test]
    fn check_bet_fold() {
        let game = KuhnPoker;
        // K vs J, check-bet-fold
        let mut state = KuhnState::new();
        state.cards = [2, 0]; // K vs J
        state.pot = [1, 1];
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Bet(1));
        state = game.apply_action(&state, Action::Fold);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), -1.0); // K folded, loses ante
        assert_eq!(game.payoff(&state, 1), 1.0);  // J wins ante
    }

    #[test]
    fn info_set_keys() {
        let game = KuhnPoker;
        let mut state = KuhnState::new();
        state.cards = [1, 2]; // Q vs K
        state.pot = [1, 1];

        assert_eq!(game.info_set_key(&state, 0), "Q");
        assert_eq!(game.info_set_key(&state, 1), "K");

        let state2 = game.apply_action(&state, Action::Check);
        assert_eq!(game.info_set_key(&state2, 0), "Q|x");
        assert_eq!(game.info_set_key(&state2, 1), "K|x");
    }

    #[test]
    fn cfr_converges_to_nash_equilibrium() {
        use gto_cfr::{train, Strategy, TrainerConfig};

        let game = KuhnPoker;
        let config = TrainerConfig {
            iterations: 50_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &config);
        let strategy = Strategy::from_solver(&solver);

        // Game value should be close to -1/18
        let exploit = solver.exploitability(&game);
        assert!(
            exploit < 0.005,
            "Exploitability {:.6} should be < 0.005",
            exploit
        );

        // Check Player 1 strategies (these have unique Nash eq values)
        let eps = 0.05;

        // P1 after check: J bets 1/3, Q checks, K always bets
        let j_x = strategy.get("J|x").unwrap();
        assert!((j_x[1] - 1.0 / 3.0).abs() < eps, "J|x bet={:.4}", j_x[1]);
        let q_x = strategy.get("Q|x").unwrap();
        assert!(q_x[1] < eps, "Q|x bet={:.4}", q_x[1]);
        let k_x = strategy.get("K|x").unwrap();
        assert!((k_x[1] - 1.0).abs() < eps, "K|x bet={:.4}", k_x[1]);

        // P1 facing bet: J folds, Q calls 1/3, K always calls
        let j_b = strategy.get("J|b1").unwrap();
        assert!(j_b[1] < eps, "J|b1 call={:.4}", j_b[1]);
        let q_b = strategy.get("Q|b1").unwrap();
        assert!(
            (q_b[1] - 1.0 / 3.0).abs() < eps,
            "Q|b1 call={:.4}",
            q_b[1]
        );
        let k_b = strategy.get("K|b1").unwrap();
        assert!((k_b[1] - 1.0).abs() < eps, "K|b1 call={:.4}", k_b[1]);

        // P0: Q should check 100%
        let q = strategy.get("Q").unwrap();
        assert!(q[1] < eps, "Q bet={:.4}", q[1]);

        // P0: J bluff frequency α ∈ [0, 1/3], K bets 3α
        let j = strategy.get("J").unwrap();
        let alpha = j[1]; // J's bet frequency
        assert!(alpha < 1.0 / 3.0 + eps, "J bet α={:.4} > 1/3", alpha);

        let k = strategy.get("K").unwrap();
        let k_bet = k[1];
        // K should bet ≈ 3α (with some tolerance)
        assert!(
            (k_bet - 3.0 * alpha).abs() < 0.1,
            "K bet={:.4}, 3α={:.4}",
            k_bet,
            3.0 * alpha
        );

        // P0 facing bet: J folds, K calls
        let j_xb = strategy.get("J|xb1").unwrap();
        assert!(j_xb[1] < eps, "J|xb1 call={:.4}", j_xb[1]);
        let k_xb = strategy.get("K|xb1").unwrap();
        assert!((k_xb[1] - 1.0).abs() < eps, "K|xb1 call={:.4}", k_xb[1]);
    }

    #[test]
    fn kuhn_chance_sampling_converges() {
        use gto_cfr::{train, TrainerConfig};

        let game = KuhnPoker;
        let config = TrainerConfig {
            iterations: 100_000,
            use_cfr_plus: false,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &config);
        let exploit = solver.exploitability(&game);

        assert!(
            exploit < 0.01,
            "Kuhn CS-MCCFR exploitability {:.6} should be < 0.01",
            exploit
        );
    }

    #[test]
    fn node_locking_fixes_strategy_and_best_responds() {
        use gto_cfr::{train, CfrSolver, Strategy, TrainerConfig};

        let game = KuhnPoker;

        // Lock P1's "J|x" (J after check) to ALWAYS bet.
        // In unlocked Nash, this is 1/3 bet (bluff). If P1 is forced to bet J always,
        // P0 with Q facing a bet should adapt toward calling more (since opponent's
        // bluff frequency is higher), and the locked node must report exactly [0, 1].
        //
        // Use the train path without locks first to get a baseline Q|xb1 call freq,
        // then train a locked solver and compare.
        let baseline_tc = TrainerConfig {
            iterations: 20_000,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
            ..Default::default()
        };
        let baseline_solver = train(&game, &baseline_tc);
        let baseline_strat = Strategy::from_solver(&baseline_solver);
        let baseline_q_call = baseline_strat.get("Q|xb1").unwrap()[1];

        // Now train with "J|x" locked to always bet.
        let mut solver = CfrSolver::new();
        solver.lock_node("J|x", vec![0.0, 1.0]);
        assert!(solver.is_locked("J|x"));

        for _ in 0..20_000 {
            solver.iterate_plus(&game);
        }

        let locked_strat = Strategy::from_solver(&solver);

        // Locked node must report exactly the locked distribution.
        let j_x_after = locked_strat.get("J|x").unwrap();
        assert!((j_x_after[0] - 0.0).abs() < 1e-6, "J|x[0] = {}", j_x_after[0]);
        assert!((j_x_after[1] - 1.0).abs() < 1e-6, "J|x[1] = {}", j_x_after[1]);

        // Q|xb1 call frequency should adapt upward (P1 now bluffs J 100% instead of 33%).
        // In this modified game P0-Q facing a bet should always call (EV: win 2 vs J, lose 2 vs K = 0 > -1 fold).
        let locked_q_call = locked_strat.get("Q|xb1").unwrap()[1];
        assert!(
            locked_q_call > baseline_q_call + 0.1,
            "Locked Q|xb1 call {:.3} should exceed baseline {:.3} by at least 0.1",
            locked_q_call,
            baseline_q_call
        );
        assert!(
            locked_q_call > 0.8,
            "Locked Q|xb1 call {:.3} should approach 1.0",
            locked_q_call
        );

        // Unlock and confirm the flag clears.
        solver.unlock_node("J|x");
        assert!(!solver.is_locked("J|x"));
    }

    #[test]
    fn target_exploitability_auto_stops() {
        // With target_exploitability set, training should halt as soon as the
        // measured exploitability drops below the target — strictly before
        // consuming the full iteration budget.
        use gto_cfr::{train_with_callback, TrainerConfig};

        let game = KuhnPoker;
        let tc = TrainerConfig {
            iterations: 200_000,             // large budget — we expect NOT to use it all
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
            target_exploitability: Some(0.01),
            exploitability_check_interval: 200,
            ..Default::default()
        };

        // Track the last iteration reported by the progress callback.
        let mut last_iter = 0usize;
        let solver = train_with_callback(&game, &tc, |iter, _total| {
            last_iter = iter;
        });

        let final_exploit = solver.exploitability(&game);
        assert!(
            final_exploit <= 0.01 + 1e-9,
            "final exploitability {:.6} should be <= target 0.01",
            final_exploit
        );
        assert!(
            last_iter < tc.iterations,
            "should have stopped early, but ran full {}/{} iterations",
            last_iter,
            tc.iterations,
        );
        // And training should have actually made progress (not stopped at iter 0).
        assert!(last_iter >= tc.exploitability_check_interval);
    }

    #[test]
    fn target_exploitability_none_runs_full_iterations() {
        // Baseline: unset target => full iteration count used.
        use gto_cfr::{train_with_callback, TrainerConfig};

        let game = KuhnPoker;
        let tc = TrainerConfig {
            iterations: 500,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
            target_exploitability: None,
            ..Default::default()
        };

        let mut last_iter = 0usize;
        let _ = train_with_callback(&game, &tc, |iter, _total| {
            last_iter = iter;
        });
        assert_eq!(
            last_iter, tc.iterations,
            "with no target, training should run all {} iterations",
            tc.iterations,
        );
    }

    #[test]
    fn linear_cfr_converges_faster_than_vanilla() {
        use gto_cfr::{train, TrainerConfig};

        let game = KuhnPoker;

        // Vanilla CFR, 2000 iterations
        let vanilla_tc = TrainerConfig {
            iterations: 2_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 0,
            ..Default::default()
        };
        let vanilla_solver = train(&game, &vanilla_tc);
        let vanilla_exploit = vanilla_solver.exploitability(&game);

        // Discount CFR (PioSolver preset), same iterations
        let linear_tc = TrainerConfig {
            iterations: 2_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 0,
            use_linear_cfr: true,
            ..Default::default()
        };
        let linear_solver = train(&game, &linear_tc);
        let linear_exploit = linear_solver.exploitability(&game);

        // Linear CFR should converge strictly better than vanilla at the same iteration budget.
        assert!(
            linear_exploit < vanilla_exploit,
            "Linear CFR exploit {:.6} should be < vanilla {:.6}",
            linear_exploit,
            vanilla_exploit
        );
        // And the absolute level should already be tight on Kuhn.
        assert!(
            linear_exploit < 0.01,
            "Linear CFR exploit {:.6} should be < 0.01 after 2k iters",
            linear_exploit
        );
    }
}
