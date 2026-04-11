use gto_cfr::Game;
use gto_core::Action;

/// Leduc Hold'em: a small poker game with 2 betting rounds.
///
/// - 6-card deck: J♠ J♦ Q♠ Q♦ K♠ K♦ (card = rank*2 + suit, rank: 0=J,1=Q,2=K)
/// - 2 players, each gets 1 hole card, 1 community card on the flop
/// - Each antes 1 chip
/// - Round 1 (preflop): fixed bet/raise of 2 chips
/// - Round 2 (flop): fixed bet/raise of 4 chips
/// - Max 2 bets per round (1 bet + 1 raise)
/// - Hand ranking: pair with board > high card; higher rank wins ties
pub struct LeducHoldem;

#[derive(Clone, Debug)]
pub struct LeducState {
    /// Hole cards for each player. card/2 = rank (0=J,1=Q,2=K). -1 = not dealt.
    pub cards: [i8; 2],
    /// Community card. -1 = not dealt.
    pub board: i8,
    /// Total chips invested by each player.
    pub pot: [u32; 2],
    /// Actions taken in preflop betting round.
    pub preflop_actions: Vec<Action>,
    /// Actions taken in flop betting round.
    pub flop_actions: Vec<Action>,
}

impl LeducState {
    fn new() -> Self {
        LeducState {
            cards: [-1, -1],
            board: -1,
            pot: [0; 2],
            preflop_actions: Vec::new(),
            flop_actions: Vec::new(),
        }
    }
}

/// Check if a betting round is complete.
fn is_round_complete(actions: &[Action]) -> bool {
    match actions.last() {
        None => false,
        Some(Action::Fold | Action::Call) => true,
        Some(Action::Check) => {
            actions.len() >= 2 && matches!(actions[actions.len() - 2], Action::Check)
        }
        _ => false,
    }
}

/// Check if the last action in a round is a fold.
fn has_fold(actions: &[Action]) -> bool {
    matches!(actions.last(), Some(Action::Fold))
}

/// Count bet + raise actions in a round.
fn num_bets(actions: &[Action]) -> u8 {
    actions
        .iter()
        .filter(|a| matches!(a, Action::Bet(_) | Action::Raise(_)))
        .count() as u8
}

fn rank_name(card: i8) -> &'static str {
    match card / 2 {
        0 => "J",
        1 => "Q",
        2 => "K",
        _ => unreachable!(),
    }
}

impl Game for LeducHoldem {
    type State = LeducState;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> LeducState {
        LeducState::new()
    }

    fn is_terminal(&self, state: &LeducState) -> bool {
        // Fold in either round
        if has_fold(&state.preflop_actions) || has_fold(&state.flop_actions) {
            return true;
        }
        // Showdown: board dealt and flop round complete
        state.board >= 0 && is_round_complete(&state.flop_actions)
    }

    fn is_chance_node(&self, state: &LeducState) -> bool {
        // Deal hole cards
        if state.cards[0] < 0 {
            return true;
        }
        // Deal board after preflop completes (no fold)
        state.board < 0
            && is_round_complete(&state.preflop_actions)
            && !has_fold(&state.preflop_actions)
    }

    fn chance_outcomes(&self, state: &LeducState) -> Vec<(LeducState, f64)> {
        if state.cards[0] < 0 {
            // Deal both hole cards: enumerate all ordered pairs from 6 cards
            let mut outcomes = Vec::new();
            for c0 in 0..6i8 {
                for c1 in 0..6i8 {
                    if c0 != c1 {
                        let mut s = state.clone();
                        s.cards = [c0, c1];
                        s.pot = [1, 1]; // antes
                        outcomes.push((s, 1.0 / 30.0)); // 6*5 = 30 ordered pairs
                    }
                }
            }
            outcomes
        } else {
            // Deal board card from remaining 4 cards
            let mut outcomes = Vec::new();
            let dealt = [state.cards[0], state.cards[1]];
            for c in 0..6i8 {
                if c != dealt[0] && c != dealt[1] {
                    let mut s = state.clone();
                    s.board = c;
                    outcomes.push((s, 1.0 / 4.0));
                }
            }
            outcomes
        }
    }

    fn current_player(&self, state: &LeducState) -> usize {
        if state.board >= 0 {
            // Flop betting: P0 acts at even indices
            state.flop_actions.len() % 2
        } else {
            // Preflop betting: P0 acts at even indices
            state.preflop_actions.len() % 2
        }
    }

    fn actions(&self, state: &LeducState) -> Vec<Action> {
        let (actions, bet_size) = if state.board >= 0 {
            (&state.flop_actions, 4u32)
        } else {
            (&state.preflop_actions, 2u32)
        };

        let bets = num_bets(actions);

        if bets == 0 {
            // No bet yet: check or bet
            vec![Action::Check, Action::Bet(bet_size)]
        } else if bets < 2 {
            // Facing a bet, can raise
            vec![Action::Fold, Action::Call, Action::Raise(bet_size)]
        } else {
            // Max bets reached: fold or call
            vec![Action::Fold, Action::Call]
        }
    }

    fn apply_action(&self, state: &LeducState, action: Action) -> LeducState {
        let mut s = state.clone();
        let player = self.current_player(state);

        match action {
            Action::Bet(amt) => {
                s.pot[player] += amt;
            }
            Action::Raise(amt) => {
                let to_call = s.pot[1 - player] - s.pot[player];
                s.pot[player] += to_call + amt;
            }
            Action::Call => {
                s.pot[player] = s.pot[1 - player];
            }
            Action::Check | Action::Fold => {}
            _ => unreachable!(),
        }

        if state.board >= 0 {
            s.flop_actions.push(action);
        } else {
            s.preflop_actions.push(action);
        }

        s
    }

    fn info_set_key(&self, state: &LeducState, player: usize) -> String {
        let mut key = rank_name(state.cards[player]).to_string();

        // Preflop actions
        if !state.preflop_actions.is_empty() {
            key.push('|');
            for a in &state.preflop_actions {
                key.push_str(&a.to_string());
            }
        }

        // Board card and flop actions
        if state.board >= 0 {
            key.push(':');
            key.push_str(rank_name(state.board));
            if !state.flop_actions.is_empty() {
                key.push('|');
                for a in &state.flop_actions {
                    key.push_str(&a.to_string());
                }
            }
        }

        key
    }

    fn payoff(&self, state: &LeducState, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));

        let opponent = 1 - player;

        // Check for fold
        if has_fold(&state.preflop_actions) {
            let folder = (state.preflop_actions.len() - 1) % 2;
            return if folder == player {
                -(state.pot[player] as f64)
            } else {
                state.pot[opponent] as f64
            };
        }
        if has_fold(&state.flop_actions) {
            let folder = (state.flop_actions.len() - 1) % 2;
            return if folder == player {
                -(state.pot[player] as f64)
            } else {
                state.pot[opponent] as f64
            };
        }

        // Showdown
        let our_rank = state.cards[player] / 2;
        let their_rank = state.cards[opponent] / 2;
        let board_rank = state.board / 2;

        let our_pair = our_rank == board_rank;
        let their_pair = their_rank == board_rank;

        if our_pair && !their_pair {
            state.pot[opponent] as f64
        } else if !our_pair && their_pair {
            -(state.pot[player] as f64)
        } else if our_rank > their_rank {
            state.pot[opponent] as f64
        } else if our_rank < their_rank {
            -(state.pot[player] as f64)
        } else {
            0.0 // tie (same rank, neither pairs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_is_chance() {
        let game = LeducHoldem;
        let state = game.initial_state();
        assert!(game.is_chance_node(&state));
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn deal_hole_cards() {
        let game = LeducHoldem;
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        assert_eq!(outcomes.len(), 30); // 6P2 = 30
        let prob_sum: f64 = outcomes.iter().map(|(_, p)| p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn preflop_check_check_deals_board() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 2]; // J♠ vs Q♠
        state.pot = [1, 1];

        // P0 checks
        state = game.apply_action(&state, Action::Check);
        assert!(!game.is_terminal(&state));
        assert!(!game.is_chance_node(&state));

        // P1 checks → preflop done, need to deal board
        state = game.apply_action(&state, Action::Check);
        assert!(!game.is_terminal(&state));
        assert!(game.is_chance_node(&state));

        // Deal board
        let outcomes = game.chance_outcomes(&state);
        assert_eq!(outcomes.len(), 4); // 6 - 2 dealt = 4 remaining
    }

    #[test]
    fn preflop_fold() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 4]; // J♠ vs K♠
        state.pot = [1, 1];

        // P0 bets, P1 folds
        state = game.apply_action(&state, Action::Bet(2));
        assert_eq!(state.pot, [3, 1]);

        state = game.apply_action(&state, Action::Fold);
        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), 1.0); // P0 wins P1's ante
        assert_eq!(game.payoff(&state, 1), -1.0);
    }

    #[test]
    fn full_hand_showdown() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 4]; // J♠ vs K♠
        state.pot = [1, 1];

        // Preflop: check-check
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        // Deal board: Q♠ (card index 2)
        state.board = 2;

        // Flop: check-check
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        assert!(game.is_terminal(&state));
        // K > J, K wins
        assert_eq!(game.payoff(&state, 0), -1.0); // J loses
        assert_eq!(game.payoff(&state, 1), 1.0); // K wins
    }

    #[test]
    fn pair_beats_high_card() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 4]; // J♠ vs K♠
        state.pot = [1, 1];

        // Preflop: bet-call
        state = game.apply_action(&state, Action::Bet(2));
        state = game.apply_action(&state, Action::Call);
        assert_eq!(state.pot, [3, 3]);

        // Board: J♦ (card index 1) → P0 pairs!
        state.board = 1;

        // Flop: bet-call
        state = game.apply_action(&state, Action::Bet(4));
        state = game.apply_action(&state, Action::Call);
        assert_eq!(state.pot, [7, 7]);

        assert!(game.is_terminal(&state));
        // J pairs with board, K doesn't
        assert_eq!(game.payoff(&state, 0), 7.0); // J wins
        assert_eq!(game.payoff(&state, 1), -7.0);
    }

    #[test]
    fn raise_and_call() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [2, 4]; // Q♠ vs K♠
        state.pot = [1, 1];

        // Preflop: P0 bets, P1 raises, P0 calls
        state = game.apply_action(&state, Action::Bet(2));
        assert_eq!(state.pot, [3, 1]);

        state = game.apply_action(&state, Action::Raise(2));
        assert_eq!(state.pot, [3, 5]);

        state = game.apply_action(&state, Action::Call);
        assert_eq!(state.pot, [5, 5]);

        // Preflop done, deal board
        assert!(game.is_chance_node(&state));
    }

    #[test]
    fn max_bets_reached() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 4];
        state.pot = [1, 1];

        // P0 bets (1 bet), P1 raises (2 bets = max)
        state = game.apply_action(&state, Action::Bet(2));
        state = game.apply_action(&state, Action::Raise(2));

        // P0 can only fold or call (no more raises)
        let actions = game.actions(&state);
        assert_eq!(actions, vec![Action::Fold, Action::Call]);
    }

    #[test]
    fn info_set_keys() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 4]; // J♠ vs K♠
        state.pot = [1, 1];

        // Preflop: P0's initial info set
        assert_eq!(game.info_set_key(&state, 0), "J");
        assert_eq!(game.info_set_key(&state, 1), "K");

        // P0 checks
        state = game.apply_action(&state, Action::Check);
        assert_eq!(game.info_set_key(&state, 0), "J|x");
        assert_eq!(game.info_set_key(&state, 1), "K|x");

        // P1 checks → preflop done
        state = game.apply_action(&state, Action::Check);

        // Deal board Q♦
        state.board = 3;

        // Flop info sets
        assert_eq!(game.info_set_key(&state, 0), "J|xx:Q");
        assert_eq!(game.info_set_key(&state, 1), "K|xx:Q");

        // P0 bets on flop
        state = game.apply_action(&state, Action::Bet(4));
        assert_eq!(game.info_set_key(&state, 0), "J|xx:Q|b4");
        assert_eq!(game.info_set_key(&state, 1), "K|xx:Q|b4");
    }

    #[test]
    fn tie_same_rank() {
        let game = LeducHoldem;
        let mut state = LeducState::new();
        state.cards = [0, 1]; // J♠ vs J♦
        state.pot = [1, 1];

        // Check-check preflop
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        // Board: K♠
        state.board = 4;

        // Check-check flop
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        assert!(game.is_terminal(&state));
        assert_eq!(game.payoff(&state, 0), 0.0); // tie
        assert_eq!(game.payoff(&state, 1), 0.0);
    }

    #[test]
    fn cfr_converges() {
        use gto_cfr::{train, TrainerConfig};

        let game = LeducHoldem;
        let config = TrainerConfig {
            iterations: 20_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &config);
        let exploit = solver.exploitability(&game);

        assert!(
            exploit < 0.05,
            "Leduc exploitability {:.6} should be < 0.05 after 20K iterations",
            exploit
        );
    }
}
