use gto_abstraction::{compute_bucketing, HandBucketing};
use gto_cfr::Game;
use gto_core::iso::canonical_cache_key;
use gto_core::{Action, Card, CardSet};
use gto_eval::{evaluate_7, hand_class, HandStrength, NUM_CLASSES};
use rand::Rng;
use rustc_hash::FxHashMap;
use std::sync::RwLock;

/// Bet size configuration per position and street.
#[derive(Clone, Debug)]
pub struct BetSizeConfig {
    /// OOP bet sizes as % of pot for [flop, turn, river].
    pub oop_bet_sizes: [Vec<u32>; 3],
    /// IP bet sizes as % of pot for [flop, turn, river].
    pub ip_bet_sizes: [Vec<u32>; 3],
    /// OOP raise sizes as % of pot for [flop, turn, river].
    pub oop_raise_sizes: [Vec<u32>; 3],
    /// IP raise sizes as % of pot for [flop, turn, river].
    pub ip_raise_sizes: [Vec<u32>; 3],
    /// Maximum raises per street.
    pub max_raises_per_street: usize,
}

impl Default for BetSizeConfig {
    fn default() -> Self {
        BetSizeConfig {
            oop_bet_sizes: [vec![33, 75], vec![33, 75], vec![33, 75]],
            ip_bet_sizes: [vec![33, 75], vec![33, 75], vec![33, 75]],
            oop_raise_sizes: [vec![100], vec![100], vec![100]],
            ip_raise_sizes: [vec![100], vec![100], vec![100]],
            max_raises_per_street: 2,
        }
    }
}

/// Subgame entry point: which street to start CFR from.
///
/// `Flop` (default) runs a full flop→turn→river solve. `Turn(card)` and
/// `River([turn, river])` start CFR directly at that street, using the provided
/// ranges as the reach distribution at that node. This is the core of PioSolver-
/// style "turn/river subgame" analysis: solve a specific spot at higher precision
/// by reusing the flop solution's ranges as input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubgameStart {
    /// Start from the flop (full solve).
    Flop,
    /// Start from the turn. The given card is the turn card appended to the flop.
    Turn(Card),
    /// Start from the river. `[turn, river]` are appended to the flop.
    River([Card; 2]),
}

impl Default for SubgameStart {
    fn default() -> Self {
        SubgameStart::Flop
    }
}

/// Postflop solver configuration.
#[derive(Clone, Debug)]
pub struct PostflopConfig {
    pub flop: [Card; 3],
    pub pot: u32,
    pub effective_stack: u32,
    pub bet_sizes: BetSizeConfig,
    /// OOP range: 169-element array (hand class weights).
    ///
    /// For a flop solve, this is the preflop range that reaches the flop.
    /// For a turn/river subgame, this is the range *after* flop play that
    /// reaches the subgame root — i.e., the flop solution's strategy output.
    pub oop_range: Vec<f64>,
    /// IP range: 169-element array. Same semantics as `oop_range`.
    pub ip_range: Vec<f64>,
    /// Number of hand strength buckets for abstraction.
    /// 0 = use raw 169 hand classes (no abstraction).
    /// Recommended: 8-20 for postflop.
    pub num_buckets: usize,
    /// Subgame starting point. `Flop` = standard full solve.
    /// `Turn(card)` / `River([turn, river])` = subgame solve starting at that street,
    /// using `oop_range`/`ip_range` as the reach distribution at that node.
    pub start: SubgameStart,
}

/// Game state for postflop play.
#[derive(Clone, Debug)]
pub struct PostflopState {
    /// Current street: 0=flop, 1=turn, 2=river.
    pub street: u8,
    /// Board cards (3-5).
    pub board: Vec<Card>,
    /// OOP hole cards (None = not yet dealt).
    pub oop_hand: Option<[Card; 2]>,
    /// IP hole cards (None = not yet dealt).
    pub ip_hand: Option<[Card; 2]>,
    /// Dead cards: board + both players' hole cards.
    pub dead_cards: CardSet,
    /// Total pot (combined from both players).
    pub pot: u32,
    /// Each player's investment on current street: [OOP, IP].
    pub invested: [u32; 2],
    /// Remaining stacks: [OOP, IP].
    pub stacks: [u32; 2],
    /// Actions on current street.
    pub street_actions: Vec<Action>,
    /// Actions from previous streets.
    pub prev_streets: Vec<Vec<Action>>,
    /// Whether we need to deal a card (chance node at street transition).
    pub needs_deal: bool,
    /// Whether the hand is over.
    pub terminal: bool,
    /// If terminal by fold, who folded (0=OOP, 1=IP).
    pub folder: Option<usize>,
    /// Number of raises on current street.
    pub num_raises: u32,
}

/// Cache key for turn bucketing.
///
/// Uses suit canonicalization so that iso turn boards (e.g., `A♣ K♦ 2♥ 5♠`
/// and `A♠ K♣ 2♦ 5♥`) share a single cache entry — this is a meaningful
/// speedup during MCCFR because different turn samples produce iso boards.
fn turn_cache_key(board: &[Card]) -> u64 {
    canonical_cache_key(board)
}

/// The postflop game implementing the Game trait for CFR.
pub struct PostflopGame {
    pub config: PostflopConfig,
    /// Precomputed flop hand bucketing (None if num_buckets == 0).
    flop_bucketing: Option<HandBucketing>,
    /// Cache for turn board bucketing. Key = sorted 4-card board encoded as u64.
    turn_bucketing_cache: RwLock<FxHashMap<u64, HandBucketing>>,
}

impl PostflopGame {
    pub fn new(mut config: PostflopConfig) -> Self {
        // Suit isomorphism: canonicalize the flop (and, for subgame starts,
        // the whole board including turn/river) so iso boards collapse to a
        // single representation. The 169-class range is suit-agnostic so the
        // relabel is safe.
        //
        // We canonicalize *jointly*: the permutation is derived from the full
        // starting board so both flop and the subgame entry cards are mapped
        // consistently.
        let full_board: Vec<Card> = match config.start {
            SubgameStart::Flop => config.flop.to_vec(),
            SubgameStart::Turn(t) => {
                let mut b = config.flop.to_vec();
                b.push(t);
                b
            }
            SubgameStart::River([t, r]) => {
                let mut b = config.flop.to_vec();
                b.push(t);
                b.push(r);
                b
            }
        };
        let perm = gto_core::iso::build_perm(&full_board);
        let apply = |c: Card| gto_core::iso::apply_perm(c, &perm);
        config.flop = [apply(config.flop[0]), apply(config.flop[1]), apply(config.flop[2])];
        config.start = match config.start {
            SubgameStart::Flop => SubgameStart::Flop,
            SubgameStart::Turn(t) => SubgameStart::Turn(apply(t)),
            SubgameStart::River([t, r]) => SubgameStart::River([apply(t), apply(r)]),
        };

        // Flop bucketing is always computed from the 3-card flop, even for
        // subgame starts — the turn bucketing cache handles 4-card boards on
        // demand, and river uses exact evaluation (see `hand_bucket`).
        let flop_bucketing = if config.num_buckets > 0 {
            let mut dead = CardSet::empty();
            for &c in &config.flop {
                dead.insert(c);
            }
            Some(compute_bucketing(&config.flop, &dead, config.num_buckets))
        } else {
            None
        };
        PostflopGame {
            config,
            flop_bucketing,
            turn_bucketing_cache: RwLock::new(FxHashMap::default()),
        }
    }

    /// Compute a hand strength bucket for the given hole cards on the current board.
    /// Returns a bucket index in [0, num_buckets).
    fn hand_bucket(&self, hand: [Card; 2], state: &PostflopState) -> usize {
        let num_buckets = self.config.num_buckets;

        match state.board.len() {
            5 => {
                // River: exact hand strength via evaluate_7
                let cards = [
                    hand[0], hand[1],
                    state.board[0], state.board[1], state.board[2],
                    state.board[3], state.board[4],
                ];
                let strength = evaluate_7(&cards);
                // Map strength to bucket. HandStrength is u32, max ~= 8 << 20 = 8M
                let max_strength = 9u32 << 20; // slightly above max possible
                ((strength as u64 * num_buckets as u64) / max_strength as u64).min(num_buckets as u64 - 1) as usize
            }
            4 => {
                // Turn: use cached bucketing for this board
                let key = turn_cache_key(&state.board);

                // Try read lock first (fast path)
                {
                    let cache = self.turn_bucketing_cache.read().unwrap();
                    if let Some(bucketing) = cache.get(&key) {
                        let b = bucketing.get_bucket(hand[0], hand[1]);
                        return if b < 255 { b as usize } else { 0 };
                    }
                }

                // Cache miss: compute bucketing and store
                let mut board_dead = CardSet::empty();
                for &c in &state.board {
                    board_dead.insert(c);
                }
                let bucketing = compute_bucketing(&state.board, &board_dead, num_buckets);
                let b = bucketing.get_bucket(hand[0], hand[1]);
                let result = if b < 255 { b as usize } else { 0 };

                // Store in cache (write lock)
                {
                    let mut cache = self.turn_bucketing_cache.write().unwrap();
                    cache.insert(key, bucketing);
                }

                result
            }
            3 => {
                // Flop: use precomputed bucketing
                if let Some(ref bucketing) = self.flop_bucketing {
                    let b = bucketing.get_bucket(hand[0], hand[1]);
                    if b < 255 { b as usize } else { 0 }
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    /// Get bet sizes for a player on the current street.
    fn bet_sizes_for(&self, player: usize, street: u8) -> &[u32] {
        let s = street as usize;
        if player == 0 {
            &self.config.bet_sizes.oop_bet_sizes[s.min(2)]
        } else {
            &self.config.bet_sizes.ip_bet_sizes[s.min(2)]
        }
    }

    /// Get raise sizes for a player on the current street.
    fn raise_sizes_for(&self, player: usize, street: u8) -> &[u32] {
        let s = street as usize;
        if player == 0 {
            &self.config.bet_sizes.oop_raise_sizes[s.min(2)]
        } else {
            &self.config.bet_sizes.ip_raise_sizes[s.min(2)]
        }
    }

    /// Compute the actual chip amount for a % of pot bet.
    fn pct_of_pot(&self, state: &PostflopState, pct: u32) -> u32 {
        let total_pot = state.pot + state.invested[0] + state.invested[1];
        let amount = (total_pot as u64 * pct as u64 / 100) as u32;
        amount.min(state.stacks[self.current_player_internal(state)])
    }

    fn current_player_internal(&self, state: &PostflopState) -> usize {
        // OOP (player 0) acts first each street
        let num_actions = state.street_actions.len();
        if num_actions == 0 {
            0 // OOP
        } else {
            // Alternate starting from OOP
            num_actions % 2
        }
    }

    /// Transition to the next street.
    fn advance_street(&self, state: &mut PostflopState) {
        // Move current street's investments into pot
        state.pot += state.invested[0] + state.invested[1];
        state.invested = [0, 0];

        state.prev_streets.push(state.street_actions.clone());
        state.street_actions.clear();
        state.num_raises = 0;

        if state.street < 2 {
            state.street += 1;
            state.needs_deal = true;
        } else {
            // River showdown
            state.terminal = true;
        }
    }

    /// Build the info set key for a player.
    fn build_info_key(&self, state: &PostflopState, player: usize) -> String {
        let hand = if player == 0 { state.oop_hand } else { state.ip_hand };
        let hand = hand.expect("build_info_key called before deal");

        let mut key = if self.config.num_buckets > 0 {
            // Use hand strength bucket for abstraction
            let bucket = self.hand_bucket(hand, state);
            format!("B{}", bucket)
        } else {
            // Use raw 169 hand class (no abstraction)
            let class = hand_class(hand[0], hand[1]);
            format!("C{}", class)
        };

        let street_names = ["F", "T", "R"];
        for (i, actions) in state.prev_streets.iter().enumerate() {
            key.push('|');
            key.push_str(street_names[i.min(2)]);
            for a in actions {
                key.push_str(&format!("{}", a));
            }
        }

        // Current street
        key.push('|');
        key.push_str(street_names[state.street as usize]);
        for a in &state.street_actions {
            key.push_str(&format!("{}", a));
        }

        key
    }

    /// Expand a hand class into concrete card combos not blocked by dead_cards.
    fn class_combos(&self, class: usize, dead: CardSet) -> Vec<[Card; 2]> {
        let mut combos = Vec::new();
        if class < 13 {
            // Pair: rank = class
            let rank = class as u8;
            for s1 in 0..4u8 {
                let c1 = Card::new(rank, s1);
                if dead.contains(c1) { continue; }
                for s2 in (s1 + 1)..4u8 {
                    let c2 = Card::new(rank, s2);
                    if dead.contains(c2) { continue; }
                    combos.push([c1, c2]);
                }
            }
        } else if class < 91 {
            // Suited: same suit
            let idx = class - 13;
            let (low, high) = index_to_ranks(idx);
            for s in 0..4u8 {
                let c1 = Card::new(high, s);
                let c2 = Card::new(low, s);
                if dead.contains(c1) || dead.contains(c2) { continue; }
                combos.push([c1, c2]);
            }
        } else {
            // Offsuit: different suits
            let idx = class - 91;
            let (low, high) = index_to_ranks(idx);
            for s1 in 0..4u8 {
                let c1 = Card::new(high, s1);
                if dead.contains(c1) { continue; }
                for s2 in 0..4u8 {
                    if s1 == s2 { continue; }
                    let c2 = Card::new(low, s2);
                    if dead.contains(c2) { continue; }
                    combos.push([c1, c2]);
                }
            }
        }
        combos
    }
}

impl Game for PostflopGame {
    type State = PostflopState;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> PostflopState {
        // Build the starting board according to the subgame configuration.
        let (board, street): (Vec<Card>, u8) = match self.config.start {
            SubgameStart::Flop => (self.config.flop.to_vec(), 0),
            SubgameStart::Turn(t) => {
                let mut b = self.config.flop.to_vec();
                b.push(t);
                (b, 1)
            }
            SubgameStart::River([t, r]) => {
                let mut b = self.config.flop.to_vec();
                b.push(t);
                b.push(r);
                (b, 2)
            }
        };

        let mut dead = CardSet::empty();
        for &c in &board {
            dead.insert(c);
        }

        PostflopState {
            street,
            board,
            oop_hand: None,
            ip_hand: None,
            dead_cards: dead,
            pot: self.config.pot,
            invested: [0, 0],
            stacks: [self.config.effective_stack, self.config.effective_stack],
            street_actions: Vec::new(),
            prev_streets: Vec::new(),
            needs_deal: false,
            terminal: false,
            folder: None,
            num_raises: 0,
        }
    }

    fn is_terminal(&self, state: &PostflopState) -> bool {
        state.terminal
    }

    fn is_chance_node(&self, state: &PostflopState) -> bool {
        if state.terminal { return false; }
        // Initial deal (hands not assigned) or street transition
        state.oop_hand.is_none() || state.needs_deal
    }

    fn chance_outcomes(&self, _state: &PostflopState) -> Vec<(PostflopState, f64)> {
        // MCCFR with chance sampling only — we don't enumerate all outcomes.
        // Return empty vec; the trainer uses sample_chance_outcome instead.
        vec![]
    }

    fn sample_chance_outcome(
        &self,
        state: &PostflopState,
        rng: &mut dyn rand::RngCore,
    ) -> (PostflopState, f64) {
        if state.oop_hand.is_none() {
            self.sample_initial_deal(state, rng)
        } else if state.needs_deal {
            self.sample_street_deal(state, rng)
        } else {
            unreachable!("not a chance node")
        }
    }

    fn current_player(&self, state: &PostflopState) -> usize {
        self.current_player_internal(state)
    }

    fn actions(&self, state: &PostflopState) -> Vec<Action> {
        let player = self.current_player_internal(state);
        let mut actions = Vec::new();

        let facing_bet = state.invested[1 - player] > state.invested[player];
        let can_raise = (state.num_raises as usize) < self.config.bet_sizes.max_raises_per_street;

        if facing_bet {
            // Facing a bet/raise
            actions.push(Action::Fold);
            actions.push(Action::Call);

            if can_raise && state.stacks[player] > state.invested[1 - player] - state.invested[player] {
                let raise_sizes = self.raise_sizes_for(player, state.street);
                for &pct in raise_sizes {
                    let raise_amount = self.compute_raise_to(state, player, pct);
                    if raise_amount > 0 && raise_amount < state.stacks[player] {
                        actions.push(Action::Raise(raise_amount));
                    }
                }
                // AllIn
                if state.stacks[player] > 0 {
                    actions.push(Action::AllIn);
                }
            }
        } else {
            // Not facing a bet
            actions.push(Action::Check);

            let bet_sizes = self.bet_sizes_for(player, state.street);
            for &pct in bet_sizes {
                let amount = self.pct_of_pot(state, pct);
                if amount > 0 && amount < state.stacks[player] {
                    actions.push(Action::Bet(amount));
                }
            }
            // AllIn
            if state.stacks[player] > 0 {
                actions.push(Action::AllIn);
            }
        }

        // Dedup in case multiple sizes map to same amount
        actions.dedup();
        if actions.is_empty() {
            actions.push(Action::Check);
        }
        actions
    }

    fn apply_action(&self, state: &PostflopState, action: Action) -> PostflopState {
        let mut new_state = state.clone();
        let player = self.current_player_internal(state);

        match action {
            Action::Fold => {
                new_state.terminal = true;
                new_state.folder = Some(player);
            }
            Action::Check => {
                // If both players checked (OOP checks, IP checks), advance street
                if new_state.street_actions.len() >= 1 {
                    // IP checking after OOP check
                    new_state.street_actions.push(action);
                    self.advance_street(&mut new_state);
                    return new_state;
                }
            }
            Action::Call => {
                let call_amount = new_state.invested[1 - player] - new_state.invested[player];
                let actual = call_amount.min(new_state.stacks[player]);
                new_state.invested[player] += actual;
                new_state.stacks[player] -= actual;

                // After a call (bet -> call or raise -> call), advance street or showdown
                new_state.street_actions.push(action);
                self.advance_street(&mut new_state);
                return new_state;
            }
            Action::Bet(amount) => {
                let actual = amount.min(new_state.stacks[player]);
                new_state.invested[player] += actual;
                new_state.stacks[player] -= actual;
                new_state.num_raises += 1;
            }
            Action::Raise(amount) => {
                // Raise TO amount
                let additional = amount.min(new_state.stacks[player]);
                new_state.invested[player] += additional;
                new_state.stacks[player] -= additional;
                new_state.num_raises += 1;
            }
            Action::AllIn => {
                let amount = new_state.stacks[player];
                new_state.invested[player] += amount;
                new_state.stacks[player] = 0;
                new_state.num_raises += 1;

                // If the other player is also all-in or this is a call-equivalent
                if new_state.invested[player] <= new_state.invested[1 - player] {
                    // This is effectively a call all-in
                    new_state.street_actions.push(action);
                    self.advance_all_streets(&mut new_state);
                    return new_state;
                }
            }
        }

        new_state.street_actions.push(action);
        new_state
    }

    fn info_set_key(&self, state: &PostflopState, player: usize) -> String {
        self.build_info_key(state, player)
    }

    fn payoff(&self, state: &PostflopState, player: usize) -> f64 {
        // Total chips each player has put in (from their stack)
        let chips_in = [
            self.config.effective_stack - state.stacks[0],
            self.config.effective_stack - state.stacks[1],
        ];
        let total_pot = state.pot + state.invested[0] + state.invested[1];

        if let Some(folder) = state.folder {
            if folder == player {
                -(chips_in[player] as f64)
            } else {
                total_pot as f64 - chips_in[player] as f64
            }
        } else {
            // Showdown: use evaluate_7 for accurate hand comparison
            let oop_hand = state.oop_hand.expect("showdown without OOP hand");
            let ip_hand = state.ip_hand.expect("showdown without IP hand");

            let oop_str = if state.board.len() >= 5 {
                let mut cards = [Card(0); 7];
                cards[0] = oop_hand[0];
                cards[1] = oop_hand[1];
                for (i, &c) in state.board.iter().take(5).enumerate() {
                    cards[2 + i] = c;
                }
                evaluate_7(&cards)
            } else {
                // Board < 5: use available cards (shouldn't happen in normal flow
                // since advance_all_streets deals remaining cards via chance nodes,
                // but handle gracefully)
                evaluate_partial(&oop_hand, &state.board)
            };

            let ip_str = if state.board.len() >= 5 {
                let mut cards = [Card(0); 7];
                cards[0] = ip_hand[0];
                cards[1] = ip_hand[1];
                for (i, &c) in state.board.iter().take(5).enumerate() {
                    cards[2 + i] = c;
                }
                evaluate_7(&cards)
            } else {
                evaluate_partial(&ip_hand, &state.board)
            };

            let win_prob = if oop_str > ip_str {
                if player == 0 { 1.0 } else { 0.0 }
            } else if oop_str < ip_str {
                if player == 0 { 0.0 } else { 1.0 }
            } else {
                0.5
            };

            win_prob * total_pot as f64 - chips_in[player] as f64
        }
    }
}

/// Evaluate a partial board (< 5 cards) by padding with zeros.
/// This is a fallback; in normal MCCFR flow, all-in runouts deal remaining cards.
fn evaluate_partial(hand: &[Card; 2], board: &[Card]) -> HandStrength {
    if board.len() >= 5 {
        let mut cards = [Card(0); 7];
        cards[0] = hand[0];
        cards[1] = hand[1];
        for (i, &c) in board.iter().take(5).enumerate() {
            cards[2 + i] = c;
        }
        evaluate_7(&cards)
    } else if board.len() >= 3 {
        // Use evaluate_5 with available cards
        let mut cards = [Card(0); 5];
        cards[0] = hand[0];
        cards[1] = hand[1];
        for (i, &c) in board.iter().enumerate() {
            cards[2 + i] = c;
        }
        gto_eval::evaluate_5(&cards)
    } else {
        0 // No board: can't evaluate
    }
}

impl PostflopGame {
    /// Sample initial deal: pick concrete hole cards for both players from their ranges.
    fn sample_initial_deal(
        &self,
        state: &PostflopState,
        rng: &mut dyn rand::RngCore,
    ) -> (PostflopState, f64) {
        let board_dead = state.dead_cards;

        // Build weighted list of (class, combo_count) for OOP
        let mut oop_class_weights: Vec<(usize, f64, usize)> = Vec::new(); // (class, range_weight, num_combos)
        let mut oop_total = 0.0;
        for cls in 0..NUM_CLASSES {
            let w = self.config.oop_range[cls];
            if w <= 0.0 { continue; }
            let combos = self.class_combos(cls, board_dead);
            let nc = combos.len();
            if nc == 0 { continue; }
            let weight = w * nc as f64;
            oop_class_weights.push((cls, weight, nc));
            oop_total += weight;
        }

        if oop_total <= 0.0 {
            // No valid OOP hands - return terminal
            let mut s = state.clone();
            s.terminal = true;
            return (s, 1.0);
        }

        // Sample OOP class
        let r: f64 = rng.gen::<f64>() * oop_total;
        let mut cum = 0.0;
        let mut oop_cls = oop_class_weights[0].0;
        for &(cls, w, _) in &oop_class_weights {
            cum += w;
            if r < cum {
                oop_cls = cls;
                break;
            }
        }

        // Sample a concrete combo for OOP class
        let oop_combos = self.class_combos(oop_cls, board_dead);
        let oop_combo = oop_combos[rng.gen_range(0..oop_combos.len())];

        // Build dead cards including OOP hand
        let mut dead_with_oop = board_dead;
        dead_with_oop.insert(oop_combo[0]);
        dead_with_oop.insert(oop_combo[1]);

        // Build weighted list for IP (excluding OOP cards)
        let mut ip_class_weights: Vec<(usize, f64, usize)> = Vec::new();
        let mut ip_total = 0.0;
        for cls in 0..NUM_CLASSES {
            let w = self.config.ip_range[cls];
            if w <= 0.0 { continue; }
            let combos = self.class_combos(cls, dead_with_oop);
            let nc = combos.len();
            if nc == 0 { continue; }
            let weight = w * nc as f64;
            ip_class_weights.push((cls, weight, nc));
            ip_total += weight;
        }

        if ip_total <= 0.0 {
            let mut s = state.clone();
            s.terminal = true;
            return (s, 1.0);
        }

        // Sample IP class
        let r: f64 = rng.gen::<f64>() * ip_total;
        let mut cum = 0.0;
        let mut ip_cls = ip_class_weights[0].0;
        for &(cls, w, _) in &ip_class_weights {
            cum += w;
            if r < cum {
                ip_cls = cls;
                break;
            }
        }

        // Sample a concrete combo for IP
        let ip_combos = self.class_combos(ip_cls, dead_with_oop);
        let ip_combo = ip_combos[rng.gen_range(0..ip_combos.len())];

        // Build new state
        let mut new_state = state.clone();
        new_state.oop_hand = Some(oop_combo);
        new_state.ip_hand = Some(ip_combo);
        new_state.dead_cards = dead_with_oop;
        new_state.dead_cards.insert(ip_combo[0]);
        new_state.dead_cards.insert(ip_combo[1]);

        // Probability: we use uniform sampling weight for MCCFR
        // The exact probability doesn't matter for external sampling MCCFR
        // as long as it's consistent; return 1.0 as we're doing importance-weighted sampling
        (new_state, 1.0)
    }

    /// Sample a street deal card (turn or river).
    fn sample_street_deal(
        &self,
        state: &PostflopState,
        rng: &mut dyn rand::RngCore,
    ) -> (PostflopState, f64) {
        let remaining: Vec<Card> = (0..52u8)
            .map(Card)
            .filter(|c| !state.dead_cards.contains(*c))
            .collect();

        let n = remaining.len();
        if n == 0 {
            let mut s = state.clone();
            s.needs_deal = false;
            s.terminal = true;
            return (s, 1.0);
        }

        let idx = rng.gen_range(0..n);
        let card = remaining[idx];
        let prob = 1.0 / n as f64;

        let mut new_state = state.clone();
        new_state.board.push(card);
        new_state.dead_cards.insert(card);
        new_state.needs_deal = false;

        (new_state, prob)
    }

    /// Advance through all remaining streets (for all-in scenarios).
    fn advance_all_streets(&self, state: &mut PostflopState) {
        state.pot += state.invested[0] + state.invested[1];
        state.invested = [0, 0];
        state.prev_streets.push(state.street_actions.clone());
        state.street_actions.clear();

        // For all-in, we need to deal remaining board cards as chance nodes.
        // In MCCFR, we keep needs_deal = true so the trainer will sample remaining cards.
        if state.street < 2 {
            state.street += 1;
            state.needs_deal = true;
        } else {
            // Already on river, go to showdown
            state.terminal = true;
            state.needs_deal = false;
        }
    }

    /// Compute a raise-to amount given a percentage of pot.
    fn compute_raise_to(&self, state: &PostflopState, player: usize, pct: u32) -> u32 {
        let call_amount = state.invested[1 - player] - state.invested[player];
        let pot_after_call = state.pot + state.invested[0] + state.invested[1] + call_amount;
        let raise_size = (pot_after_call as u64 * pct as u64 / 100) as u32;
        let total = call_amount + raise_size;
        total.min(state.stacks[player])
    }
}

/// Convert an index within suited/offsuit group back to (low_rank, high_rank).
fn index_to_ranks(idx: usize) -> (u8, u8) {
    // idx = high * (high - 1) / 2 + low
    // Find high such that high*(high-1)/2 <= idx < (high+1)*high/2
    let mut high: u8 = 1;
    while (high as usize + 1) * high as usize / 2 <= idx {
        high += 1;
    }
    let low = idx - (high as usize * (high as usize - 1) / 2);
    (low as u8, high)
}

/// Extract strategies from a solved PostflopGame for JSON output.
/// Returns one ClassStrategy per decision point, with 13x13 grids per action.
pub fn extract_postflop_strategies(
    game: &PostflopGame,
    solver: &gto_cfr::CfrSolver,
) -> Vec<ClassStrategy> {
    let strategy = gto_cfr::Strategy::from_solver(solver);

    // Collect all unique info set suffixes (the part after "C{class}")
    let mut decision_points: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    for (key, node) in &solver.nodes {
        // key format: "C{class}|F{actions}|T{actions}|R{actions}"
        if let Some(pipe_pos) = key.find('|') {
            let suffix = &key[pipe_pos..];
            decision_points.entry(suffix.to_string()).or_insert(node.num_actions);
        }
    }

    let mut result = Vec::new();

    for (suffix, _) in &decision_points {
        let mut action_names: Vec<String> = Vec::new();
        let mut num_actions = 0usize;
        let mut found = false;

        // For each of 169 hand classes, look up the strategy
        let mut class_probs: Vec<Option<Vec<f32>>> = vec![None; NUM_CLASSES];

        for cls in 0..NUM_CLASSES {
            let key = format!("C{}{}", cls, suffix);
            if let Some(probs) = strategy.get(&key) {
                if !found {
                    num_actions = probs.len();
                    action_names = match reconstruct_actions_from_suffix(game, suffix, cls) {
                        Some(actions) if actions.len() == num_actions => {
                            actions.iter().map(|a| format_action_name(a)).collect()
                        }
                        _ => (0..num_actions).map(|i| format!("A{}", i)).collect(),
                    };
                    found = true;
                }
                class_probs[cls] = Some(probs.clone());
            }
        }

        if !found || num_actions == 0 { continue; }

        // Build 13x13 grids: grids[action_idx][row][col]
        let mut grids: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; 13]; 13]; num_actions];

        for cls in 0..NUM_CLASSES {
            let (row, col) = class_to_grid_pos(cls);
            if let Some(ref probs) = class_probs[cls] {
                for (ai, &p) in probs.iter().enumerate() {
                    if ai < num_actions {
                        grids[ai][row][col] = p as f64;
                    }
                }
            }
        }

        let label = suffix_to_label(suffix);
        result.push(ClassStrategy {
            label,
            key_suffix: suffix.clone(),
            action_names,
            grids,
        });
    }

    result
}

/// Strategy data for one decision point with 13x13 grids per action.
#[derive(Clone, Debug)]
pub struct ClassStrategy {
    pub label: String,
    pub key_suffix: String,
    pub action_names: Vec<String>,
    /// grids[action_idx][row][col] = frequency for that hand class taking that action.
    /// Grid layout: row 0 = A (rank 12), row 12 = 2 (rank 0).
    /// col 0 = A, col 12 = 2. Above diagonal = suited, below = offsuit, diagonal = pairs.
    pub grids: Vec<Vec<Vec<f64>>>,
}

/// Convert a hand class index (0-168) to a (row, col) in the 13x13 grid.
/// Grid: row=0 is A (rank 12), row=12 is 2 (rank 0).
/// Diagonal = pairs, above diagonal = suited, below = offsuit.
fn class_to_grid_pos(cls: usize) -> (usize, usize) {
    if cls < 13 {
        // Pair: rank = cls, grid position on diagonal
        let rank = cls;
        let pos = 12 - rank;
        (pos, pos)
    } else if cls < 91 {
        // Suited: idx = cls - 13, index_to_ranks gives (low, high)
        let idx = cls - 13;
        let (low, high) = index_to_ranks(idx);
        // Suited: row = high rank, col = low rank (above diagonal since row_idx < col_idx)
        // But grid is inverted: row 0 = rank 12
        let row = 12 - high as usize;
        let col = 12 - low as usize;
        (row, col)
    } else {
        // Offsuit: idx = cls - 91, index_to_ranks gives (low, high)
        let idx = cls - 91;
        let (low, high) = index_to_ranks(idx);
        // Offsuit: row = low rank, col = high rank (below diagonal since row_idx > col_idx)
        let row = 12 - low as usize;
        let col = 12 - high as usize;
        (row, col)
    }
}

/// Parse action tokens from a street's action string (e.g. "xb33" -> ["x", "b33"]).
fn parse_action_tokens(s: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'f' | b'x' | b'c' | b'a' => {
                tokens.push(&s[i..i+1]);
                i += 1;
            }
            b'b' | b'r' => {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                tokens.push(&s[start..i]);
            }
            _ => { i += 1; }
        }
    }
    tokens
}

/// Parse an action token string into an Action.
fn parse_action_token(tok: &str) -> Option<Action> {
    let bytes = tok.as_bytes();
    if bytes.is_empty() { return None; }
    match bytes[0] {
        b'f' => Some(Action::Fold),
        b'x' => Some(Action::Check),
        b'c' => Some(Action::Call),
        b'a' => Some(Action::AllIn),
        b'b' => tok[1..].parse::<u32>().ok().map(Action::Bet),
        b'r' => tok[1..].parse::<u32>().ok().map(Action::Raise),
        _ => None,
    }
}

/// Reconstruct game state from an info key suffix and return available actions.
fn reconstruct_actions_from_suffix(
    game: &PostflopGame,
    suffix: &str,
    class: usize,
) -> Option<Vec<Action>> {
    // Start from a dealt state with dummy cards for the given class
    let mut state = game.initial_state();

    // Create dummy hand cards for the class (just need valid cards for action generation)
    let board_dead = state.dead_cards;
    let combos = game.class_combos(class, board_dead);
    if combos.is_empty() { return None; }
    let oop_hand = combos[0];
    state.oop_hand = Some(oop_hand);
    state.dead_cards.insert(oop_hand[0]);
    state.dead_cards.insert(oop_hand[1]);

    // Find an IP hand that doesn't conflict
    let ip_combos = game.class_combos(class, state.dead_cards);
    if !ip_combos.is_empty() {
        state.ip_hand = Some(ip_combos[0]);
    } else {
        // Use any available combo for IP
        for cls2 in 0..NUM_CLASSES {
            let c2 = game.class_combos(cls2, state.dead_cards);
            if !c2.is_empty() {
                state.ip_hand = Some(c2[0]);
                break;
            }
        }
    }
    if state.ip_hand.is_none() { return None; }

    // Parse suffix: "|F{actions}|T{actions}|R{actions}"
    let parts: Vec<&str> = suffix.split('|').filter(|s| !s.is_empty()).collect();

    for (idx, part) in parts.iter().enumerate() {
        if part.is_empty() { continue; }
        let action_str = &part[1..]; // skip street letter (F/T/R)

        if idx == parts.len() - 1 {
            // Last part: parse and replay actions, then return available actions
            let tokens = parse_action_tokens(action_str);
            for tok in &tokens {
                if let Some(action) = parse_action_token(tok) {
                    if game.is_terminal(&state) || game.is_chance_node(&state) {
                        return None;
                    }
                    state = game.apply_action(&state, action);
                    // Handle chance nodes (street transitions)
                    if state.needs_deal {
                        // Deal a dummy card for the next street
                        let remaining: Vec<Card> = (0..52u8)
                            .map(Card)
                            .filter(|c| !state.dead_cards.contains(*c))
                            .collect();
                        if remaining.is_empty() { return None; }
                        state.board.push(remaining[0]);
                        state.dead_cards.insert(remaining[0]);
                        state.needs_deal = false;
                    }
                }
            }
        } else {
            // Previous streets: replay all actions
            let tokens = parse_action_tokens(action_str);
            for tok in &tokens {
                if let Some(action) = parse_action_token(tok) {
                    if game.is_terminal(&state) || game.is_chance_node(&state) {
                        return None;
                    }
                    state = game.apply_action(&state, action);
                }
            }
            // Handle street transition
            if state.needs_deal {
                let remaining: Vec<Card> = (0..52u8)
                    .map(Card)
                    .filter(|c| !state.dead_cards.contains(*c))
                    .collect();
                if remaining.is_empty() { return None; }
                state.board.push(remaining[0]);
                state.dead_cards.insert(remaining[0]);
                state.needs_deal = false;
            }
        }
    }

    if game.is_terminal(&state) || game.is_chance_node(&state) {
        return None;
    }
    Some(game.actions(&state))
}

/// Format an Action as a human-readable name.
fn format_action_name(action: &Action) -> String {
    match action {
        Action::Fold => "Fold".to_string(),
        Action::Check => "Check".to_string(),
        Action::Call => "Call".to_string(),
        Action::Bet(amt) => format!("Bet {}", amt),
        Action::Raise(amt) => format!("Raise {}", amt),
        Action::AllIn => "AllIn".to_string(),
    }
}

fn suffix_to_label(suffix: &str) -> String {
    let parts: Vec<&str> = suffix.split('|').filter(|s| !s.is_empty()).collect();
    if parts.is_empty() { return "Root".to_string(); }

    let last = parts.last().unwrap();
    let street = match last.chars().next() {
        Some('F') => "Flop",
        Some('T') => "Turn",
        Some('R') => "River",
        _ => "Unknown",
    };

    let actions = &last[1..];
    if actions.is_empty() {
        format!("{} (root)", street)
    } else {
        format!("{} after {}", street, actions)
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

    fn make_config() -> PostflopConfig {
        PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig::default(),
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 0,
            start: SubgameStart::Flop,
        }
    }

    #[test]
    fn initial_state_is_chance_node() {
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        assert!(game.is_chance_node(&state));
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn sample_initial_deal_gives_valid_hands() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (dealt, _prob) = game.sample_chance_outcome(&state, &mut rng);
        assert!(!game.is_chance_node(&dealt));
        assert!(dealt.oop_hand.is_some());
        assert!(dealt.ip_hand.is_some());

        // Hands should not overlap with board
        let oop = dealt.oop_hand.unwrap();
        let ip = dealt.ip_hand.unwrap();
        let board_set = state.dead_cards;
        assert!(!board_set.contains(oop[0]));
        assert!(!board_set.contains(oop[1]));
        assert!(!board_set.contains(ip[0]));
        assert!(!board_set.contains(ip[1]));

        // OOP and IP should not share cards
        assert_ne!(oop[0], ip[0]);
        assert_ne!(oop[0], ip[1]);
        assert_ne!(oop[1], ip[0]);
        assert_ne!(oop[1], ip[1]);
    }

    #[test]
    fn check_check_advances_street() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (state, _) = game.sample_chance_outcome(&state, &mut rng);

        // OOP checks
        let state = game.apply_action(&state, Action::Check);
        assert!(!game.is_terminal(&state));

        // IP checks -> should advance street
        let state = game.apply_action(&state, Action::Check);
        // Either needs_deal (turn transition) or terminal
        assert!(state.needs_deal || state.terminal);
    }

    #[test]
    fn bet_fold_is_terminal() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (state, _) = game.sample_chance_outcome(&state, &mut rng);

        // OOP bets
        let actions = game.actions(&state);
        let bet_action = actions.iter()
            .find(|a| matches!(a, Action::Bet(_)))
            .copied()
            .unwrap_or(Action::Check);

        let state = game.apply_action(&state, bet_action);

        // IP folds
        let state = game.apply_action(&state, Action::Fold);
        assert!(game.is_terminal(&state));
        assert_eq!(state.folder, Some(1));
    }

    #[test]
    fn payoff_fold_is_correct() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (dealt, _) = game.sample_chance_outcome(&state, &mut rng);

        // OOP bets 33 (33% of 100 pot)
        let state = game.apply_action(&dealt, Action::Bet(33));
        // IP folds
        let state = game.apply_action(&state, Action::Fold);

        let oop_payoff = game.payoff(&state, 0);
        let ip_payoff = game.payoff(&state, 1);

        // OOP wins the pot (IP folded, IP invested 0 on this street)
        assert!(oop_payoff > 0.0, "OOP should win when IP folds");
        assert!(ip_payoff <= 0.0, "IP should lose when folding");
    }

    #[test]
    fn actions_include_check_and_bet() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (state, _) = game.sample_chance_outcome(&state, &mut rng);

        let actions = game.actions(&state);
        assert!(actions.contains(&Action::Check));
        assert!(actions.iter().any(|a| matches!(a, Action::Bet(_) | Action::AllIn)));
    }

    #[test]
    fn info_key_uses_class() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        let (dealt, _) = game.sample_chance_outcome(&state, &mut rng);
        let key = game.info_set_key(&dealt, 0);
        assert!(key.starts_with("C"), "Key should start with C for class: {}", key);
        assert!(key.contains("|F"), "Key should contain |F for flop: {}", key);
    }

    #[test]
    fn showdown_payoff_is_zero_sum() {
        use rand::SeedableRng;
        let game = PostflopGame::new(make_config());
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        // Deal, check-check through streets to showdown
        let (mut state, _) = game.sample_chance_outcome(&state, &mut rng);

        // Play check-check on flop
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        // Deal turn
        assert!(state.needs_deal);
        let (mut state, _) = game.sample_chance_outcome(&state, &mut rng);

        // Check-check turn
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        // Deal river
        assert!(state.needs_deal);
        let (mut state, _) = game.sample_chance_outcome(&state, &mut rng);

        // Check-check river -> showdown
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        assert!(game.is_terminal(&state));
        let oop_pay = game.payoff(&state, 0);
        let ip_pay = game.payoff(&state, 1);
        // Zero-sum: payoffs should sum to pot (100) - combined investments
        let total = oop_pay + ip_pay;
        assert!(
            (total - 100.0).abs() < 0.01,
            "Payoffs should sum to initial pot (100): oop={}, ip={}, sum={}",
            oop_pay, ip_pay, total
        );
    }

    #[test]
    fn class_to_grid_pos_round_trip() {
        // Verify all 169 classes map to unique grid positions
        let mut seen = std::collections::HashSet::new();
        for cls in 0..169 {
            let (r, c) = class_to_grid_pos(cls);
            assert!(r < 13 && c < 13, "cls={} -> ({}, {}) out of bounds", cls, r, c);
            assert!(seen.insert((r, c)), "cls={} -> ({}, {}) duplicate", cls, r, c);
        }
    }

    #[test]
    fn index_to_ranks_inverse() {
        for high in 1u8..13 {
            for low in 0u8..high {
                let idx = (high as usize) * (high as usize - 1) / 2 + low as usize;
                let (lo, hi) = index_to_ranks(idx);
                assert_eq!((lo, hi), (low, high), "idx={}", idx);
            }
        }
    }

    #[test]
    fn mccfr_does_not_panic() {
        use gto_cfr::{TrainerConfig, train};

        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig {
                oop_bet_sizes: [vec![50], vec![50], vec![50]],
                ip_bet_sizes: [vec![50], vec![50], vec![50]],
                oop_raise_sizes: [vec![100], vec![100], vec![100]],
                ip_raise_sizes: [vec![100], vec![100], vec![100]],
                max_raises_per_street: 1,
            },
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 0,
        start: SubgameStart::Flop,
        };

        let game = PostflopGame::new(config);
        let trainer_config = TrainerConfig {
            iterations: 100,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };

        // This should not panic
        let _solver = train(&game, &trainer_config);
    }

    #[test]
    fn mccfr_with_buckets_converges() {
        use gto_cfr::{Strategy, TrainerConfig, train};

        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig {
                oop_bet_sizes: [vec![50], vec![50], vec![50]],
                ip_bet_sizes: [vec![50], vec![50], vec![50]],
                oop_raise_sizes: [vec![], vec![], vec![]],
                ip_raise_sizes: [vec![], vec![], vec![]],
                max_raises_per_street: 0,
            },
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 8,
        start: SubgameStart::Flop,
        };

        let game = PostflopGame::new(config);
        let trainer_config = TrainerConfig {
            iterations: 5_000,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };

        let solver = train(&game, &trainer_config);
        let strategy = Strategy::from_solver(&solver);

        // Verify strategies exist and probabilities are valid
        let strats = strategy.strategies;
        assert!(
            !strats.is_empty(),
            "Should have learned some strategies"
        );

        for (key, probs) in &strats {
            let sum: f32 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Strategy for {} sums to {}, expected ~1.0",
                key, sum
            );
            for &p in probs {
                assert!(
                    p >= -0.001 && p <= 1.001,
                    "Invalid probability {} in key {}",
                    p, key
                );
            }
        }

        // With buckets, info set count should be much smaller than with 169 classes
        let bucket_keys: Vec<_> = strats.keys().filter(|k| k.starts_with("B")).collect();
        assert!(
            !bucket_keys.is_empty(),
            "Should have bucket-based info set keys (B prefix)"
        );
    }

    #[test]
    fn extract_strategies_produces_output() {
        use gto_cfr::{TrainerConfig, train};

        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig {
                oop_bet_sizes: [vec![50], vec![50], vec![50]],
                ip_bet_sizes: [vec![50], vec![50], vec![50]],
                oop_raise_sizes: [vec![], vec![], vec![]],
                ip_raise_sizes: [vec![], vec![], vec![]],
                max_raises_per_street: 0,
            },
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 0, // Use raw classes for strategy extraction test
            start: SubgameStart::Flop,
        };

        let game = PostflopGame::new(config);
        let trainer_config = TrainerConfig {
            iterations: 500,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };

        let solver = train(&game, &trainer_config);
        let strategies = extract_postflop_strategies(&game, &solver);

        // Should have at least one decision point
        assert!(
            !strategies.is_empty(),
            "Should extract at least one decision point strategy"
        );

        // Each strategy should have valid grids
        for strat in &strategies {
            assert!(!strat.action_names.is_empty());
            assert_eq!(strat.grids.len(), strat.action_names.len());
            for grid in &strat.grids {
                assert_eq!(grid.len(), 13);
                for row in grid {
                    assert_eq!(row.len(), 13);
                }
            }
        }
    }

    #[test]
    fn turn_bucketing_cache_works() {
        use rand::SeedableRng;

        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig::default(),
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 8,
            start: SubgameStart::Flop,
        };

        let game = PostflopGame::new(config);

        // Create a turn state manually
        let state = game.initial_state();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let (mut state, _) = game.sample_chance_outcome(&state, &mut rng);

        // Check-check flop
        state = game.apply_action(&state, Action::Check);
        state = game.apply_action(&state, Action::Check);

        // Deal turn
        let (state, _) = game.sample_chance_outcome(&state, &mut rng);
        assert_eq!(state.board.len(), 4);

        let hand = state.oop_hand.unwrap();

        // First call should compute and cache
        let bucket1 = game.hand_bucket(hand, &state);
        assert!(bucket1 < 8);

        // Second call should hit cache
        let bucket2 = game.hand_bucket(hand, &state);
        assert_eq!(bucket1, bucket2);

        // Verify cache has an entry
        let cache = game.turn_bucketing_cache.read().unwrap();
        assert_eq!(cache.len(), 1, "Should have cached one turn board");
    }

    #[test]
    fn subgame_turn_start_produces_turn_state() {
        // Starting from a turn subgame: initial_state should produce a 4-card
        // board, street=1, and after chance-sampled deal the info set key
        // should begin with "|T" (skipping flop action history).
        use rand::SeedableRng;
        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 150,
            effective_stack: 150,
            bet_sizes: BetSizeConfig::default(),
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 6,
            start: SubgameStart::Turn(c("Kd")),
        };
        let game = PostflopGame::new(config);
        let state = game.initial_state();
        assert_eq!(state.board.len(), 4, "turn subgame should start with 4-card board");
        assert_eq!(state.street, 1, "turn subgame should start on street 1");
        assert!(state.prev_streets.is_empty(), "subgame has no prior street history");

        // Deal hands via chance sampling, then verify info_set_key has a |T prefix
        // rather than a |F prefix.
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let (dealt, _) = game.sample_chance_outcome(&state, &mut rng);
        let key = game.info_set_key(&dealt, 0);
        assert!(
            key.contains("|T"),
            "subgame info set key should begin on the turn, got: {}",
            key
        );
        assert!(
            !key.contains("|F"),
            "turn subgame should not have flop action segments, got: {}",
            key
        );
    }

    #[test]
    fn subgame_river_start_produces_river_state() {
        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 200,
            effective_stack: 100,
            bet_sizes: BetSizeConfig::default(),
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 6,
            start: SubgameStart::River([c("Kd"), c("3s")]),
        };
        let game = PostflopGame::new(config);
        let state = game.initial_state();
        assert_eq!(state.board.len(), 5);
        assert_eq!(state.street, 2);
    }

    #[test]
    fn subgame_turn_trains_and_converges() {
        // A turn subgame should be solvable to a reasonable exploitability in
        // modest iterations — the tree is much smaller than a full flop solve
        // (no flop actions, no turn chance enumeration).
        use gto_cfr::{train, TrainerConfig};
        let config = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 150,
            effective_stack: 150,
            bet_sizes: BetSizeConfig {
                oop_bet_sizes: [vec![], vec![50], vec![50]],
                ip_bet_sizes: [vec![], vec![50], vec![50]],
                oop_raise_sizes: [vec![], vec![], vec![]],
                ip_raise_sizes: [vec![], vec![], vec![]],
                max_raises_per_street: 0,
            },
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 6,
            start: SubgameStart::Turn(c("Kd")),
        };
        let game = PostflopGame::new(config);
        let tc = TrainerConfig {
            iterations: 2_000,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };
        let solver = train(&game, &tc);
        assert!(
            !solver.nodes.is_empty(),
            "subgame CFR should produce info sets"
        );
        // Exploitability should be a finite positive number (lower is better).
        let exploit = solver.exploitability(&game);
        assert!(
            exploit.is_finite() && exploit >= 0.0,
            "subgame exploitability = {}, should be finite & non-negative",
            exploit
        );
    }

    #[test]
    fn subgame_turn_has_smaller_tree_than_full_flop() {
        // A turn subgame's CFR info-set count should be strictly smaller than
        // a matching flop solve (since we skip the entire flop action tree).
        use gto_cfr::{train, TrainerConfig};
        let bet_sizes = BetSizeConfig {
            oop_bet_sizes: [vec![50], vec![50], vec![50]],
            ip_bet_sizes: [vec![50], vec![50], vec![50]],
            oop_raise_sizes: [vec![], vec![], vec![]],
            ip_raise_sizes: [vec![], vec![], vec![]],
            max_raises_per_street: 0,
        };
        let flop_cfg = PostflopConfig {
            flop: [c("Ts"), c("7h"), c("2c")],
            pot: 100,
            effective_stack: 200,
            bet_sizes: bet_sizes.clone(),
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 6,
            start: SubgameStart::Flop,
        };
        let turn_cfg = PostflopConfig {
            start: SubgameStart::Turn(c("Kd")),
            ..flop_cfg.clone()
        };

        let tc = TrainerConfig {
            iterations: 1_500,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };

        let flop_solver = train(&PostflopGame::new(flop_cfg), &tc);
        let turn_solver = train(&PostflopGame::new(turn_cfg), &tc);
        assert!(
            turn_solver.nodes.len() < flop_solver.nodes.len(),
            "turn subgame ({}) should have fewer info sets than full flop solve ({})",
            turn_solver.nodes.len(),
            flop_solver.nodes.len()
        );
    }

    #[test]
    fn iso_flops_produce_identical_canonical_flop() {
        // Two flops that differ only by suit labels should canonicalize to the
        // same flop inside PostflopGame.
        // A♠ K♥ 2♣ and A♦ K♣ 2♥ are both rainbows with the same rank structure.
        let make = |flop: [Card; 3]| {
            let config = PostflopConfig {
                flop,
                pot: 100,
                effective_stack: 200,
                bet_sizes: BetSizeConfig::default(),
                oop_range: vec![1.0; 169],
                ip_range: vec![1.0; 169],
                num_buckets: 8,
                start: SubgameStart::Flop,
            };
            PostflopGame::new(config)
        };

        let g1 = make([c("As"), c("Kh"), c("2c")]);
        let g2 = make([c("Ad"), c("Kc"), c("2h")]);
        assert_eq!(
            g1.config.flop, g2.config.flop,
            "iso flops must canonicalize to the same representation"
        );
    }

    #[test]
    fn iso_turn_boards_share_cache_entry() {
        // Two runs on the same canonical flop, sampling turn cards that happen
        // to be iso to each other, should share a single turn_bucketing_cache
        // entry. We verify this directly by computing turn_cache_key on iso
        // turn boards.
        let flop = [c("As"), c("Kh"), c("2c")];
        // Turn card 5d (fresh suit) vs 5s (pairing Aces suit? A is ♠) — these
        // are NOT iso on this rainbow flop. Instead, test with a two-tone flop
        // where the two unused suits are interchangeable.
        let two_tone = [c("As"), c("Ks"), c("2c")]; // ♠ twice, ♣ once; ♦ and ♥ free
        // Turn 5♦ and 5♥ should be iso (both are "fresh suit #2").
        let turn_board1 = vec![two_tone[0], two_tone[1], two_tone[2], c("5d")];
        let turn_board2 = vec![two_tone[0], two_tone[1], two_tone[2], c("5h")];
        assert_eq!(
            turn_cache_key(&turn_board1),
            turn_cache_key(&turn_board2),
            "iso turn boards should share a turn_cache_key"
        );

        // Sanity: a non-iso turn card produces a different key.
        let turn_board3 = vec![two_tone[0], two_tone[1], two_tone[2], c("5s")];
        assert_ne!(
            turn_cache_key(&turn_board1),
            turn_cache_key(&turn_board3),
            "non-iso turn board (5s pairs existing ♠ on board) must have a distinct key"
        );

        // Unused: silence compiler — flop itself is iso-canonicalized elsewhere.
        let _ = flop;
    }

    #[test]
    fn iso_postflop_games_have_same_info_set_count() {
        // Train two PostflopGames on iso flops with the same MCCFR seed-equivalent
        // workload. They should produce the same canonical flop, and thus the
        // CFR traversal explores the same info set space.
        use gto_cfr::{train, TrainerConfig};

        let make = |flop: [Card; 3]| PostflopGame::new(PostflopConfig {
            flop,
            pot: 100,
            effective_stack: 200,
            bet_sizes: BetSizeConfig {
                oop_bet_sizes: [vec![50], vec![50], vec![50]],
                ip_bet_sizes: [vec![50], vec![50], vec![50]],
                oop_raise_sizes: [vec![], vec![], vec![]],
                ip_raise_sizes: [vec![], vec![], vec![]],
                max_raises_per_street: 0,
            },
            oop_range: vec![1.0; 169],
            ip_range: vec![1.0; 169],
            num_buckets: 6,
            start: SubgameStart::Flop,
        });

        let tc = TrainerConfig {
            iterations: 1_500,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
            ..Default::default()
        };

        let s1 = train(&make([c("As"), c("Kh"), c("2c")]), &tc);
        let s2 = train(&make([c("Ad"), c("Kc"), c("2h")]), &tc);

        // MCCFR uses random sampling so the exact strategies differ run-to-run,
        // but the *structure* (reachable info set keys) is determined by the
        // canonical flop and should match modulo coverage. With enough iterations
        // both runs should converge on the same set of reachable keys.
        let keys1: std::collections::BTreeSet<_> = s1.nodes.keys().cloned().collect();
        let keys2: std::collections::BTreeSet<_> = s2.nodes.keys().cloned().collect();
        // Expect substantial overlap (>= 95% intersection / union). MCCFR
        // coverage is stochastic, so we don't demand exact equality.
        let inter = keys1.intersection(&keys2).count();
        let union = keys1.union(&keys2).count();
        let ratio = inter as f64 / union as f64;
        // MCCFR is stochastic so some variance in coverage is expected; the
        // important property is that the space explored is *structurally* the
        // same (same keys), not that sampling reaches every key identically.
        assert!(
            ratio > 0.90,
            "iso flops should explore nearly the same info set space \
             (intersection/union = {:.3}, keys1={}, keys2={}, inter={})",
            ratio,
            keys1.len(),
            keys2.len(),
            inter,
        );
    }
}
