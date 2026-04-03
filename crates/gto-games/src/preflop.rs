use gto_cfr::Game;
use gto_core::Action;
use rayon::prelude::*;

use crate::push_fold::{class_index_to_name, PushFoldData, NUM_CLASSES};

/// Positions at a 6-max table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Position {
    UTG,
    HJ,
    CO,
    BTN,
    SB,
    BB,
}

impl Position {
    /// All opener positions in standard order.
    pub const ALL: [Position; 5] = [
        Position::UTG,
        Position::HJ,
        Position::CO,
        Position::BTN,
        Position::SB,
    ];

    /// Display name.
    pub fn name(self) -> &'static str {
        match self {
            Position::UTG => "UTG",
            Position::HJ => "HJ",
            Position::CO => "CO",
            Position::BTN => "BTN",
            Position::SB => "SB",
            Position::BB => "BB",
        }
    }

    /// Blind amount for this position.
    pub fn blind(self) -> f64 {
        match self {
            Position::SB => 0.5,
            Position::BB => 1.0,
            _ => 0.0,
        }
    }

    /// Defender positions for this opener.
    pub fn defenders(self) -> &'static [Position] {
        match self {
            Position::UTG => &[Position::HJ, Position::CO, Position::BTN, Position::SB, Position::BB],
            Position::HJ => &[Position::CO, Position::BTN, Position::SB, Position::BB],
            Position::CO => &[Position::BTN, Position::SB, Position::BB],
            Position::BTN => &[Position::SB, Position::BB],
            Position::SB => &[Position::BB],
            Position::BB => &[],
        }
    }

    /// All 15 matchups (opener, defender) in standard order.
    pub fn all_matchups() -> Vec<(Position, Position)> {
        let mut matchups = Vec::new();
        for &opener in &Position::ALL {
            for &defender in opener.defenders() {
                matchups.push((opener, defender));
            }
        }
        matchups
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for preflop bet sizing.
#[derive(Clone, Debug)]
pub struct PreflopConfig {
    pub stack_bb: f64,
    pub position: Position,
    pub defender: Position,
    pub dead_money: f64,
    pub open_size: f64,
    pub three_bet_size: f64,
    pub four_bet_size: f64,
    pub limp_raise_size: f64,
    pub limp_reraise_size: f64,
    /// Tree depth limit: 1=Open only, 2=up to 3bet, 3=full (4bet).
    pub max_raises: usize,
}

impl PreflopConfig {
    /// Config for a specific matchup.
    pub fn for_matchup(stack_bb: f64, opener: Position, defender: Position) -> Self {
        let (three_bet, four_bet) = match opener {
            Position::UTG | Position::HJ => (9.0, 22.0),
            Position::CO => (8.5, 21.0),
            _ => (8.0, 20.0),
        };
        let dead_money = 1.5 - opener.blind() - defender.blind();
        PreflopConfig {
            stack_bb,
            position: opener,
            defender,
            dead_money,
            open_size: 2.5,
            three_bet_size: three_bet,
            four_bet_size: four_bet,
            limp_raise_size: 3.5,
            limp_reraise_size: 10.0,
            max_raises: 3,
        }
    }

    /// Default config for a given position vs BB (backward-compatible).
    pub fn for_position(stack_bb: f64, position: Position) -> Self {
        Self::for_matchup(stack_bb, position, Position::BB)
    }

    /// Backward-compatible default (SB vs BB).
    pub fn default_for_stack(stack_bb: f64) -> Self {
        Self::for_matchup(stack_bb, Position::SB, Position::BB)
    }

    /// Set the maximum number of raises (tree depth limit).
    pub fn with_max_raises(mut self, n: usize) -> Self {
        self.max_raises = n.clamp(1, 3);
        self
    }

    /// Whether limp is allowed (only SB vs BB).
    pub fn can_limp(&self) -> bool {
        self.position == Position::SB && self.defender == Position::BB
    }
}

/// Full preflop game: opener vs defender with open/limp/3bet/4bet actions.
pub struct PreflopGame {
    pub config: PreflopConfig,
    pub data: PushFoldData,
}

/// State of a preflop hand.
#[derive(Clone, Debug)]
pub struct PreflopState {
    pub opener_class: i16,
    pub defender_class: i16,
    pub history: Vec<Action>,
    pub opener_invested: f64,
    pub defender_invested: f64,
}

impl PreflopGame {
    pub fn new(config: PreflopConfig, data: PushFoldData) -> Self {
        PreflopGame { config, data }
    }

    /// Get available actions given current state.
    fn get_actions(&self, state: &PreflopState) -> Vec<Action> {
        let stack = self.config.stack_bb;
        let cfg = &self.config;

        match state.history.as_slice() {
            // Opener first action: Fold / [Limp(Call) only if SB vs BB] / Open(Raise) / AllIn
            [] => {
                let mut actions = vec![Action::Fold];
                if cfg.can_limp() {
                    actions.push(Action::Call); // Call = limp to 1bb
                }
                if cfg.open_size < stack {
                    actions.push(Action::Raise((cfg.open_size * 10.0) as u32));
                }
                actions.push(Action::AllIn);
                actions
            }

            // Opener folded → terminal
            [Action::Fold] => unreachable!(),

            // Opener limped → Defender: Check / Raise / AllIn
            [Action::Call] => {
                let mut actions = vec![Action::Check];
                if cfg.max_raises >= 2 && cfg.limp_raise_size < stack {
                    actions.push(Action::Raise((cfg.limp_raise_size * 10.0) as u32));
                }
                actions.push(Action::AllIn);
                actions
            }

            // Opener limped, Defender checked → terminal (showdown)
            [Action::Call, Action::Check] => unreachable!(),

            // Opener limped, Defender raised → Opener: Fold / Call / Reraise / AllIn
            [Action::Call, Action::Raise(_)] => {
                let mut actions = vec![Action::Fold, Action::Call];
                if cfg.max_raises >= 3 && cfg.limp_reraise_size < stack {
                    actions.push(Action::Raise((cfg.limp_reraise_size * 10.0) as u32));
                }
                actions.push(Action::AllIn);
                actions
            }

            [Action::Call, Action::Raise(_), Action::Fold] => unreachable!(),
            [Action::Call, Action::Raise(_), Action::Call] => unreachable!(),

            // Opener limped, Defender raised, Opener reraised → Defender: Fold / Call / AllIn
            [Action::Call, Action::Raise(_), Action::Raise(_)] => {
                let mut actions = vec![Action::Fold, Action::Call];
                actions.push(Action::AllIn);
                actions
            }

            [Action::Call, Action::Raise(_), Action::Raise(_), Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            [Action::Call, Action::Raise(_), Action::Raise(_), _] => unreachable!(),

            [Action::Call, Action::Raise(_), Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            [Action::Call, Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            // Opener opened → Defender: Fold / Call / 3bet / AllIn
            [Action::Raise(_)] => {
                let mut actions = vec![Action::Fold, Action::Call];
                if cfg.max_raises >= 2 && cfg.three_bet_size < stack {
                    actions.push(Action::Raise((cfg.three_bet_size * 10.0) as u32));
                }
                actions.push(Action::AllIn);
                actions
            }

            // Opener opened, Defender 3bet → Opener: Fold / Call / 4bet / AllIn
            [Action::Raise(_), Action::Raise(_)] => {
                let mut actions = vec![Action::Fold, Action::Call];
                if cfg.max_raises >= 3 && cfg.four_bet_size < stack {
                    actions.push(Action::Raise((cfg.four_bet_size * 10.0) as u32));
                }
                actions.push(Action::AllIn);
                actions
            }

            // Opener opened, Defender 3bet, Opener 4bet → Defender: Fold / Call / AllIn
            [Action::Raise(_), Action::Raise(_), Action::Raise(_)] => {
                let mut actions = vec![Action::Fold, Action::Call];
                actions.push(Action::AllIn);
                actions
            }

            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            [Action::Raise(_), Action::Raise(_), Action::Raise(_), _] => unreachable!(),

            [Action::Raise(_), Action::Raise(_), Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            [Action::Raise(_), Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            [Action::AllIn] => {
                vec![Action::Fold, Action::Call]
            }

            _ => unreachable!("unexpected history: {:?}", state.history),
        }
    }

    /// Compute invested amounts after applying an action.
    fn apply_investments(
        &self,
        state: &PreflopState,
        action: Action,
    ) -> (f64, f64) {
        let stack = self.config.stack_bb;
        let mut opener_inv = state.opener_invested;
        let mut defender_inv = state.defender_invested;
        let is_opener_acting = self.current_player_impl(state) == 0;

        match action {
            Action::Fold | Action::Check => {}
            Action::Call => {
                if is_opener_acting {
                    opener_inv = defender_inv.min(stack);
                } else {
                    defender_inv = opener_inv.min(stack);
                }
            }
            Action::Raise(amt_x10) => {
                let amt = amt_x10 as f64 / 10.0;
                let effective = amt.min(stack);
                if is_opener_acting {
                    opener_inv = effective;
                } else {
                    defender_inv = effective;
                }
            }
            Action::AllIn => {
                if is_opener_acting {
                    opener_inv = stack;
                } else {
                    defender_inv = stack;
                }
            }
            _ => {}
        }

        (opener_inv, defender_inv)
    }

    fn current_player_impl(&self, state: &PreflopState) -> usize {
        // Player 0 = opener, Player 1 = defender
        match state.history.as_slice() {
            [] => 0,
            [Action::Call] => 1,
            [Action::Call, Action::Raise(_)] => 0,
            [Action::Call, Action::AllIn] => 0,
            [Action::Call, Action::Raise(_), Action::Raise(_)] => 1,
            [Action::Call, Action::Raise(_), Action::Raise(_), Action::AllIn] => 0,
            [Action::Call, Action::Raise(_), Action::AllIn] => 1,
            [Action::Raise(_)] => 1,
            [Action::Raise(_), Action::Raise(_)] => 0,
            [Action::Raise(_), Action::AllIn] => 0,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_)] => 1,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::AllIn] => 0,
            [Action::Raise(_), Action::Raise(_), Action::AllIn] => 1,
            [Action::AllIn] => 1,
            _ => unreachable!("unexpected history for current_player: {:?}", state.history),
        }
    }
}

impl Game for PreflopGame {
    type State = PreflopState;

    fn num_players(&self) -> usize {
        2
    }

    fn initial_state(&self) -> PreflopState {
        PreflopState {
            opener_class: -1,
            defender_class: -1,
            history: Vec::new(),
            opener_invested: self.config.position.blind(),
            defender_invested: self.config.defender.blind(),
        }
    }

    fn is_terminal(&self, state: &PreflopState) -> bool {
        match state.history.as_slice() {
            [Action::Fold] => true,
            [Action::Call, Action::Check] => true,
            [Action::Call, Action::Raise(_), Action::Fold] => true,
            [Action::Call, Action::Raise(_), Action::Call] => true,
            [Action::Call, Action::Raise(_), Action::Raise(_), Action::Fold] => true,
            [Action::Call, Action::Raise(_), Action::Raise(_), Action::Call] => true,
            [Action::Call, Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Fold] => true,
            [Action::Call, Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Call] => true,
            [Action::Call, Action::Raise(_), Action::AllIn, Action::Fold] => true,
            [Action::Call, Action::Raise(_), Action::AllIn, Action::Call] => true,
            [Action::Call, Action::AllIn, Action::Fold] => true,
            [Action::Call, Action::AllIn, Action::Call] => true,
            [Action::Raise(_), Action::Fold] => true,
            [Action::Raise(_), Action::Call] => true,
            [Action::Raise(_), Action::Raise(_), Action::Fold] => true,
            [Action::Raise(_), Action::Raise(_), Action::Call] => true,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::Fold] => true,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::Call] => true,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Fold] => true,
            [Action::Raise(_), Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Call] => true,
            [Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Fold] => true,
            [Action::Raise(_), Action::Raise(_), Action::AllIn, Action::Call] => true,
            [Action::Raise(_), Action::AllIn, Action::Fold] => true,
            [Action::Raise(_), Action::AllIn, Action::Call] => true,
            [Action::AllIn, Action::Fold] => true,
            [Action::AllIn, Action::Call] => true,
            _ => false,
        }
    }

    fn is_chance_node(&self, state: &PreflopState) -> bool {
        state.opener_class < 0
    }

    fn chance_outcomes(&self, state: &PreflopState) -> Vec<(PreflopState, f64)> {
        let mut outcomes = Vec::with_capacity(NUM_CLASSES * NUM_CLASSES);
        for i in 0..NUM_CLASSES {
            for j in 0..NUM_CLASSES {
                let w = self.data.weights[i][j];
                if w > 0.0 {
                    outcomes.push((
                        PreflopState {
                            opener_class: i as i16,
                            defender_class: j as i16,
                            history: state.history.clone(),
                            opener_invested: state.opener_invested,
                            defender_invested: state.defender_invested,
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
        state: &PreflopState,
        rng: &mut dyn rand::RngCore,
    ) -> (PreflopState, f64) {
        use rand::Rng;
        let r: f64 = rng.gen();
        let idx = self.data.cumulative_weights
            .partition_point(|&cw| cw <= r)
            .min(NUM_CLASSES * NUM_CLASSES - 1);
        let i = idx / NUM_CLASSES;
        let j = idx % NUM_CLASSES;
        let prob = self.data.weights[i][j];
        (
            PreflopState {
                opener_class: i as i16,
                defender_class: j as i16,
                history: state.history.clone(),
                opener_invested: state.opener_invested,
                defender_invested: state.defender_invested,
            },
            prob,
        )
    }

    fn current_player(&self, state: &PreflopState) -> usize {
        self.current_player_impl(state)
    }

    fn actions(&self, state: &PreflopState) -> Vec<Action> {
        self.get_actions(state)
    }

    fn apply_action(&self, state: &PreflopState, action: Action) -> PreflopState {
        let (opener_inv, defender_inv) = self.apply_investments(state, action);
        let mut s = state.clone();
        s.history.push(action);
        s.opener_invested = opener_inv;
        s.defender_invested = defender_inv;
        s
    }

    fn info_set_key(&self, state: &PreflopState, player: usize) -> String {
        let class = if player == 0 {
            state.opener_class
        } else {
            state.defender_class
        };
        let name = class_index_to_name(class as usize);

        if state.history.is_empty() {
            return name;
        }

        let hist: String = state
            .history
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join("");
        format!("{}|{}", name, hist)
    }

    fn payoff(&self, state: &PreflopState, player: usize) -> f64 {
        debug_assert!(self.is_terminal(state));

        let last = state.history.last().unwrap();
        let is_showdown = !matches!(last, Action::Fold);
        let dead = self.config.dead_money;

        if is_showdown {
            let eq = self.data.equity[state.opener_class as usize][state.defender_class as usize];
            let pot = state.opener_invested + state.defender_invested + dead;
            let opener_ev = eq * pot - state.opener_invested;
            if player == 0 {
                opener_ev
            } else {
                -opener_ev
            }
        } else {
            let folder = self.current_player_impl_at_fold(state);
            if folder == player {
                // Folder loses their investment
                if player == 0 {
                    -state.opener_invested
                } else {
                    -state.defender_invested
                }
            } else {
                // Winner gets opponent's investment + dead money
                if player == 0 {
                    state.defender_invested + dead
                } else {
                    state.opener_invested + dead
                }
            }
        }
    }
}

impl PreflopGame {
    /// Who was the player that folded?
    fn current_player_impl_at_fold(&self, state: &PreflopState) -> usize {
        let mut pre_fold = state.clone();
        pre_fold.history.pop();
        self.current_player_impl(&pre_fold)
    }
}

/// Encode action history to a short string for display.
pub fn history_to_string(history: &[Action]) -> String {
    history.iter().map(|a| a.to_string()).collect::<Vec<_>>().join("")
}

/// Decision point identifiers for extracting strategy frequencies.
#[derive(Clone, Debug)]
pub struct PreflopStrategySet {
    pub label: String,
    pub history_prefix: String,
    pub action_names: Vec<String>,
    pub freqs: Vec<[f64; NUM_CLASSES]>, // one per action
}

/// Extract all interesting strategy slices from a solved preflop game.
pub fn extract_preflop_strategies(
    strategy: &gto_cfr::Strategy,
    config: &PreflopConfig,
) -> Vec<PreflopStrategySet> {
    let stack = config.stack_bb;
    let opener_name = config.position.name();
    let defender_name = config.defender.name();
    let open_r = (config.open_size * 10.0) as u32;
    let three_bet_r = (config.three_bet_size * 10.0) as u32;
    let four_bet_r = (config.four_bet_size * 10.0) as u32;
    let limp_raise_r = (config.limp_raise_size * 10.0) as u32;
    let limp_reraise_r = (config.limp_reraise_size * 10.0) as u32;

    let mut sets = Vec::new();

    let build_actions = |actions: &[(Action, &str)]| -> Vec<(Action, String)> {
        actions
            .iter()
            .filter(|(a, _)| {
                match a {
                    Action::Raise(amt) => (*amt as f64 / 10.0) < stack,
                    _ => true,
                }
            })
            .map(|(a, name)| (a.clone(), name.to_string()))
            .collect()
    };

    let extract_freqs = |hist: &str, actions: &[(Action, String)]| -> Vec<[f64; NUM_CLASSES]> {
        let mut freqs: Vec<[f64; NUM_CLASSES]> = vec![[0.0; NUM_CLASSES]; actions.len()];
        for i in 0..NUM_CLASSES {
            let key = if hist.is_empty() {
                class_index_to_name(i)
            } else {
                format!("{}|{}", class_index_to_name(i), hist)
            };
            if let Some(probs) = strategy.get(&key) {
                for (ai, _) in actions.iter().enumerate() {
                    if ai < probs.len() {
                        freqs[ai][i] = probs[ai];
                    }
                }
            }
        }
        freqs
    };

    // 1. Opener action
    {
        let mut action_defs: Vec<(Action, &str)> = vec![(Action::Fold, "Fold")];
        if config.can_limp() {
            action_defs.push((Action::Call, "Limp"));
        }
        action_defs.push((Action::Raise(open_r), &"placeholder"));
        action_defs.push((Action::AllIn, "AllIn"));

        let open_label = format!("Open({:.1}bb)", config.open_size);
        let actions: Vec<(Action, String)> = action_defs
            .iter()
            .filter(|(a, _)| match a {
                Action::Raise(amt) => (*amt as f64 / 10.0) < stack,
                _ => true,
            })
            .map(|(a, name)| {
                let n = if matches!(a, Action::Raise(_)) {
                    open_label.clone()
                } else {
                    name.to_string()
                };
                (a.clone(), n)
            })
            .collect();

        let freqs = extract_freqs("", &actions);
        sets.push(PreflopStrategySet {
            label: format!("{} Action", opener_name),
            history_prefix: String::new(),
            action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
            freqs,
        });
    }

    // 2. Defender vs Open
    {
        let hist = format!("r{}", open_r);
        let actions = build_actions(&[
            (Action::Fold, "Fold"),
            (Action::Call, "Call"),
            (Action::Raise(three_bet_r), &format!("3bet({:.0}bb)", config.three_bet_size)),
            (Action::AllIn, "AllIn"),
        ]);
        let freqs = extract_freqs(&hist, &actions);
        sets.push(PreflopStrategySet {
            label: format!("{} vs {} Open", defender_name, opener_name),
            history_prefix: hist,
            action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
            freqs,
        });
    }

    // 3. Opener vs 3bet — only when max_raises >= 2
    if config.max_raises >= 2 {
        let hist = format!("r{}r{}", open_r, three_bet_r);
        let actions = build_actions(&[
            (Action::Fold, "Fold"),
            (Action::Call, "Call"),
            (Action::Raise(four_bet_r), &format!("4bet({:.0}bb)", config.four_bet_size)),
            (Action::AllIn, "AllIn"),
        ]);
        let freqs = extract_freqs(&hist, &actions);
        sets.push(PreflopStrategySet {
            label: format!("{} vs 3bet", opener_name),
            history_prefix: hist,
            action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
            freqs,
        });
    }

    // 4. Defender vs 4bet — only when max_raises >= 3
    if config.max_raises >= 3 {
        let hist = format!("r{}r{}r{}", open_r, three_bet_r, four_bet_r);
        let actions = build_actions(&[
            (Action::Fold, "Fold"),
            (Action::Call, "Call"),
            (Action::AllIn, "AllIn"),
        ]);
        let freqs = extract_freqs(&hist, &actions);
        sets.push(PreflopStrategySet {
            label: format!("{} vs {} 4bet", defender_name, opener_name),
            history_prefix: hist,
            action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
            freqs,
        });
    }

    // 5-6. Limp-related sets (SB vs BB only)
    if config.can_limp() {
        // Defender vs Limp
        {
            let hist = "c".to_string();
            let actions = build_actions(&[
                (Action::Check, "Check"),
                (Action::Raise(limp_raise_r), &format!("Raise({:.1}bb)", config.limp_raise_size)),
                (Action::AllIn, "AllIn"),
            ]);
            let freqs = extract_freqs(&hist, &actions);
            sets.push(PreflopStrategySet {
                label: format!("{} vs Limp", defender_name),
                history_prefix: hist,
                action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
                freqs,
            });
        }

        // Opener vs Limp-Raise — only when max_raises >= 2
        if config.max_raises >= 2 {
            let hist = format!("cr{}", limp_raise_r);
            let actions = build_actions(&[
                (Action::Fold, "Fold"),
                (Action::Call, "Call"),
                (Action::Raise(limp_reraise_r), &format!("Reraise({:.0}bb)", config.limp_reraise_size)),
                (Action::AllIn, "AllIn"),
            ]);
            let freqs = extract_freqs(&hist, &actions);
            sets.push(PreflopStrategySet {
                label: format!("{} vs Limp-Raise", opener_name),
                history_prefix: hist,
                action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
                freqs,
            });
        }
    }

    sets
}

/// Display a preflop strategy as a chart.
pub fn display_preflop_chart(label: &str, action_name: &str, freqs: &[f64; NUM_CLASSES]) {
    use crate::push_fold::display_chart;
    let title = format!("{} — {}", label, action_name);
    display_chart(&title, freqs);
}

/// Result of solving one matchup.
#[derive(Clone, Debug)]
pub struct MatchupResult {
    pub opener: Position,
    pub defender: Position,
    pub config: PreflopConfig,
    pub strategies: Vec<PreflopStrategySet>,
    pub exploitability: f64,
}

/// Solve all 15 preflop matchups in parallel using rayon.
///
/// Returns a `MatchupResult` for each (opener, defender) pair.
/// `push_fold_data` is shared across all matchups (same equity matrix).
pub fn solve_all_matchups(
    stack_bb: f64,
    push_fold_data: &PushFoldData,
    trainer_config: &gto_cfr::TrainerConfig,
) -> Vec<MatchupResult> {
    let matchups = Position::all_matchups();

    matchups
        .par_iter()
        .map(|&(opener, defender)| {
            let config = PreflopConfig::for_matchup(stack_bb, opener, defender);
            let game = PreflopGame {
                config: config.clone(),
                data: PushFoldData::new(
                    push_fold_data.equity.clone(),
                    push_fold_data.weights.clone(),
                ),
            };

            let solver = gto_cfr::train(&game, trainer_config);
            let exploitability = solver.exploitability(&game);
            let strategy = gto_cfr::Strategy::from_solver(&solver);
            let strategies = extract_preflop_strategies(&strategy, &config);

            MatchupResult {
                opener,
                defender,
                config,
                strategies,
                exploitability,
            }
        })
        .collect()
}

/// Summary of opening frequencies across all positions.
#[derive(Clone, Debug)]
pub struct OpeningRangeSummary {
    pub position: Position,
    /// Open-raise frequency per hand class (averaged across all defender matchups).
    pub open_freq: [f64; NUM_CLASSES],
    /// Number of matchups averaged over.
    pub num_matchups: usize,
}

/// Extract opening range summaries from solved matchup results.
/// For each opener position, averages the open-raise frequency across all its defender matchups.
pub fn summarize_opening_ranges(results: &[MatchupResult]) -> Vec<OpeningRangeSummary> {
    let mut summaries = Vec::new();

    for &pos in &Position::ALL {
        let mut total_freq = [0.0f64; NUM_CLASSES];
        let mut count = 0usize;

        for result in results {
            if result.opener != pos {
                continue;
            }
            // Find the opener's first action strategy set (label: "{pos} Action")
            for strat_set in &result.strategies {
                if strat_set.label.contains("Action") {
                    // Find the Open(raise) action
                    for (ai, name) in strat_set.action_names.iter().enumerate() {
                        if name.contains("Open") || name.contains("Raise") {
                            for c in 0..NUM_CLASSES {
                                total_freq[c] += strat_set.freqs[ai][c];
                            }
                            count += 1;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        if count > 0 {
            for c in 0..NUM_CLASSES {
                total_freq[c] /= count as f64;
            }
        }

        summaries.push(OpeningRangeSummary {
            position: pos,
            open_freq: total_freq,
            num_matchups: count,
        });
    }

    summaries
}

#[cfg(test)]
mod tests {
    use super::*;
    use gto_core::Action;

    fn make_test_data() -> PushFoldData {
        PushFoldData::new(
            vec![vec![0.5; NUM_CLASSES]; NUM_CLASSES],
            {
                let w = 1.0 / (NUM_CLASSES * NUM_CLASSES) as f64;
                vec![vec![w; NUM_CLASSES]; NUM_CLASSES]
            },
        )
    }

    fn make_test_game(stack: f64) -> PreflopGame {
        make_test_game_pos(stack, Position::SB)
    }

    fn make_test_game_pos(stack: f64, position: Position) -> PreflopGame {
        let config = PreflopConfig::for_position(stack, position);
        PreflopGame::new(config, make_test_data())
    }

    fn make_test_matchup(stack: f64, opener: Position, defender: Position) -> PreflopGame {
        let config = PreflopConfig::for_matchup(stack, opener, defender);
        PreflopGame::new(config, make_test_data())
    }

    #[test]
    fn initial_state_is_chance() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        assert!(game.is_chance_node(&state));
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn sb_fold_is_terminal() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let folded = game.apply_action(dealt, Action::Fold);
        assert!(game.is_terminal(&folded));
        // SB vs BB: dead_money=0, SB loses 0.5bb
        assert!((game.payoff(&folded, 0) - (-0.5)).abs() < 0.01);
        assert!((game.payoff(&folded, 1) - 0.5).abs() < 0.01);
    }

    #[test]
    fn open_fold_is_terminal() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // SB opens
        let opened = game.apply_action(dealt, Action::Raise(25)); // 2.5bb
        assert!(!game.is_terminal(&opened));
        assert_eq!(game.current_player(&opened), 1); // Defender acts

        // BB folds
        let bb_fold = game.apply_action(&opened, Action::Fold);
        assert!(game.is_terminal(&bb_fold));
        // SB vs BB: dead_money=0, SB wins BB's 1bb
        assert!((game.payoff(&bb_fold, 0) - 1.0).abs() < 0.01);
        assert!((game.payoff(&bb_fold, 1) - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn open_call_showdown() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let opened = game.apply_action(dealt, Action::Raise(25));
        let called = game.apply_action(&opened, Action::Call);
        assert!(game.is_terminal(&called));
        // SB vs BB: pot=5bb+0 dead, eq=0.5 → EV = 0.5*5 - 2.5 = 0
        assert!(game.payoff(&called, 0).abs() < 0.01);
    }

    #[test]
    fn three_bet_fold() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let opened = game.apply_action(dealt, Action::Raise(25));
        let three_bet = game.apply_action(&opened, Action::Raise(80));
        assert_eq!(game.current_player(&three_bet), 0); // Opener acts

        let sb_fold = game.apply_action(&three_bet, Action::Fold);
        assert!(game.is_terminal(&sb_fold));
        // SB loses 2.5bb
        assert!((game.payoff(&sb_fold, 0) - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn limp_check_showdown() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let limped = game.apply_action(dealt, Action::Call);
        assert_eq!(game.current_player(&limped), 1); // Defender acts
        let checked = game.apply_action(&limped, Action::Check);
        assert!(game.is_terminal(&checked));
        // Pot = 2bb + 0 dead, eq=0.5 → EV = 0.5*2 - 1 = 0
        assert!(game.payoff(&checked, 0).abs() < 0.01);
    }

    #[test]
    fn four_bet_call_showdown() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let opened = game.apply_action(dealt, Action::Raise(25));
        let three_bet = game.apply_action(&opened, Action::Raise(80));
        let four_bet = game.apply_action(&three_bet, Action::Raise(200));
        assert_eq!(game.current_player(&four_bet), 1);

        let called = game.apply_action(&four_bet, Action::Call);
        assert!(game.is_terminal(&called));
        // Pot = 40bb + 0 dead, eq=0.5 → EV = 0.5*40 - 20 = 0
        assert!(game.payoff(&called, 0).abs() < 0.01);
    }

    #[test]
    fn info_set_key_format() {
        let game = make_test_game(100.0);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let key = game.info_set_key(dealt, 0);
        assert!(!key.contains('|'), "initial key should have no pipe: {}", key);

        let opened = game.apply_action(dealt, Action::Raise(25));
        let bb_key = game.info_set_key(&opened, 1);
        assert!(bb_key.contains("|r25"), "Defender key should have history: {}", bb_key);
    }

    // --- Non-SB position tests ---

    #[test]
    fn utg_no_limp_action() {
        let game = make_test_game_pos(100.0, Position::UTG);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let actions = game.actions(dealt);
        assert!(!actions.contains(&Action::Call), "UTG should not have limp: {:?}", actions);
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::AllIn));
    }

    #[test]
    fn utg_initial_investment_zero() {
        let game = make_test_game_pos(100.0, Position::UTG);
        let state = game.initial_state();
        assert!((state.opener_invested - 0.0).abs() < 0.001);
        assert!((state.defender_invested - 1.0).abs() < 0.001);
    }

    #[test]
    fn utg_fold_vs_bb() {
        // UTG vs BB: dead_money = 0.5
        let game = make_test_game_pos(100.0, Position::UTG);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let folded = game.apply_action(dealt, Action::Fold);
        assert!(game.is_terminal(&folded));
        // UTG invested 0, loses 0
        assert!((game.payoff(&folded, 0) - 0.0).abs() < 0.01);
        // BB wins 0 (UTG inv) + 0.5 (dead) = 0.5
        assert!((game.payoff(&folded, 1) - 0.5).abs() < 0.01);
    }

    #[test]
    fn btn_open_bb_fold() {
        // BTN vs BB: dead_money = 0.5
        let game = make_test_game_pos(100.0, Position::BTN);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let opened = game.apply_action(dealt, Action::Raise(25)); // 2.5bb
        let bb_fold = game.apply_action(&opened, Action::Fold);
        assert!(game.is_terminal(&bb_fold));
        // BTN wins BB's 1bb + 0.5 dead = 1.5
        assert!((game.payoff(&bb_fold, 0) - 1.5).abs() < 0.01);
    }

    #[test]
    fn sb_still_has_limp() {
        let game = make_test_game_pos(100.0, Position::SB);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let actions = game.actions(dealt);
        assert!(actions.contains(&Action::Call), "SB should have limp: {:?}", actions);
    }

    // --- Dead money matchup tests ---

    #[test]
    fn dead_money_values() {
        // SB vs BB: 0.0
        let cfg = PreflopConfig::for_matchup(100.0, Position::SB, Position::BB);
        assert!((cfg.dead_money - 0.0).abs() < 0.001);

        // UTG vs BB: 0.5
        let cfg = PreflopConfig::for_matchup(100.0, Position::UTG, Position::BB);
        assert!((cfg.dead_money - 0.5).abs() < 0.001);

        // UTG vs HJ: 1.5
        let cfg = PreflopConfig::for_matchup(100.0, Position::UTG, Position::HJ);
        assert!((cfg.dead_money - 1.5).abs() < 0.001);

        // BTN vs SB: 1.0
        let cfg = PreflopConfig::for_matchup(100.0, Position::BTN, Position::SB);
        assert!((cfg.dead_money - 1.0).abs() < 0.001);

        // CO vs BTN: 1.5
        let cfg = PreflopConfig::for_matchup(100.0, Position::CO, Position::BTN);
        assert!((cfg.dead_money - 1.5).abs() < 0.001);
    }

    #[test]
    fn utg_vs_hj_fold_payoffs() {
        // UTG vs HJ: dead_money = 1.5, both blinds = 0
        let game = make_test_matchup(100.0, Position::UTG, Position::HJ);
        let state = game.initial_state();
        assert!((state.opener_invested - 0.0).abs() < 0.001);
        assert!((state.defender_invested - 0.0).abs() < 0.001);

        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // UTG folds: loses 0, HJ wins dead_money 1.5
        let folded = game.apply_action(dealt, Action::Fold);
        assert!(game.is_terminal(&folded));
        assert!((game.payoff(&folded, 0) - 0.0).abs() < 0.01);
        assert!((game.payoff(&folded, 1) - 1.5).abs() < 0.01);
    }

    #[test]
    fn utg_vs_hj_open_fold() {
        // UTG vs HJ: dead_money = 1.5
        let game = make_test_matchup(100.0, Position::UTG, Position::HJ);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // UTG opens 2.5bb, HJ folds
        let opened = game.apply_action(dealt, Action::Raise(25));
        let hj_fold = game.apply_action(&opened, Action::Fold);
        assert!(game.is_terminal(&hj_fold));
        // UTG wins: HJ inv (0) + dead (1.5) = 1.5
        assert!((game.payoff(&hj_fold, 0) - 1.5).abs() < 0.01);
        // HJ loses 0
        assert!((game.payoff(&hj_fold, 1) - 0.0).abs() < 0.01);
    }

    #[test]
    fn utg_vs_hj_open_call_showdown() {
        // UTG vs HJ: dead_money = 1.5, eq=0.5
        let game = make_test_matchup(100.0, Position::UTG, Position::HJ);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let opened = game.apply_action(dealt, Action::Raise(25));
        let called = game.apply_action(&opened, Action::Call);
        assert!(game.is_terminal(&called));
        // pot = 2.5 + 2.5 + 1.5 = 6.5, EV = 0.5*6.5 - 2.5 = 0.75
        assert!((game.payoff(&called, 0) - 0.75).abs() < 0.01);
        assert!((game.payoff(&called, 1) - (-0.75)).abs() < 0.01);
    }

    #[test]
    fn btn_vs_sb_dead_money() {
        // BTN vs SB: dead_money = 1.0
        let game = make_test_matchup(100.0, Position::BTN, Position::SB);
        let state = game.initial_state();
        assert!((state.opener_invested - 0.0).abs() < 0.001);
        assert!((state.defender_invested - 0.5).abs() < 0.001);

        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // BTN opens, SB folds
        let opened = game.apply_action(dealt, Action::Raise(25));
        let sb_fold = game.apply_action(&opened, Action::Fold);
        assert!(game.is_terminal(&sb_fold));
        // BTN wins: SB inv (0.5) + dead (1.0) = 1.5
        assert!((game.payoff(&sb_fold, 0) - 1.5).abs() < 0.01);
        // SB loses 0.5
        assert!((game.payoff(&sb_fold, 1) - (-0.5)).abs() < 0.01);
    }

    #[test]
    fn no_limp_for_non_sb_bb() {
        // BTN vs SB: no limp allowed
        let game = make_test_matchup(100.0, Position::BTN, Position::SB);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;
        let actions = game.actions(dealt);
        assert!(!actions.contains(&Action::Call), "BTN vs SB should not have limp: {:?}", actions);

        // SB vs BB: limp allowed
        let game = make_test_matchup(100.0, Position::SB, Position::BB);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;
        let actions = game.actions(dealt);
        assert!(actions.contains(&Action::Call), "SB vs BB should have limp: {:?}", actions);
    }

    #[test]
    fn all_matchups_count() {
        let matchups = Position::all_matchups();
        assert_eq!(matchups.len(), 15);
    }

    #[test]
    #[ignore] // slow: ~3-5 min in release mode
    fn preflop_cfr_converges() {
        use gto_cfr::{train, TrainerConfig};

        let data = PushFoldData::compute(500_000);
        let config = PreflopConfig::default_for_stack(25.0);
        let game = PreflopGame::new(config, data);

        let tc = TrainerConfig {
            iterations: 10_000,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let exploit = solver.exploitability(&game);

        assert!(
            exploit < 0.5,
            "Preflop exploitability {:.4} should be < 0.5",
            exploit
        );

        let strategy = gto_cfr::Strategy::from_solver(&solver);
        if let Some(probs) = strategy.get("AA") {
            let fold_freq = probs[0];
            assert!(
                fold_freq < 0.05,
                "AA fold freq = {:.4}, should be < 0.05",
                fold_freq
            );
        }
    }

    #[test]
    fn preflop_cfr_basic_convergence() {
        use gto_cfr::{train, TrainerConfig};

        let config = PreflopConfig::default_for_stack(25.0);
        let game = PreflopGame::new(config, make_test_data());

        let tc = TrainerConfig {
            iterations: 100,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let _strategy = gto_cfr::Strategy::from_solver(&solver);
        assert!(solver.nodes.len() > 100, "Should have many info sets, got {}", solver.nodes.len());
    }

    #[test]
    fn utg_cfr_basic_convergence() {
        use gto_cfr::{train, TrainerConfig};

        let config = PreflopConfig::for_position(25.0, Position::UTG);
        let game = PreflopGame::new(config, make_test_data());

        let tc = TrainerConfig {
            iterations: 100,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let _strategy = gto_cfr::Strategy::from_solver(&solver);
        assert!(solver.nodes.len() > 50, "Should have info sets, got {}", solver.nodes.len());
    }

    #[test]
    fn utg_vs_hj_cfr_basic() {
        use gto_cfr::{train, TrainerConfig};

        let config = PreflopConfig::for_matchup(25.0, Position::UTG, Position::HJ);
        let game = PreflopGame::new(config, make_test_data());

        let tc = TrainerConfig {
            iterations: 100,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let _strategy = gto_cfr::Strategy::from_solver(&solver);
        assert!(solver.nodes.len() > 50, "Should have info sets, got {}", solver.nodes.len());
    }

    // --- max_raises tests ---

    #[test]
    fn max_raises_1_no_3bet() {
        let config = PreflopConfig::for_matchup(100.0, Position::BTN, Position::BB)
            .with_max_raises(1);
        let game = PreflopGame::new(config, make_test_data());
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // Open
        let opened = game.apply_action(dealt, Action::Raise(25));
        let actions = game.actions(&opened);
        // Should have Fold, Call, AllIn but no Raise (3bet)
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "max_raises=1 should have no 3bet: {:?}",
            actions
        );
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(actions.contains(&Action::AllIn));
    }

    #[test]
    fn max_raises_2_no_4bet() {
        let config = PreflopConfig::for_matchup(100.0, Position::BTN, Position::BB)
            .with_max_raises(2);
        let game = PreflopGame::new(config, make_test_data());
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // Open
        let opened = game.apply_action(dealt, Action::Raise(25));
        let actions = game.actions(&opened);
        // 3bet should be present
        assert!(
            actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "max_raises=2 should have 3bet: {:?}",
            actions
        );

        // After 3bet
        let three_bet = game.apply_action(&opened, Action::Raise(80));
        let actions2 = game.actions(&three_bet);
        // 4bet should NOT be present
        assert!(
            !actions2.iter().any(|a| matches!(a, Action::Raise(_))),
            "max_raises=2 should have no 4bet: {:?}",
            actions2
        );
    }

    #[test]
    fn max_raises_1_limp_no_raise() {
        let config = PreflopConfig::for_matchup(100.0, Position::SB, Position::BB)
            .with_max_raises(1);
        let game = PreflopGame::new(config, make_test_data());
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        // Limp
        let limped = game.apply_action(dealt, Action::Call);
        let actions = game.actions(&limped);
        // Should have Check, AllIn but no Raise
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "max_raises=1 limp should have no raise: {:?}",
            actions
        );
    }

    #[test]
    fn max_raises_1_cfr_converges() {
        use gto_cfr::{train, TrainerConfig};

        let config = PreflopConfig::default_for_stack(25.0).with_max_raises(1);
        let game = PreflopGame::new(config, make_test_data());

        let tc = TrainerConfig {
            iterations: 100,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let _strategy = gto_cfr::Strategy::from_solver(&solver);
        assert!(solver.nodes.len() > 0, "Should have info sets");
    }

    #[test]
    fn max_raises_3_is_default() {
        let config = PreflopConfig::for_matchup(100.0, Position::BTN, Position::BB);
        assert_eq!(config.max_raises, 3);
    }

    #[test]
    fn extract_strategies_respects_max_raises() {
        use gto_cfr::{train, Strategy, TrainerConfig};

        let config = PreflopConfig::for_matchup(25.0, Position::BTN, Position::BB)
            .with_max_raises(1);
        let game = PreflopGame::new(config.clone(), make_test_data());

        let tc = TrainerConfig {
            iterations: 50,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &tc);
        let strategy = Strategy::from_solver(&solver);

        let sets = extract_preflop_strategies(&strategy, &config);
        // max_raises=1: should have Opener Action + Defender vs Open = 2 sets
        assert_eq!(sets.len(), 2, "max_raises=1 should produce 2 strategy sets, got {}", sets.len());
    }

    #[test]
    fn solve_all_matchups_returns_15_results() {
        use gto_cfr::TrainerConfig;

        let data = make_test_data();
        let tc = TrainerConfig {
            iterations: 50,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };

        let results = solve_all_matchups(25.0, &data, &tc);
        assert_eq!(results.len(), 15, "Should solve all 15 matchups");

        // Verify all position pairs are present
        let expected = Position::all_matchups();
        for (i, &(opener, defender)) in expected.iter().enumerate() {
            assert_eq!(results[i].opener, opener);
            assert_eq!(results[i].defender, defender);
        }

        // Each result should have strategies
        for result in &results {
            assert!(
                !result.strategies.is_empty(),
                "{} vs {} should have strategies",
                result.opener, result.defender
            );
        }
    }

    #[test]
    fn summarize_opening_ranges_covers_all_positions() {
        use gto_cfr::TrainerConfig;

        let data = make_test_data();
        let tc = TrainerConfig {
            iterations: 50,
            use_cfr_plus: true,
            use_chance_sampling: false,
            print_interval: 0,
        };

        let results = solve_all_matchups(25.0, &data, &tc);
        let summaries = summarize_opening_ranges(&results);

        assert_eq!(summaries.len(), 5, "Should have 5 opener positions");

        // UTG should have 5 matchups, SB should have 1
        let utg_summary = summaries.iter().find(|s| s.position == Position::UTG).unwrap();
        let sb_summary = summaries.iter().find(|s| s.position == Position::SB).unwrap();
        assert_eq!(utg_summary.num_matchups, 5);
        assert_eq!(sb_summary.num_matchups, 1);
    }
}
