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
    /// Raise sizes in bb for each raise level in the open-raise line.
    /// Index 0 = open, 1 = 3bet, 2 = 4bet, 3 = 5bet, etc.
    pub raise_sizes: Vec<f64>,
    /// Raise sizes in bb for the limp line.
    /// Index 0 = limp-raise (by defender), 1 = limp-reraise (by opener), etc.
    pub limp_raise_sizes: Vec<f64>,
    /// Maximum number of raises allowed. 0 = unlimited (stack is the limit).
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
            raise_sizes: vec![2.5, three_bet, four_bet],
            limp_raise_sizes: vec![3.5, 10.0],
            max_raises: 0, // unlimited
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
    /// 0 = unlimited (constrained by stack and raise_sizes length).
    pub fn with_max_raises(mut self, n: usize) -> Self {
        self.max_raises = n;
        self
    }

    /// Whether limp is allowed (only SB vs BB).
    pub fn can_limp(&self) -> bool {
        self.position == Position::SB && self.defender == Position::BB
    }

    /// Backward-compatible accessors.
    pub fn open_size(&self) -> f64 {
        self.raise_sizes.first().copied().unwrap_or(2.5)
    }
    pub fn three_bet_size(&self) -> f64 {
        self.raise_sizes.get(1).copied().unwrap_or(9.0)
    }
    pub fn four_bet_size(&self) -> f64 {
        self.raise_sizes.get(2).copied().unwrap_or(22.0)
    }

    /// Get the raise size for a given raise level (0-based).
    /// If the level exceeds the configured sizes, extrapolate by ~2.5x the previous size.
    fn raise_size_for_level(&self, level: usize) -> f64 {
        if level < self.raise_sizes.len() {
            self.raise_sizes[level]
        } else {
            // Extrapolate: ~2.5x last configured size, then keep doubling
            let last = *self.raise_sizes.last().unwrap_or(&2.5);
            let extra = level - self.raise_sizes.len() + 1;
            last * 2.5_f64.powi(extra as i32)
        }
    }

    /// Get the limp-raise size for a given level (0-based).
    fn limp_raise_size_for_level(&self, level: usize) -> f64 {
        if level < self.limp_raise_sizes.len() {
            self.limp_raise_sizes[level]
        } else {
            let last = *self.limp_raise_sizes.last().unwrap_or(&3.5);
            let extra = level - self.limp_raise_sizes.len() + 1;
            last * 2.5_f64.powi(extra as i32)
        }
    }

    /// Effective max raises, considering both the explicit limit and raise_sizes length.
    /// When max_raises == 0 (unlimited), constrained only by stack.
    fn effective_max_raises(&self) -> usize {
        if self.max_raises > 0 {
            self.max_raises
        } else {
            // Practical upper bound: stack-limited, but cap at a sane maximum
            20
        }
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

    /// Whether AllIn should be offered as an action.
    /// Only when the next raise size is a significant fraction of the stack,
    /// making AllIn a natural alternative to a sized raise.
    /// Threshold: raise >= 60% of stack, or stack <= 15bb with no raise available.
    fn should_offer_allin(&self, next_raise: Option<f64>) -> bool {
        let stack = self.config.stack_bb;
        match next_raise {
            Some(size) => size >= stack * 0.6,
            None => stack <= 15.0, // short stack only when no raise available
        }
    }

    /// Classify the current history into its line type and count raises.
    fn classify_history(history: &[Action]) -> HistoryInfo {
        if history.is_empty() {
            return HistoryInfo { line: Line::Open, num_raises: 0, last_action: None };
        }
        let first = &history[0];
        let line = if matches!(first, Action::Call) { Line::Limp } else { Line::Open };
        let num_raises = history.iter().filter(|a| matches!(a, Action::Raise(_))).count();
        let last_action = history.last().cloned();
        HistoryInfo { line, num_raises, last_action }
    }

    /// Check if another raise is allowed given current raise count.
    fn can_add_raise(&self, num_raises: usize) -> bool {
        let mr = self.config.max_raises;
        mr == 0 || num_raises < mr
    }

    /// Get available actions given current state.
    fn get_actions(&self, state: &PreflopState) -> Vec<Action> {
        let stack = self.config.stack_bb;
        let cfg = &self.config;
        let info = Self::classify_history(&state.history);

        match (&info.line, &info.last_action) {
            // Opener's first action
            (Line::Open, None) => {
                let mut actions = vec![Action::Fold];
                if cfg.can_limp() {
                    actions.push(Action::Call);
                }
                let open_size = cfg.raise_size_for_level(0);
                if open_size < stack {
                    actions.push(Action::Raise((open_size * 10.0) as u32));
                }
                if self.should_offer_allin(Some(open_size)) {
                    actions.push(Action::AllIn);
                }
                actions
            }

            // After a limp → defender checks or raises
            (Line::Limp, Some(Action::Call)) if info.num_raises == 0 => {
                let mut actions = vec![Action::Check];
                let raise_size = cfg.limp_raise_size_for_level(0);
                // Limp counts as the first "action" so limp-raise is the 2nd raise-level action
                let has_raise = self.can_add_raise(1) && raise_size < stack;
                if has_raise {
                    actions.push(Action::Raise((raise_size * 10.0) as u32));
                }
                if self.should_offer_allin(if has_raise { Some(raise_size) } else { None }) {
                    actions.push(Action::AllIn);
                }
                actions
            }

            // After AllIn → opponent: Fold / Call
            (_, Some(Action::AllIn)) => {
                vec![Action::Fold, Action::Call]
            }

            // After a Raise (in either line) → Fold / Call / [re-raise] / [AllIn]
            (line, Some(Action::Raise(_))) => {
                let mut actions = vec![Action::Fold, Action::Call];
                // For max_raises limit: in the limp line, limp counts as the 1st "action level"
                let effective_count = match line {
                    Line::Open => info.num_raises,
                    Line::Limp => info.num_raises + 1, // limp is level 1
                };
                let within_limit = self.can_add_raise(effective_count);
                let raise_size = match line {
                    Line::Open => cfg.raise_size_for_level(info.num_raises),
                    Line::Limp => cfg.limp_raise_size_for_level(info.num_raises),
                };
                let has_raise = within_limit && raise_size < stack;
                if has_raise {
                    actions.push(Action::Raise((raise_size * 10.0) as u32));
                }
                if self.should_offer_allin(if has_raise { Some(raise_size) } else { None }) {
                    actions.push(Action::AllIn);
                }
                actions
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

    /// Determine current player dynamically from history.
    /// Player 0 = opener, Player 1 = defender.
    fn current_player_impl(&self, state: &PreflopState) -> usize {
        if state.history.is_empty() {
            return 0; // Opener acts first
        }
        let first = &state.history[0];
        if matches!(first, Action::Call) {
            // Limp line: opener limps (0), defender acts (1), then alternates on raises
            // [Call] → 1, [Call,Raise] → 0, [Call,Raise,Raise] → 1, ...
            // [Call,AllIn] → 0, [Call,Raise,AllIn] → 1, ...
            let after_limp = &state.history[1..];
            if after_limp.is_empty() {
                return 1; // Defender acts after limp
            }
            // Count actions after limp (excluding the limp itself)
            // Defender goes first after limp, then alternates
            if after_limp.len() % 2 == 0 { 1 } else { 0 }
        } else {
            // Open/Raise/AllIn line: opener acts (0), then alternates
            // [Raise] → 1, [Raise,Raise] → 0, [Raise,Raise,Raise] → 1, ...
            // [AllIn] → 1, [Raise,AllIn] → 0, ...
            if state.history.len() % 2 == 0 { 0 } else { 1 }
        }
    }
}

/// Helper for classifying history.
#[derive(Debug)]
enum Line { Open, Limp }
#[derive(Debug)]
struct HistoryInfo {
    line: Line,
    num_raises: usize,
    last_action: Option<Action>,
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
        if state.history.is_empty() {
            return false;
        }
        let last = state.history.last().unwrap();
        match last {
            // Fold is always terminal (after at least one action)
            Action::Fold => state.history.len() >= 1,
            // Call after a raise/allin is terminal (showdown or fold response)
            Action::Call => {
                // Call as first action = limp, not terminal
                if state.history.len() == 1 {
                    return false;
                }
                // Call after Raise or AllIn = terminal
                let prev = &state.history[state.history.len() - 2];
                matches!(prev, Action::Raise(_) | Action::AllIn)
            }
            // Check is terminal only in limp line (limp → check = showdown)
            Action::Check => true,
            // Raise and AllIn are never terminal
            Action::Raise(_) | Action::AllIn => false,
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

/// Raise level names: 1=Open, 2=3bet, 3=4bet, 4=5bet, ...
fn raise_level_name(level: usize) -> &'static str {
    match level {
        0 => "Open",
        1 => "3bet",
        2 => "4bet",
        3 => "5bet",
        4 => "6bet",
        5 => "7bet",
        _ => "Raise",
    }
}

/// Extract all interesting strategy slices from a solved preflop game.
pub fn extract_preflop_strategies(
    strategy: &gto_cfr::Strategy,
    config: &PreflopConfig,
) -> Vec<PreflopStrategySet> {
    let stack = config.stack_bb;
    let opener_name = config.position.name();
    let defender_name = config.defender.name();

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
        let open_size = config.raise_size_for_level(0);
        let open_r = (open_size * 10.0) as u32;
        let mut action_defs: Vec<(Action, &str)> = vec![(Action::Fold, "Fold")];
        if config.can_limp() {
            action_defs.push((Action::Call, "Limp"));
        }
        action_defs.push((Action::Raise(open_r), "placeholder"));
        action_defs.push((Action::AllIn, "AllIn"));

        let open_label = format!("Open({:.1}bb)", open_size);
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

    // Dynamic raise line: Open → 3bet → 4bet → 5bet → ...
    {
        let max_r = config.effective_max_raises();
        let mut hist = String::new();
        let mut raise_amts: Vec<u32> = Vec::new();

        for level in 0.. {
            let size = config.raise_size_for_level(level);
            let r = (size * 10.0) as u32;
            if level == 0 {
                hist = format!("r{}", r);
            } else {
                hist = format!("{}r{}", hist, r);
            }
            raise_amts.push(r);

            let num_raises = level + 1;
            let is_opener = num_raises % 2 == 0; // even raises = opener's turn
            let player_name = if is_opener { opener_name } else { defender_name };

            // Check if next raise is possible
            let within_limit = max_r == 0 || num_raises < max_r;
            let next_size = config.raise_size_for_level(level + 1);
            let next_r = (next_size * 10.0) as u32;
            let next_name = raise_level_name(level + 1);

            let mut action_defs: Vec<(Action, String)> = vec![
                (Action::Fold, "Fold".to_string()),
                (Action::Call, "Call".to_string()),
            ];
            if within_limit && next_size < stack {
                action_defs.push((Action::Raise(next_r), format!("{}({:.0}bb)", next_name, next_size)));
            }
            action_defs.push((Action::AllIn, "AllIn".to_string()));

            let actions: Vec<(Action, String)> = action_defs
                .into_iter()
                .filter(|(a, _)| match a {
                    Action::Raise(amt) => (*amt as f64 / 10.0) < stack,
                    _ => true,
                })
                .collect();

            let freqs = extract_freqs(&hist, &actions);

            // Check if any strategy data exists for this level
            let has_data = freqs.iter().any(|f| f.iter().any(|&v| v > 0.0));
            if !has_data && num_raises >= 2 {
                break;
            }

            let label = if level == 0 {
                format!("{} vs {} Open", player_name, if is_opener { defender_name } else { opener_name })
            } else {
                let prev_name = raise_level_name(level);
                if is_opener {
                    format!("{} vs {}", player_name, prev_name)
                } else {
                    format!("{} vs {} {}", player_name, opener_name, prev_name)
                }
            };

            sets.push(PreflopStrategySet {
                label,
                history_prefix: hist.clone(),
                action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
                freqs,
            });

            // Stop if no further raise is possible
            if !within_limit || next_size >= stack {
                break;
            }
        }
    }

    // Limp-related sets (SB vs BB only)
    if config.can_limp() {
        let limp_raise_size = config.limp_raise_size_for_level(0);
        let limp_raise_r = (limp_raise_size * 10.0) as u32;

        // Defender vs Limp
        {
            let hist = "c".to_string();
            let actions = build_actions(&[
                (Action::Check, "Check"),
                (Action::Raise(limp_raise_r), &format!("Raise({:.1}bb)", limp_raise_size)),
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

        // Dynamic limp-raise line
        {
            let max_r = config.effective_max_raises();
            let mut hist = format!("cr{}", limp_raise_r);
            for level in 1.. {
                let within_limit = max_r == 0 || (level + 1) < max_r;
                let size = config.limp_raise_size_for_level(level);
                let r = (size * 10.0) as u32;
                let is_opener = level % 2 == 1; // odd level = opener
                let player_name = if is_opener { opener_name } else { defender_name };

                let mut action_defs: Vec<(Action, String)> = vec![
                    (Action::Fold, "Fold".to_string()),
                    (Action::Call, "Call".to_string()),
                ];
                if within_limit && size < stack {
                    action_defs.push((Action::Raise(r), format!("Reraise({:.0}bb)", size)));
                }
                action_defs.push((Action::AllIn, "AllIn".to_string()));

                let actions: Vec<(Action, String)> = action_defs
                    .into_iter()
                    .filter(|(a, _)| match a {
                        Action::Raise(amt) => (*amt as f64 / 10.0) < stack,
                        _ => true,
                    })
                    .collect();

                let freqs = extract_freqs(&hist, &actions);
                let has_data = freqs.iter().any(|f| f.iter().any(|&v| v > 0.0));
                if !has_data && level >= 2 {
                    break;
                }

                sets.push(PreflopStrategySet {
                    label: format!("{} vs Limp-Raise{}", player_name, if level > 1 { format!(" ({})", level) } else { String::new() }),
                    history_prefix: hist.clone(),
                    action_names: actions.iter().map(|(_, n)| n.clone()).collect(),
                    freqs,
                });

                if !within_limit || size >= stack {
                    break;
                }
                hist = format!("{}r{}", hist, r);
            }
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
        // At 100bb, AllIn should NOT be offered alongside a 2.5bb open
        assert!(!actions.contains(&Action::AllIn), "UTG 100bb should not have AllIn as open: {:?}", actions);
    }

    #[test]
    fn short_stack_has_allin() {
        // At 3bb, open 2.5bb is 83% of stack → AllIn should be offered
        let game = make_test_game_pos(3.0, Position::BTN);
        let state = game.initial_state();
        let outcomes = game.chance_outcomes(&state);
        let dealt = &outcomes[0].0;

        let actions = game.actions(dealt);
        assert!(actions.contains(&Action::AllIn), "3bb BTN should have AllIn: {:?}", actions);
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
        // Should have Fold, Call but no Raise (3bet) and no AllIn (100bb deep)
        assert!(
            !actions.iter().any(|a| matches!(a, Action::Raise(_))),
            "max_raises=1 should have no 3bet: {:?}",
            actions
        );
        assert!(actions.contains(&Action::Fold));
        assert!(actions.contains(&Action::Call));
        assert!(!actions.contains(&Action::AllIn), "100bb should not have AllIn vs open: {:?}", actions);
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
    fn max_raises_0_is_default_unlimited() {
        let config = PreflopConfig::for_matchup(100.0, Position::BTN, Position::BB);
        assert_eq!(config.max_raises, 0, "default should be 0 (unlimited)");
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
