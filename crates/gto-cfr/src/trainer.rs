use crate::cfr::CfrSolver;
use gto_core::Action;
use rand::Rng;

/// Trait that all games must implement to be solvable by CFR.
pub trait Game {
    /// The game state type.
    type State: Clone;

    /// Number of players.
    fn num_players(&self) -> usize;

    /// Create the initial game state (before any chance or player actions).
    fn initial_state(&self) -> Self::State;

    /// Is this a terminal state (hand is over)?
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Is this a chance node (cards being dealt)?
    fn is_chance_node(&self, state: &Self::State) -> bool;

    /// Get all possible chance outcomes with their probabilities.
    /// Returns (next_state, probability) pairs.
    fn chance_outcomes(&self, state: &Self::State) -> Vec<(Self::State, f64)>;

    /// Sample a single chance outcome. Returns (next_state, probability).
    /// Default implementation uses linear scan over chance_outcomes().
    fn sample_chance_outcome(
        &self,
        state: &Self::State,
        rng: &mut dyn rand::RngCore,
    ) -> (Self::State, f64) {
        let outcomes = self.chance_outcomes(state);
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        for (next_state, prob) in outcomes {
            cumulative += prob;
            if r < cumulative {
                return (next_state, prob);
            }
        }
        // Fallback to last outcome (rounding)
        unreachable!("chance_outcomes probabilities should sum to 1.0")
    }

    /// Which player acts at this state?
    fn current_player(&self, state: &Self::State) -> usize;

    /// Available actions at this state.
    fn actions(&self, state: &Self::State) -> Vec<Action>;

    /// Apply an action and return the new state.
    fn apply_action(&self, state: &Self::State, action: Action) -> Self::State;

    /// Get the information set key for a player at this state.
    /// This encodes what the player can observe (their private cards + public actions).
    fn info_set_key(&self, state: &Self::State, player: usize) -> String;

    /// Get the payoff for a player at a terminal state.
    fn payoff(&self, state: &Self::State, player: usize) -> f64;
}

/// Training configuration.
pub struct TrainerConfig {
    pub iterations: usize,
    pub use_cfr_plus: bool,
    pub use_chance_sampling: bool,
    pub print_interval: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        TrainerConfig {
            iterations: 100_000,
            use_cfr_plus: false,
            use_chance_sampling: false,
            print_interval: 10_000,
        }
    }
}

/// Run CFR training on a game and return the solver with trained strategy.
pub fn train<G: Game + Sync>(game: &G, config: &TrainerConfig) -> CfrSolver
where
    G::State: Send,
{
    train_with_callback(game, config, |_, _| {})
}

/// Run CFR training with a progress callback.
/// The callback receives (current_iteration, total_iterations).
pub fn train_with_callback<G: Game + Sync, F: FnMut(usize, usize)>(
    game: &G,
    config: &TrainerConfig,
    mut on_progress: F,
) -> CfrSolver
where
    G::State: Send,
{
    let mut solver = CfrSolver::new();
    let mut cumulative_value = 0.0;
    // Report progress roughly 20 times during training
    let report_interval = (config.iterations / 20).max(1);

    for i in 1..=config.iterations {
        let value = match (config.use_chance_sampling, config.use_cfr_plus) {
            (true, true) => solver.iterate_chance_sampling_plus(game),
            (true, false) => solver.iterate_chance_sampling(game),
            (false, true) => solver.iterate_plus(game),
            (false, false) => solver.iterate(game),
        };
        cumulative_value += value;

        if config.print_interval > 0 && i % config.print_interval == 0 {
            let avg_value = cumulative_value / i as f64;
            let exploit = solver.exploitability(game);
            println!(
                "Iteration {:>7}: avg game value = {:.6}, exploitability = {:.6}",
                i, avg_value, exploit
            );
        }

        if i % report_interval == 0 || i == config.iterations {
            on_progress(i, config.iterations);
        }
    }

    solver
}
