use crate::info_set::{InfoSetKey, InfoSetNode};
use crate::trainer::Game;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Vanilla CFR (Counterfactual Regret Minimization) solver.
///
/// Recursively traverses the game tree, computing counterfactual values
/// and accumulating regrets. The average strategy converges to a Nash equilibrium.
pub struct CfrSolver {
    /// Map from information set key to node data (regrets + cumulative strategy).
    pub nodes: FxHashMap<InfoSetKey, InfoSetNode>,
    /// RNG for chance sampling MCCFR.
    rng: StdRng,
}

impl CfrSolver {
    pub fn new() -> Self {
        CfrSolver {
            nodes: FxHashMap::default(),
            rng: StdRng::from_entropy(),
        }
    }

    /// Run one iteration of CFR for all players.
    /// Returns the expected game value for player 0.
    pub fn iterate<G: Game>(&mut self, game: &G) -> f64 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0;

        // Must traverse for each player so that all players' regrets get updated
        for p in 0..num_players {
            let reach_probs = vec![1.0; num_players];
            let v = self.cfr(game, &initial_state, &reach_probs, p);
            if p == 0 {
                value = v;
            }
        }

        value
    }

    /// Core CFR traversal.
    ///
    /// Returns the counterfactual value for `traversing_player` at the given state.
    fn cfr<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach_probs: &[f64],
        traversing_player: usize,
    ) -> f64 {
        // Terminal node: return payoff
        if game.is_terminal(state) {
            return game.payoff(state, traversing_player);
        }

        // Chance node: enumerate all outcomes and compute expected value
        if game.is_chance_node(state) {
            let outcomes = game.chance_outcomes(state);
            let mut ev = 0.0;
            for (next_state, prob) in &outcomes {
                ev += prob * self.cfr(game, next_state, reach_probs, traversing_player);
            }
            return ev;
        }

        let current_player = game.current_player(state);
        let actions = game.actions(state);
        let num_actions = actions.len();
        let info_key = game.info_set_key(state, current_player);

        // Get or create node
        if !self.nodes.contains_key(&info_key) {
            self.nodes.insert(info_key.clone(), InfoSetNode::new(num_actions));
        }

        // Compute current strategy via regret matching (cached)
        let node = self.nodes.get_mut(&info_key).unwrap();
        let strategy = node.current_strategy();

        // Accumulate strategy weighted by this player's reach probability
        // Reuse already-computed strategy to avoid double computation
        node.accumulate_strategy_with(&strategy, reach_probs[current_player]);

        // Compute counterfactual value for each action
        let mut action_values = vec![0.0; num_actions];
        let mut node_value = 0.0;

        for (i, action) in actions.iter().enumerate() {
            let next_state = game.apply_action(state, *action);

            // Update reach probabilities
            let mut new_reach = reach_probs.to_vec();
            new_reach[current_player] *= strategy[i];

            action_values[i] = self.cfr(game, &next_state, &new_reach, traversing_player);
            node_value += strategy[i] * action_values[i];
        }

        // Update regrets only for the traversing player's information sets
        if current_player == traversing_player {
            // Counterfactual reach: product of all opponents' reach probabilities
            let mut opponent_reach = 1.0;
            for (p, &r) in reach_probs.iter().enumerate() {
                if p != current_player {
                    opponent_reach *= r;
                }
            }

            let node = self.nodes.get_mut(&info_key).unwrap();
            for i in 0..num_actions {
                let regret = opponent_reach * (action_values[i] - node_value);
                node.cumulative_regret[i] += regret;
            }
            node.invalidate_cache();
        }

        node_value
    }

    /// Run CFR+ iteration: same as CFR but clamps negative regrets to zero.
    pub fn iterate_plus<G: Game>(&mut self, game: &G) -> f64 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0;

        for p in 0..num_players {
            let reach_probs = vec![1.0; num_players];
            let v = self.cfr_plus(game, &initial_state, &reach_probs, p);
            if p == 0 {
                value = v;
            }
        }

        // Clamp all negative regrets to 0 (CFR+ modification)
        for node in self.nodes.values_mut() {
            let mut changed = false;
            for r in &mut node.cumulative_regret {
                if *r < 0.0 {
                    *r = 0.0;
                    changed = true;
                }
            }
            if changed {
                node.invalidate_cache();
            }
        }

        value
    }

    /// CFR+ traversal (identical to CFR, clamping happens after full iteration).
    fn cfr_plus<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach_probs: &[f64],
        traversing_player: usize,
    ) -> f64 {
        // Reuse the same logic as vanilla CFR
        self.cfr(game, state, reach_probs, traversing_player)
    }

    /// Run one iteration of Chance Sampling MCCFR.
    /// Samples a single chance outcome instead of enumerating all.
    pub fn iterate_chance_sampling<G: Game>(&mut self, game: &G) -> f64 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0;

        for p in 0..num_players {
            let reach_probs = vec![1.0; num_players];
            let v = self.cfr_chance_sampling(game, &initial_state, &reach_probs, p);
            if p == 0 {
                value = v;
            }
        }

        value
    }

    /// Run one iteration of Chance Sampling MCCFR with CFR+ (clamp negative regrets).
    pub fn iterate_chance_sampling_plus<G: Game>(&mut self, game: &G) -> f64 {
        let value = self.iterate_chance_sampling(game);

        for node in self.nodes.values_mut() {
            let mut changed = false;
            for r in &mut node.cumulative_regret {
                if *r < 0.0 {
                    *r = 0.0;
                    changed = true;
                }
            }
            if changed {
                node.invalidate_cache();
            }
        }

        value
    }

    /// Core Chance Sampling MCCFR traversal.
    /// At chance nodes, samples one outcome instead of enumerating all.
    fn cfr_chance_sampling<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach_probs: &[f64],
        traversing_player: usize,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.payoff(state, traversing_player);
        }

        // Chance node: sample one outcome
        if game.is_chance_node(state) {
            let (next_state, _prob) = game.sample_chance_outcome(state, &mut self.rng);
            // Importance sampling: don't multiply/divide by prob (unbiased estimator)
            return self.cfr_chance_sampling(game, &next_state, reach_probs, traversing_player);
        }

        let current_player = game.current_player(state);
        let actions = game.actions(state);
        let num_actions = actions.len();
        let info_key = game.info_set_key(state, current_player);

        if !self.nodes.contains_key(&info_key) {
            self.nodes.insert(info_key.clone(), InfoSetNode::new(num_actions));
        }

        let node = self.nodes.get_mut(&info_key).unwrap();
        let strategy = node.current_strategy();
        node.accumulate_strategy_with(&strategy, reach_probs[current_player]);

        let mut action_values = vec![0.0; num_actions];
        let mut node_value = 0.0;

        for (i, action) in actions.iter().enumerate() {
            let next_state = game.apply_action(state, *action);
            let mut new_reach = reach_probs.to_vec();
            new_reach[current_player] *= strategy[i];
            action_values[i] = self.cfr_chance_sampling(game, &next_state, &new_reach, traversing_player);
            node_value += strategy[i] * action_values[i];
        }

        if current_player == traversing_player {
            let mut opponent_reach = 1.0;
            for (p, &r) in reach_probs.iter().enumerate() {
                if p != current_player {
                    opponent_reach *= r;
                }
            }

            let node = self.nodes.get_mut(&info_key).unwrap();
            for i in 0..num_actions {
                let regret = opponent_reach * (action_values[i] - node_value);
                node.cumulative_regret[i] += regret;
            }
            node.invalidate_cache();
        }

        node_value
    }

    /// Get the average strategy for all information sets.
    pub fn get_average_strategies(&self) -> FxHashMap<InfoSetKey, Vec<f64>> {
        self.nodes
            .iter()
            .map(|(key, node)| (key.clone(), node.average_strategy()))
            .collect()
    }

    /// Compute exploitability: how much an optimal opponent can gain.
    /// For a 2-player zero-sum game, exploitability = (v1 + v2) / 2
    /// where v_i is the best-response value for player i against the average strategy.
    /// Uses rayon to compute best-response values for each player in parallel.
    pub fn exploitability<G: Game + Sync>(&self, game: &G) -> f64
    where
        G::State: Send,
    {
        let num_players = game.num_players();
        let total: f64 = (0..num_players)
            .into_par_iter()
            .map(|player| self.best_response_value(game, player))
            .sum();
        total / num_players as f64
    }

    /// Compute the best-response value for `br_player` against
    /// the current average strategy of the opponent(s).
    ///
    /// Uses an iterative approach: repeatedly compute optimal actions at each
    /// info set (bottom-up), until the BR strategy stabilizes.
    fn best_response_value<G: Game>(&self, game: &G, br_player: usize) -> f64 {
        let initial_state = game.initial_state();
        let mut br_strategy: FxHashMap<String, usize> = FxHashMap::default();

        // Pre-compute and cache all average strategies to avoid repeated computation
        let avg_strategies: FxHashMap<&InfoSetKey, Vec<f64>> = self
            .nodes
            .iter()
            .map(|(key, node)| (key, node.average_strategy()))
            .collect();

        // Iteratively refine BR strategy (converges in depth-of-game iterations)
        for _ in 0..20 {
            let mut action_values: FxHashMap<String, Vec<f64>> = FxHashMap::default();
            self.collect_br_action_values(
                game,
                &initial_state,
                br_player,
                1.0,
                &br_strategy,
                &avg_strategies,
                &mut action_values,
            );

            let mut changed = false;
            for (key, values) in &action_values {
                let best = values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                if br_strategy.get(key).copied() != Some(best) {
                    changed = true;
                    br_strategy.insert(key.clone(), best);
                }
            }
            if !changed {
                break;
            }
        }

        self.eval_with_br_strategy(game, &initial_state, br_player, &br_strategy, &avg_strategies)
    }

    /// Traverse the game tree collecting weighted action values for each
    /// of `br_player`'s information sets.
    fn collect_br_action_values<G: Game>(
        &self,
        game: &G,
        state: &G::State,
        br_player: usize,
        weight: f64,
        br_strategy: &FxHashMap<String, usize>,
        avg_strategies: &FxHashMap<&InfoSetKey, Vec<f64>>,
        action_values: &mut FxHashMap<String, Vec<f64>>,
    ) {
        if game.is_terminal(state) {
            return;
        }

        if game.is_chance_node(state) {
            for (next, prob) in game.chance_outcomes(state) {
                self.collect_br_action_values(
                    game,
                    &next,
                    br_player,
                    weight * prob,
                    br_strategy,
                    avg_strategies,
                    action_values,
                );
            }
            return;
        }

        let player = game.current_player(state);
        let actions = game.actions(state);

        if player != br_player {
            let info_key = game.info_set_key(state, player);
            let strategy = avg_strategies
                .get(&info_key)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / actions.len() as f64; actions.len()]);
            for (i, action) in actions.iter().enumerate() {
                let next = game.apply_action(state, *action);
                self.collect_br_action_values(
                    game,
                    &next,
                    br_player,
                    weight * strategy[i],
                    br_strategy,
                    avg_strategies,
                    action_values,
                );
            }
            return;
        }

        // BR player's decision node
        let info_key = game.info_set_key(state, br_player);
        let num_actions = actions.len();
        let entry = action_values
            .entry(info_key.clone())
            .or_insert_with(|| vec![0.0; num_actions]);

        // Compute value of each action's subtree using current BR strategy for deeper decisions
        for (i, action) in actions.iter().enumerate() {
            let next = game.apply_action(state, *action);
            let val = self.eval_with_br_strategy(game, &next, br_player, br_strategy, avg_strategies);
            entry[i] += weight * val;
        }

        // Recurse into the chosen action's subtree to collect values for deeper info sets
        if let Some(&chosen) = br_strategy.get(&info_key) {
            let next = game.apply_action(state, actions[chosen]);
            self.collect_br_action_values(
                game,
                &next,
                br_player,
                weight,
                br_strategy,
                avg_strategies,
                action_values,
            );
        } else {
            // No strategy yet: recurse into all subtrees with uniform weight
            let w = weight / num_actions as f64;
            for action in &actions {
                let next = game.apply_action(state, *action);
                self.collect_br_action_values(
                    game, &next, br_player, w, br_strategy, avg_strategies, action_values,
                );
            }
        }
    }

    /// Evaluate the game value using a fixed BR strategy for the BR player,
    /// and the cached average strategy for the opponent.
    fn eval_with_br_strategy<G: Game>(
        &self,
        game: &G,
        state: &G::State,
        br_player: usize,
        br_strategy: &FxHashMap<String, usize>,
        avg_strategies: &FxHashMap<&InfoSetKey, Vec<f64>>,
    ) -> f64 {
        if game.is_terminal(state) {
            return game.payoff(state, br_player);
        }

        if game.is_chance_node(state) {
            return game
                .chance_outcomes(state)
                .iter()
                .map(|(next, prob)| {
                    prob * self.eval_with_br_strategy(game, next, br_player, br_strategy, avg_strategies)
                })
                .sum();
        }

        let player = game.current_player(state);
        let actions = game.actions(state);

        if player != br_player {
            let info_key = game.info_set_key(state, player);
            let strategy = avg_strategies
                .get(&info_key)
                .cloned()
                .unwrap_or_else(|| vec![1.0 / actions.len() as f64; actions.len()]);
            actions
                .iter()
                .enumerate()
                .map(|(i, action)| {
                    let next = game.apply_action(state, *action);
                    strategy[i] * self.eval_with_br_strategy(game, &next, br_player, br_strategy, avg_strategies)
                })
                .sum()
        } else {
            let info_key = game.info_set_key(state, br_player);
            if let Some(&best_idx) = br_strategy.get(&info_key) {
                let next = game.apply_action(state, actions[best_idx]);
                self.eval_with_br_strategy(game, &next, br_player, br_strategy, avg_strategies)
            } else {
                // No strategy determined yet: take max (optimistic upper bound)
                actions
                    .iter()
                    .map(|action| {
                        let next = game.apply_action(state, *action);
                        self.eval_with_br_strategy(game, &next, br_player, br_strategy, avg_strategies)
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            }
        }
    }
}

impl Default for CfrSolver {
    fn default() -> Self {
        Self::new()
    }
}
