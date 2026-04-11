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
///
/// Internal storage uses `f32` (PioSolver-style) to halve memory footprint.
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

    /// Lock an information set's strategy to a fixed distribution.
    /// The CFR traversal will use this strategy instead of regret-matching
    /// and will not update cumulative regrets or cumulative strategy for this node.
    /// `average_strategy()` for this node returns the locked distribution.
    ///
    /// If the node does not yet exist, it is pre-created with the appropriate
    /// number of actions. If it exists with a mismatched action count, the call panics.
    ///
    /// The provided strategy is normalized and clamped to [0, 1] for safety.
    pub fn lock_node(&mut self, key: &str, strategy: Vec<f32>) {
        let num_actions = strategy.len();
        assert!(num_actions > 0, "locked strategy must have at least one action");

        // Normalize + clamp to a valid probability distribution.
        let mut s = strategy;
        for x in &mut s {
            if !x.is_finite() || *x < 0.0 {
                *x = 0.0;
            }
        }
        let sum: f32 = s.iter().sum();
        if sum > 0.0 {
            for x in &mut s {
                *x /= sum;
            }
        } else {
            let u = 1.0 / num_actions as f32;
            for x in &mut s {
                *x = u;
            }
        }

        if let Some(node) = self.nodes.get_mut(key) {
            assert_eq!(
                node.num_actions, num_actions,
                "lock_node: num_actions mismatch for {} (node has {}, strategy has {})",
                key, node.num_actions, num_actions
            );
            node.locked_strategy = Some(s);
            node.invalidate_cache();
        } else {
            let mut node = InfoSetNode::new(num_actions);
            node.locked_strategy = Some(s);
            self.nodes.insert(key.into(), node);
        }
    }

    /// Remove the lock on an information set, allowing CFR to resume regret updates.
    /// Cumulative regrets/strategies accumulated before the lock remain intact.
    pub fn unlock_node(&mut self, key: &str) {
        if let Some(node) = self.nodes.get_mut(key) {
            node.locked_strategy = None;
            node.invalidate_cache();
        }
    }

    /// Is this information set currently locked?
    pub fn is_locked(&self, key: &str) -> bool {
        self.nodes.get(key).map(|n| n.is_locked()).unwrap_or(false)
    }

    /// Lock multiple information sets at once. Convenience for API consumers.
    pub fn lock_nodes<I, K>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (K, Vec<f32>)>,
        K: AsRef<str>,
    {
        for (key, strategy) in entries {
            self.lock_node(key.as_ref(), strategy);
        }
    }

    /// Run one iteration of CFR for all players.
    /// Returns the expected game value for player 0.
    pub fn iterate<G: Game>(&mut self, game: &G) -> f32 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0;

        // Must traverse for each player so that all players' regrets get updated
        for p in 0..num_players {
            let reach_probs = vec![1.0f32; num_players];
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
        reach_probs: &[f32],
        traversing_player: usize,
    ) -> f32 {
        // Terminal node: return payoff
        if game.is_terminal(state) {
            return game.payoff(state, traversing_player) as f32;
        }

        // Chance node: enumerate all outcomes and compute expected value
        if game.is_chance_node(state) {
            let outcomes = game.chance_outcomes(state);
            let mut ev = 0.0f32;
            for (next_state, prob) in &outcomes {
                ev += (*prob as f32) * self.cfr(game, next_state, reach_probs, traversing_player);
            }
            return ev;
        }

        let current_player = game.current_player(state);
        let actions = game.actions(state);
        let num_actions = actions.len();
        // Box<str> is stored as the map key to save 8 bytes/entry vs String.
        let info_key: Box<str> = game.info_set_key(state, current_player).into_boxed_str();

        // Get or create node via the `entry` API — a single hash/lookup per
        // visit, instead of the `contains_key` + `get_mut` double lookup.
        let node = self
            .nodes
            .entry(info_key.clone())
            .or_insert_with(|| InfoSetNode::new(num_actions));
        let locked = node.is_locked();
        let strategy = node.current_strategy();

        // Accumulate strategy weighted by this player's reach probability.
        // Skip for locked nodes — average_strategy() returns the locked distribution directly.
        if !locked {
            node.accumulate_strategy_with(&strategy, reach_probs[current_player]);
        }

        // Compute counterfactual value for each action
        let mut action_values = vec![0.0f32; num_actions];
        let mut node_value = 0.0f32;

        for (i, action) in actions.iter().enumerate() {
            let next_state = game.apply_action(state, *action);

            // Update reach probabilities
            let mut new_reach = reach_probs.to_vec();
            new_reach[current_player] *= strategy[i];

            action_values[i] = self.cfr(game, &next_state, &new_reach, traversing_player);
            node_value += strategy[i] * action_values[i];
        }

        // Update regrets only for the traversing player's information sets.
        // Locked nodes do not receive regret updates.
        if current_player == traversing_player && !locked {
            // Counterfactual reach: product of all opponents' reach probabilities
            let mut opponent_reach = 1.0f32;
            for (p, &r) in reach_probs.iter().enumerate() {
                if p != current_player {
                    opponent_reach *= r;
                }
            }

            let node = self.nodes.get_mut(info_key.as_ref()).unwrap();
            for i in 0..num_actions {
                let regret = opponent_reach * (action_values[i] - node_value);
                node.cumulative_regret[i] += regret;
            }
            node.invalidate_cache();
        }

        node_value
    }

    /// Run CFR+ iteration: same as CFR but clamps negative regrets to zero.
    pub fn iterate_plus<G: Game>(&mut self, game: &G) -> f32 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0f32;

        for p in 0..num_players {
            let reach_probs = vec![1.0f32; num_players];
            let v = self.cfr_plus(game, &initial_state, &reach_probs, p);
            if p == 0 {
                value = v;
            }
        }

        // Clamp all negative regrets to 0 (CFR+ modification).
        // Locked nodes are skipped — their regrets are not updated anyway.
        for node in self.nodes.values_mut() {
            if node.is_locked() {
                continue;
            }
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
        reach_probs: &[f32],
        traversing_player: usize,
    ) -> f32 {
        // Reuse the same logic as vanilla CFR
        self.cfr(game, state, reach_probs, traversing_player)
    }

    /// Run one iteration of Discount CFR (Brown & Sandholm 2019).
    /// Aka "Linear CFR" for the special case alpha=beta=gamma=1.
    ///
    /// After a normal CFR traversal, applies discount factors to regrets and
    /// cumulative strategy so that the most recent iterations dominate:
    /// - positive regrets *= t^alpha / (t^alpha + 1)
    /// - negative regrets *= t^beta  / (t^beta  + 1)
    /// - cumulative strategy *= (t / (t + 1))^gamma
    ///
    /// PioSolver-recommended preset: alpha=1.5, beta=0.0, gamma=2.0.
    /// Pass `iteration` as 1-based iteration number (1, 2, ...).
    pub fn iterate_discount<G: Game>(
        &mut self,
        game: &G,
        iteration: usize,
        alpha: f32,
        beta: f32,
        gamma: f32,
    ) -> f32 {
        let value = self.iterate(game);
        self.apply_discount(iteration, alpha, beta, gamma);
        value
    }

    /// Apply Discount CFR weights to all stored regrets and cumulative strategies.
    fn apply_discount(&mut self, iteration: usize, alpha: f32, beta: f32, gamma: f32) {
        let t = iteration as f32;
        let pos_w = {
            let ta = t.powf(alpha);
            ta / (ta + 1.0)
        };
        let neg_w = {
            let tb = t.powf(beta);
            tb / (tb + 1.0)
        };
        let strat_w = (t / (t + 1.0)).powf(gamma);

        for node in self.nodes.values_mut() {
            // Locked nodes have no meaningful regrets or cumulative strategy; skip them.
            if node.is_locked() {
                continue;
            }
            let mut regret_changed = false;
            for r in &mut node.cumulative_regret {
                if *r > 0.0 {
                    *r *= pos_w;
                } else if *r < 0.0 {
                    *r *= neg_w;
                }
                regret_changed = true;
            }
            for s in &mut node.cumulative_strategy {
                *s *= strat_w;
            }
            if regret_changed {
                node.invalidate_cache();
            }
        }
    }

    /// Run one iteration of Chance Sampling MCCFR.
    /// Samples a single chance outcome instead of enumerating all.
    pub fn iterate_chance_sampling<G: Game>(&mut self, game: &G) -> f32 {
        let initial_state = game.initial_state();
        let num_players = game.num_players();
        let mut value = 0.0f32;

        for p in 0..num_players {
            let reach_probs = vec![1.0f32; num_players];
            let v = self.cfr_chance_sampling(game, &initial_state, &reach_probs, p);
            if p == 0 {
                value = v;
            }
        }

        value
    }

    /// Run one iteration of Chance Sampling MCCFR with CFR+ (clamp negative regrets).
    pub fn iterate_chance_sampling_plus<G: Game>(&mut self, game: &G) -> f32 {
        let value = self.iterate_chance_sampling(game);

        for node in self.nodes.values_mut() {
            if node.is_locked() {
                continue;
            }
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

    /// Discount CFR variant of chance sampling MCCFR.
    pub fn iterate_chance_sampling_discount<G: Game>(
        &mut self,
        game: &G,
        iteration: usize,
        alpha: f32,
        beta: f32,
        gamma: f32,
    ) -> f32 {
        let value = self.iterate_chance_sampling(game);
        self.apply_discount(iteration, alpha, beta, gamma);
        value
    }

    /// Core Chance Sampling MCCFR traversal.
    /// At chance nodes, samples one outcome instead of enumerating all.
    fn cfr_chance_sampling<G: Game>(
        &mut self,
        game: &G,
        state: &G::State,
        reach_probs: &[f32],
        traversing_player: usize,
    ) -> f32 {
        if game.is_terminal(state) {
            return game.payoff(state, traversing_player) as f32;
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
        let info_key: Box<str> = game.info_set_key(state, current_player).into_boxed_str();

        let node = self
            .nodes
            .entry(info_key.clone())
            .or_insert_with(|| InfoSetNode::new(num_actions));
        let locked = node.is_locked();
        let strategy = node.current_strategy();
        if !locked {
            node.accumulate_strategy_with(&strategy, reach_probs[current_player]);
        }

        let mut action_values = vec![0.0f32; num_actions];
        let mut node_value = 0.0f32;

        for (i, action) in actions.iter().enumerate() {
            let next_state = game.apply_action(state, *action);
            let mut new_reach = reach_probs.to_vec();
            new_reach[current_player] *= strategy[i];
            action_values[i] = self.cfr_chance_sampling(game, &next_state, &new_reach, traversing_player);
            node_value += strategy[i] * action_values[i];
        }

        if current_player == traversing_player && !locked {
            let mut opponent_reach = 1.0f32;
            for (p, &r) in reach_probs.iter().enumerate() {
                if p != current_player {
                    opponent_reach *= r;
                }
            }

            let node = self.nodes.get_mut(info_key.as_ref()).unwrap();
            for i in 0..num_actions {
                let regret = opponent_reach * (action_values[i] - node_value);
                node.cumulative_regret[i] += regret;
            }
            node.invalidate_cache();
        }

        node_value
    }

    /// Get the average strategy for all information sets.
    pub fn get_average_strategies(&self) -> FxHashMap<InfoSetKey, Vec<f32>> {
        self.nodes
            .iter()
            .map(|(key, node)| (key.clone(), node.average_strategy()))
            .collect()
    }

    /// Compute exploitability: how much an optimal opponent can gain.
    /// For a 2-player zero-sum game, exploitability = (v1 + v2) / 2
    /// where v_i is the best-response value for player i against the average strategy.
    /// Uses rayon to compute best-response values for each player in parallel.
    ///
    /// Best-response computation uses `f64` internally for numerical stability
    /// even though node storage is `f32`.
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
        let mut br_strategy: FxHashMap<Box<str>, usize> = FxHashMap::default();

        // Pre-compute and cache all average strategies (cast to f64 for BR math).
        // Keyed by `&str` (via the stored `Box<str>`) to avoid nested references.
        let avg_strategies: FxHashMap<&str, Vec<f64>> = self
            .nodes
            .iter()
            .map(|(key, node)| {
                let avg_f32 = node.average_strategy();
                let avg_f64: Vec<f64> = avg_f32.iter().map(|&x| x as f64).collect();
                (key.as_ref(), avg_f64)
            })
            .collect();

        // Iteratively refine BR strategy (converges in depth-of-game iterations)
        for _ in 0..20 {
            let mut action_values: FxHashMap<Box<str>, Vec<f64>> = FxHashMap::default();
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
                if br_strategy.get(key.as_ref()).copied() != Some(best) {
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
        br_strategy: &FxHashMap<Box<str>, usize>,
        avg_strategies: &FxHashMap<&str, Vec<f64>>,
        action_values: &mut FxHashMap<Box<str>, Vec<f64>>,
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
                .get(info_key.as_str())
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
        let info_key: Box<str> = game.info_set_key(state, br_player).into_boxed_str();
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
        if let Some(&chosen) = br_strategy.get(info_key.as_ref()) {
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
        br_strategy: &FxHashMap<Box<str>, usize>,
        avg_strategies: &FxHashMap<&str, Vec<f64>>,
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
                .get(info_key.as_str())
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
            if let Some(&best_idx) = br_strategy.get(info_key.as_str()) {
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
