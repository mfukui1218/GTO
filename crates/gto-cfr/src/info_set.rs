use serde::{Deserialize, Serialize};

/// Key identifying an information set (what a player knows).
/// Encoded as an immutable, heap-allocated string: e.g. "J|cb" = holding Jack,
/// opponent checked then we bet.
///
/// `Box<str>` is used instead of `String` to save 8 bytes per entry (no capacity
/// field) — meaningful when millions of info sets are stored.
pub type InfoSetKey = Box<str>;

/// Data stored at each information set node.
/// Tracks cumulative regrets and cumulative strategy for regret matching.
///
/// Stored in `f32` (PioSolver-style) to halve memory footprint vs `f64`.
/// CFR computations are not sensitive to the extra precision in practice.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfoSetNode {
    /// Number of actions available at this node.
    pub num_actions: usize,
    /// Cumulative counterfactual regret for each action.
    pub cumulative_regret: Vec<f32>,
    /// Cumulative strategy (weighted by reach probability) for each action.
    pub cumulative_strategy: Vec<f32>,
    /// Cached current strategy. Invalidated when regrets change.
    #[serde(skip)]
    cached_strategy: Option<Vec<f32>>,
    /// If set, this node's strategy is locked (node locking feature).
    /// `current_strategy()` and `average_strategy()` return this value,
    /// and CFR skips regret updates for this node.
    #[serde(skip)]
    pub locked_strategy: Option<Vec<f32>>,
}

impl InfoSetNode {
    pub fn new(num_actions: usize) -> Self {
        InfoSetNode {
            num_actions,
            cumulative_regret: vec![0.0; num_actions],
            cumulative_strategy: vec![0.0; num_actions],
            cached_strategy: None,
            locked_strategy: None,
        }
    }

    /// Is this node's strategy locked to a fixed distribution?
    pub fn is_locked(&self) -> bool {
        self.locked_strategy.is_some()
    }

    /// Compute current strategy via regret matching.
    /// Uses cached result if available; recomputes and caches otherwise.
    /// If locked, returns the locked strategy.
    pub fn current_strategy(&mut self) -> Vec<f32> {
        if let Some(ref locked) = self.locked_strategy {
            return locked.clone();
        }
        if let Some(ref cached) = self.cached_strategy {
            return cached.clone();
        }

        let strategy = self.compute_strategy();
        self.cached_strategy = Some(strategy.clone());
        strategy
    }

    /// Compute current strategy without caching (for read-only access).
    /// If locked, returns the locked strategy.
    pub fn current_strategy_readonly(&self) -> Vec<f32> {
        if let Some(ref locked) = self.locked_strategy {
            return locked.clone();
        }
        if let Some(ref cached) = self.cached_strategy {
            return cached.clone();
        }
        self.compute_strategy()
    }

    /// Internal strategy computation via regret matching.
    fn compute_strategy(&self) -> Vec<f32> {
        let mut strategy = vec![0.0f32; self.num_actions];
        let mut normalizing_sum = 0.0f32;

        for i in 0..self.num_actions {
            let positive_regret = self.cumulative_regret[i].max(0.0);
            strategy[i] = positive_regret;
            normalizing_sum += positive_regret;
        }

        if normalizing_sum > 0.0 {
            for s in &mut strategy {
                *s /= normalizing_sum;
            }
        } else {
            let uniform = 1.0 / self.num_actions as f32;
            for s in &mut strategy {
                *s = uniform;
            }
        }

        strategy
    }

    /// Get the average strategy (Nash equilibrium approximation).
    /// This is the cumulative strategy normalized to a probability distribution.
    /// If locked, returns the locked strategy directly.
    pub fn average_strategy(&self) -> Vec<f32> {
        if let Some(ref locked) = self.locked_strategy {
            return locked.clone();
        }
        let mut avg = vec![0.0f32; self.num_actions];
        let sum: f32 = self.cumulative_strategy.iter().sum();

        if sum > 0.0 {
            for i in 0..self.num_actions {
                avg[i] = self.cumulative_strategy[i] / sum;
            }
        } else {
            let uniform = 1.0 / self.num_actions as f32;
            for a in &mut avg {
                *a = uniform;
            }
        }

        avg
    }

    /// Update cumulative strategy with a pre-computed strategy weighted by reach probability.
    /// Avoids recomputing current_strategy() when the caller already has it.
    pub fn accumulate_strategy_with(&mut self, strategy: &[f32], reach_prob: f32) {
        for i in 0..self.num_actions {
            self.cumulative_strategy[i] += reach_prob * strategy[i];
        }
    }

    /// Update cumulative strategy with current strategy weighted by reach probability.
    pub fn accumulate_strategy(&mut self, reach_prob: f32) {
        let strategy = self.current_strategy();
        self.accumulate_strategy_with(&strategy, reach_prob);
    }

    /// Invalidate the cached strategy. Must be called after modifying cumulative_regret.
    pub fn invalidate_cache(&mut self) {
        self.cached_strategy = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regret_matching_uniform() {
        let mut node = InfoSetNode::new(3);
        let s = node.current_strategy();
        assert_eq!(s.len(), 3);
        for &p in &s {
            assert!((p - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn regret_matching_positive() {
        let mut node = InfoSetNode::new(3);
        node.cumulative_regret = vec![10.0, 5.0, 0.0];
        let s = node.current_strategy();
        assert!((s[0] - 10.0 / 15.0).abs() < 1e-6);
        assert!((s[1] - 5.0 / 15.0).abs() < 1e-6);
        assert!((s[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn regret_matching_negative() {
        let mut node = InfoSetNode::new(2);
        node.cumulative_regret = vec![-5.0, -3.0];
        let s = node.current_strategy();
        assert!((s[0] - 0.5).abs() < 1e-6);
        assert!((s[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn average_strategy_computation() {
        let mut node = InfoSetNode::new(2);
        node.cumulative_strategy = vec![75.0, 25.0];
        let avg = node.average_strategy();
        assert!((avg[0] - 0.75).abs() < 1e-6);
        assert!((avg[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn strategy_sums_to_one() {
        let mut node = InfoSetNode::new(4);
        node.cumulative_regret = vec![1.0, 2.0, 3.0, -1.0];
        let s = node.current_strategy();
        let sum: f32 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
