use serde::{Deserialize, Serialize};

/// Key identifying an information set (what a player knows).
/// Encoded as a string: e.g. "J|cb" = holding Jack, opponent checked then we bet.
pub type InfoSetKey = String;

/// Data stored at each information set node.
/// Tracks cumulative regrets and cumulative strategy for regret matching.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InfoSetNode {
    /// Number of actions available at this node.
    pub num_actions: usize,
    /// Cumulative counterfactual regret for each action.
    pub cumulative_regret: Vec<f64>,
    /// Cumulative strategy (weighted by reach probability) for each action.
    pub cumulative_strategy: Vec<f64>,
    /// Cached current strategy. Invalidated when regrets change.
    #[serde(skip)]
    cached_strategy: Option<Vec<f64>>,
}

impl InfoSetNode {
    pub fn new(num_actions: usize) -> Self {
        InfoSetNode {
            num_actions,
            cumulative_regret: vec![0.0; num_actions],
            cumulative_strategy: vec![0.0; num_actions],
            cached_strategy: None,
        }
    }

    /// Compute current strategy via regret matching.
    /// Uses cached result if available; recomputes and caches otherwise.
    pub fn current_strategy(&mut self) -> Vec<f64> {
        if let Some(ref cached) = self.cached_strategy {
            return cached.clone();
        }

        let strategy = self.compute_strategy();
        self.cached_strategy = Some(strategy.clone());
        strategy
    }

    /// Compute current strategy without caching (for read-only access).
    pub fn current_strategy_readonly(&self) -> Vec<f64> {
        if let Some(ref cached) = self.cached_strategy {
            return cached.clone();
        }
        self.compute_strategy()
    }

    /// Internal strategy computation via regret matching.
    fn compute_strategy(&self) -> Vec<f64> {
        let mut strategy = vec![0.0; self.num_actions];
        let mut normalizing_sum = 0.0;

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
            let uniform = 1.0 / self.num_actions as f64;
            for s in &mut strategy {
                *s = uniform;
            }
        }

        strategy
    }

    /// Get the average strategy (Nash equilibrium approximation).
    /// This is the cumulative strategy normalized to a probability distribution.
    pub fn average_strategy(&self) -> Vec<f64> {
        let mut avg = vec![0.0; self.num_actions];
        let sum: f64 = self.cumulative_strategy.iter().sum();

        if sum > 0.0 {
            for i in 0..self.num_actions {
                avg[i] = self.cumulative_strategy[i] / sum;
            }
        } else {
            let uniform = 1.0 / self.num_actions as f64;
            for a in &mut avg {
                *a = uniform;
            }
        }

        avg
    }

    /// Update cumulative strategy with a pre-computed strategy weighted by reach probability.
    /// Avoids recomputing current_strategy() when the caller already has it.
    pub fn accumulate_strategy_with(&mut self, strategy: &[f64], reach_prob: f64) {
        for i in 0..self.num_actions {
            self.cumulative_strategy[i] += reach_prob * strategy[i];
        }
    }

    /// Update cumulative strategy with current strategy weighted by reach probability.
    pub fn accumulate_strategy(&mut self, reach_prob: f64) {
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
            assert!((p - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn regret_matching_positive() {
        let mut node = InfoSetNode::new(3);
        node.cumulative_regret = vec![10.0, 5.0, 0.0];
        let s = node.current_strategy();
        assert!((s[0] - 10.0 / 15.0).abs() < 1e-10);
        assert!((s[1] - 5.0 / 15.0).abs() < 1e-10);
        assert!((s[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn regret_matching_negative() {
        let mut node = InfoSetNode::new(2);
        node.cumulative_regret = vec![-5.0, -3.0];
        let s = node.current_strategy();
        assert!((s[0] - 0.5).abs() < 1e-10);
        assert!((s[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn average_strategy_computation() {
        let mut node = InfoSetNode::new(2);
        node.cumulative_strategy = vec![75.0, 25.0];
        let avg = node.average_strategy();
        assert!((avg[0] - 0.75).abs() < 1e-10);
        assert!((avg[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn strategy_sums_to_one() {
        let mut node = InfoSetNode::new(4);
        node.cumulative_regret = vec![1.0, 2.0, 3.0, -1.0];
        let s = node.current_strategy();
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
