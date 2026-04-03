use crate::cfr::CfrSolver;
use crate::info_set::InfoSetKey;
use rustc_hash::FxHashMap;

/// A solved strategy: maps information set keys to action probability distributions.
#[derive(Clone, Debug)]
pub struct Strategy {
    pub strategies: FxHashMap<InfoSetKey, Vec<f64>>,
}

impl Strategy {
    /// Extract the average strategy from a trained CFR solver.
    pub fn from_solver(solver: &CfrSolver) -> Self {
        Strategy {
            strategies: solver.get_average_strategies(),
        }
    }

    /// Get the strategy (action probabilities) at an information set.
    pub fn get(&self, key: &str) -> Option<&Vec<f64>> {
        self.strategies.get(key)
    }

    /// Print all strategies in a readable format.
    pub fn print_with_actions(&self, action_names: &[&str]) {
        let mut keys: Vec<_> = self.strategies.keys().collect();
        keys.sort();

        for key in keys {
            let probs = &self.strategies[key];
            print!("  {:<12}", key);
            for (i, &p) in probs.iter().enumerate() {
                if i < action_names.len() {
                    print!("  {:>5}: {:.4}", action_names[i], p);
                } else {
                    print!("  a{}: {:.4}", i, p);
                }
            }
            println!();
        }
    }
}
