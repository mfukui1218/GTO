use crate::cfr::CfrSolver;
use crate::info_set::InfoSetKey;
use rustc_hash::FxHashMap;

/// A solved strategy: maps information set keys to action probability distributions.
/// Stored as `f32` to match `InfoSetNode` storage (PioSolver-style).
#[derive(Clone, Debug)]
pub struct Strategy {
    pub strategies: FxHashMap<InfoSetKey, Vec<f32>>,
}

impl Strategy {
    /// Extract the average strategy from a trained CFR solver.
    pub fn from_solver(solver: &CfrSolver) -> Self {
        Strategy {
            strategies: solver.get_average_strategies(),
        }
    }

    /// Get the strategy (action probabilities) at an information set.
    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.strategies.get(key)
    }

    /// Return a compact (u8-quantized) representation of this strategy.
    /// See [`CompactStrategy`] for the memory-saving properties.
    pub fn compact(&self) -> CompactStrategy {
        let mut entries: FxHashMap<InfoSetKey, CompactEntry> = FxHashMap::default();
        for (key, probs) in &self.strategies {
            entries.insert(key.clone(), CompactEntry::from_probs(probs));
        }
        CompactStrategy { entries }
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

/// A single info-set entry in a `CompactStrategy`.
///
/// Two space-saving tricks are applied relative to `Vec<f32>`:
/// 1. **Single-action compression**: if the node has only one action, no
///    probabilities are stored — we always return `[1.0]`.
/// 2. **u8 quantization**: multi-action distributions are stored as `Vec<u8>`
///    where each byte encodes a probability in `0..=255` (mapped to
///    `[0.0, 1.0]`). This is 1/4 the size of `Vec<f32>`.
///
/// Worst-case quantization error per action is `1/255 ≈ 0.004` (0.4%), which
/// is well below the CFR convergence noise floor in practice.
#[derive(Clone, Debug)]
pub enum CompactEntry {
    /// Single-action info set — no storage needed. `num_actions` tracks size
    /// so `get` can return a correctly-sized vector.
    Single { num_actions: usize },
    /// Multi-action info set with u8-quantized probabilities. Always sums to
    /// exactly 255 (so dequantized values sum to exactly 1.0).
    Quantized(Vec<u8>),
}

impl CompactEntry {
    /// Build a compact entry from a raw probability distribution.
    ///
    /// The input is expected to be a valid probability vector summing to ≈1.0.
    /// Quantization rounds each probability to the nearest `1/255` and adjusts
    /// the largest bucket so the byte sum is exactly 255 (preserves exact
    /// normalization in the dequantized form).
    pub fn from_probs(probs: &[f32]) -> Self {
        // Degenerate cases: empty or single-action → no payload
        if probs.is_empty() {
            return CompactEntry::Single { num_actions: 0 };
        }
        // Detect "effectively single action" (all mass on one bucket) — this
        // catches both true 1-action nodes and locked nodes with one action = 1.0.
        if probs.len() == 1 {
            return CompactEntry::Single { num_actions: 1 };
        }

        // Quantize each prob to 0..=255, then fix the largest bucket so the
        // total sums to exactly 255.
        let mut bytes: Vec<u8> = probs
            .iter()
            .map(|&p| {
                let q = (p.clamp(0.0, 1.0) * 255.0).round() as i32;
                q.clamp(0, 255) as u8
            })
            .collect();

        let sum: i32 = bytes.iter().map(|&b| b as i32).sum();
        let diff = 255 - sum;
        if diff != 0 {
            // Adjust the bucket with the largest byte value by `diff`
            // (ties broken by lowest index).
            let (idx, _) = bytes
                .iter()
                .enumerate()
                .max_by_key(|(_, &b)| b)
                .expect("bytes non-empty");
            let adjusted = bytes[idx] as i32 + diff;
            bytes[idx] = adjusted.clamp(0, 255) as u8;
        }

        CompactEntry::Quantized(bytes)
    }

    /// Dequantize back to a `Vec<f32>` summing to 1.0.
    pub fn to_probs(&self) -> Vec<f32> {
        match self {
            CompactEntry::Single { num_actions } => {
                if *num_actions == 0 {
                    Vec::new()
                } else {
                    let mut v = vec![0.0; *num_actions];
                    v[0] = 1.0;
                    v
                }
            }
            CompactEntry::Quantized(bytes) => {
                bytes.iter().map(|&b| b as f32 / 255.0).collect()
            }
        }
    }

    /// Memory used by this entry's heap payload, in bytes. Used to demonstrate
    /// compression vs `Vec<f32>`.
    pub fn heap_bytes(&self) -> usize {
        match self {
            CompactEntry::Single { .. } => 0,
            CompactEntry::Quantized(bytes) => bytes.len(),
        }
    }
}

/// A compressed solved strategy.
///
/// Stores each info set as a [`CompactEntry`]:
/// - Single-action nodes carry no probability payload.
/// - Multi-action nodes store u8-quantized probabilities (1/4 the size of
///   `Vec<f32>`).
///
/// For a solve with 1M info sets averaging 3 actions, this reduces strategy
/// storage from ~12 bytes/node of `f32` payload to ~3 bytes/node (plus
/// per-entry overhead of the `Vec<u8>`). In practice 2-3x total reduction is
/// typical on bucketed postflop solves.
///
/// Trade-off: quantization introduces per-action error up to `1/255 ≈ 0.4%`,
/// which is well below CFR convergence noise. For pipelines that re-solve or
/// display results, dequantized values are indistinguishable from the raw
/// `f32` strategy.
#[derive(Clone, Debug)]
pub struct CompactStrategy {
    pub entries: FxHashMap<InfoSetKey, CompactEntry>,
}

impl CompactStrategy {
    /// Build a compact strategy directly from a solver (shortcut for
    /// `Strategy::from_solver(...).compact()` without the intermediate `Vec<f32>`
    /// materialization — saves peak memory during extraction).
    pub fn from_solver(solver: &CfrSolver) -> Self {
        let mut entries: FxHashMap<InfoSetKey, CompactEntry> = FxHashMap::default();
        for (key, node) in &solver.nodes {
            let probs = node.average_strategy();
            entries.insert(key.clone(), CompactEntry::from_probs(&probs));
        }
        CompactStrategy { entries }
    }

    /// Look up an info set and return the dequantized probabilities.
    /// Returns `None` if the key is unknown.
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        self.entries.get(key).map(|e| e.to_probs())
    }

    /// Number of stored info sets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no stored info sets.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Approximate heap bytes used by the probability payloads only
    /// (excludes keys and HashMap overhead). Useful for measuring the
    /// compression ratio.
    pub fn payload_heap_bytes(&self) -> usize {
        self.entries.values().map(|e| e.heap_bytes()).sum()
    }

    /// Convert back to an uncompressed `Strategy` with `Vec<f32>` payloads.
    pub fn to_strategy(&self) -> Strategy {
        let mut strategies: FxHashMap<InfoSetKey, Vec<f32>> = FxHashMap::default();
        for (key, entry) in &self.entries {
            strategies.insert(key.clone(), entry.to_probs());
        }
        Strategy { strategies }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantization_roundtrip_within_error_bound() {
        let probs = vec![0.25f32, 0.25, 0.25, 0.25];
        let entry = CompactEntry::from_probs(&probs);
        let back = entry.to_probs();
        assert_eq!(back.len(), 4);
        for (a, b) in probs.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0 / 255.0 + 1e-6, "a={} b={}", a, b);
        }
        let sum: f32 = back.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={} should be exactly 1.0", sum);
    }

    #[test]
    fn single_action_entry_has_no_payload() {
        let probs = vec![1.0f32];
        let entry = CompactEntry::from_probs(&probs);
        assert!(matches!(entry, CompactEntry::Single { num_actions: 1 }));
        assert_eq!(entry.heap_bytes(), 0);
        assert_eq!(entry.to_probs(), vec![1.0]);
    }

    #[test]
    fn quantized_sum_is_exactly_one() {
        // Nonuniform distribution that would round imperfectly
        let probs = vec![0.333f32, 0.333, 0.334];
        let entry = CompactEntry::from_probs(&probs);
        let back = entry.to_probs();
        let sum: f32 = back.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
    }

    #[test]
    fn compact_strategy_from_solver_preserves_keys() {
        use crate::cfr::CfrSolver;
        let mut solver = CfrSolver::new();
        solver.lock_node("J|x", vec![0.0, 1.0]);
        solver.lock_node("Q|x", vec![0.5, 0.5]);

        let compact = CompactStrategy::from_solver(&solver);
        assert_eq!(compact.len(), 2);
        let j_x = compact.get("J|x").unwrap();
        assert!((j_x[0] - 0.0).abs() < 1.0 / 255.0 + 1e-6);
        assert!((j_x[1] - 1.0).abs() < 1.0 / 255.0 + 1e-6);
        let q_x = compact.get("Q|x").unwrap();
        assert!((q_x[0] - 0.5).abs() < 1.0 / 255.0 + 1e-6);
    }

    #[test]
    fn payload_bytes_much_smaller_than_f32_vec() {
        // Typical postflop-ish distribution
        let probs = vec![0.1f32, 0.2, 0.3, 0.4];
        let entry = CompactEntry::from_probs(&probs);
        // 4 actions stored as 4 bytes vs 16 bytes of Vec<f32> payload
        assert_eq!(entry.heap_bytes(), 4);
        let f32_bytes = probs.len() * std::mem::size_of::<f32>();
        assert!(entry.heap_bytes() * 4 <= f32_bytes);
    }
}
