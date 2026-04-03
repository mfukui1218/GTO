use gto_games::push_fold::PushFoldData;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AppState {
    pub push_fold_cache: Arc<Mutex<Option<PushFoldDataCache>>>,
    pub jobs: Arc<Mutex<HashMap<String, JobStatus>>>,
}

pub struct PushFoldDataCache {
    pub data: PushFoldData,
    pub samples: usize,
}

#[allow(dead_code)]
pub enum JobStatus {
    Running {
        progress: String,
        pct: f64,           // 0.0–100.0, negative if indeterminate
        current_step: usize,
        total_steps: usize,
    },
    Completed { result: serde_json::Value },
    Failed { error: String },
}
