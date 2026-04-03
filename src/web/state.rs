use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::Json;
use gto_games::push_fold::PushFoldData;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

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
        pct: f64, // 0.0–100.0, negative if indeterminate
        current_step: usize,
        total_steps: usize,
    },
    Completed {
        result: serde_json::Value,
    },
    Failed {
        error: String,
    },
}

/// Unified API error type for all handlers.
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
        }
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

impl From<String> for ApiError {
    fn from(msg: String) -> Self {
        ApiError::bad_request(msg)
    }
}

impl<T> From<PoisonError<MutexGuard<'_, T>>> for ApiError {
    fn from(e: PoisonError<MutexGuard<'_, T>>) -> Self {
        ApiError::internal(format!("Lock error: {}", e))
    }
}
