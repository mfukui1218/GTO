use axum::routing::{get, post};
use axum::Router;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use super::handlers;
use super::state::AppState;

pub async fn run_server(port: u16) {
    let state = AppState {
        push_fold_cache: Arc::new(Mutex::new(None)),
        jobs: Arc::new(Mutex::new(HashMap::new())),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(handlers::index_html))
        .route("/api/kuhn", post(handlers::solve_kuhn))
        .route("/api/leduc", post(handlers::solve_leduc))
        .route("/api/equity", post(handlers::calc_equity))
        .route("/api/pushfold/start", post(handlers::start_pushfold))
        .route("/api/pushfold/status/{job_id}", get(handlers::pushfold_status))
        .route("/api/preflop/start", post(handlers::start_preflop))
        .route("/api/preflop/status/{job_id}", get(handlers::preflop_status))
        .route("/api/range-equity", post(handlers::calc_range_equity))
        .route("/api/range-equity/status/{job_id}", get(handlers::range_equity_status))
        .route("/api/postflop-review", post(handlers::calc_postflop_review))
        .route("/api/postflop-solve/start", post(handlers::start_postflop_solve))
        .route("/api/postflop-solve/status/{job_id}", get(handlers::postflop_solve_status))
        .route("/api/range-presets", get(handlers::get_range_presets))
        .layer(cors)
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("GTO Solver Web UI running at http://localhost:{}", port);
    axum::serve(listener, app).await.unwrap();
}
