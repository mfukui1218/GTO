use axum::extract::{Path, State};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use gto_cfr::{train, train_with_callback, Strategy, TrainerConfig};
use gto_core::Card;
use gto_eval::{
    board_texture, category_name, equity_exact, hand_class_name, hand_vs_range_equity,
    range_vs_range_equity, range_vs_range_monte_carlo, NUM_CLASSES,
};
use gto_eval::range_equity::{class_index_to_name as re_class_name, range_stats as re_range_stats};
use gto_games::preflop::{
    extract_preflop_strategies, Position, PreflopConfig, PreflopGame,
};
use gto_games::push_fold::{
    class_index_to_name, extract_call_range, extract_push_range, PushFoldData, PushFoldGame,
};
use gto_games::postflop::{
    extract_postflop_strategies, BetSizeConfig, PostflopConfig, PostflopGame,
};
use gto_games::{KuhnPoker, LeducHoldem};

use super::state::{AppState, JobStatus, PushFoldDataCache};

static INDEX_HTML: &str = include_str!("../../static/index.html");

pub async fn index_html() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/html; charset=utf-8")], INDEX_HTML)
}

// --- Card parsing ---

fn parse_card(s: &str) -> Result<Card, String> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return Err(format!("Invalid card: '{}'", s));
    }
    let rank = match bytes[0] {
        b'2' => 0,
        b'3' => 1,
        b'4' => 2,
        b'5' => 3,
        b'6' => 4,
        b'7' => 5,
        b'8' => 6,
        b'9' => 7,
        b'T' => 8,
        b'J' => 9,
        b'Q' => 10,
        b'K' => 11,
        b'A' => 12,
        _ => return Err(format!("Invalid rank in card: '{}'", s)),
    };
    let suit = match bytes[1] {
        b'c' => 0,
        b'd' => 1,
        b'h' => 2,
        b's' => 3,
        _ => return Err(format!("Invalid suit in card: '{}'", s)),
    };
    Ok(Card::new(rank, suit))
}

// --- Kuhn Poker ---

#[derive(Deserialize)]
pub struct KuhnRequest {
    pub iterations: Option<usize>,
    pub cfr_plus: Option<bool>,
}

pub async fn solve_kuhn(
    Json(req): Json<KuhnRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let result = tokio::task::spawn_blocking(move || {
        let game = KuhnPoker;
        let config = TrainerConfig {
            iterations: req.iterations.unwrap_or(100_000),
            use_cfr_plus: req.cfr_plus.unwrap_or(false),
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &config);
        let strategy = Strategy::from_solver(&solver);
        let exploit = solver.exploitability(&game);

        let cards = ["J", "Q", "K"];

        let get_strats = |keys: Vec<String>, actions: &[&str]| -> Vec<Value> {
            keys.into_iter()
                .filter_map(|key| {
                    strategy.get(&key).map(|probs| {
                        let action_map: Vec<Value> = actions
                            .iter()
                            .zip(probs.iter())
                            .map(|(a, p)| json!({"action": a, "prob": *p}))
                            .collect();
                        json!({"key": key, "probs": action_map})
                    })
                })
                .collect()
        };

        let p0_opening = get_strats(
            cards.iter().map(|c| c.to_string()).collect(),
            &["Check", "Bet"],
        );
        let p0_facing_bet = get_strats(
            cards.iter().map(|c| format!("{}|xb1", c)).collect(),
            &["Fold", "Call"],
        );
        let p1_after_check = get_strats(
            cards.iter().map(|c| format!("{}|x", c)).collect(),
            &["Check", "Bet"],
        );
        let p1_facing_bet = get_strats(
            cards.iter().map(|c| format!("{}|b1", c)).collect(),
            &["Fold", "Call"],
        );

        json!({
            "exploitability": exploit,
            "num_info_sets": solver.nodes.len(),
            "nash_value": -1.0 / 18.0,
            "strategies": {
                "p0_opening": p0_opening,
                "p0_facing_bet": p0_facing_bet,
                "p1_after_check": p1_after_check,
                "p1_facing_bet": p1_facing_bet,
            }
        })
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(result))
}

// --- Leduc Hold'em ---

#[derive(Deserialize)]
pub struct LeducRequest {
    pub iterations: Option<usize>,
    pub cfr_plus: Option<bool>,
}

pub async fn solve_leduc(
    Json(req): Json<LeducRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let result = tokio::task::spawn_blocking(move || {
        let game = LeducHoldem;
        let config = TrainerConfig {
            iterations: req.iterations.unwrap_or(100_000),
            use_cfr_plus: req.cfr_plus.unwrap_or(false),
            use_chance_sampling: false,
            print_interval: 0,
        };
        let solver = train(&game, &config);
        let strategy = Strategy::from_solver(&solver);
        let exploit = solver.exploitability(&game);

        let cards = ["J", "Q", "K"];

        let get_strats = |keys: Vec<String>, actions: &[&str]| -> Vec<Value> {
            keys.into_iter()
                .filter_map(|key| {
                    strategy.get(&key).map(|probs| {
                        let action_map: Vec<Value> = actions
                            .iter()
                            .zip(probs.iter())
                            .map(|(a, p)| json!({"action": a, "prob": *p}))
                            .collect();
                        json!({"key": key, "probs": action_map})
                    })
                })
                .collect()
        };

        // Preflop
        let opening = get_strats(
            cards.iter().map(|c| c.to_string()).collect(),
            &["Check", "Bet"],
        );
        let facing_open = get_strats(
            cards.iter().map(|c| format!("{}|x", c)).collect(),
            &["Check", "Bet"],
        );
        let facing_bet = get_strats(
            cards.iter().map(|c| format!("{}|b2", c)).collect(),
            &["Fold", "Call", "Raise"],
        );

        // Flop (after check-check)
        let mut flop = Vec::new();
        for board in &cards {
            let mut rows = Vec::new();
            for card in &cards {
                let key = format!("{}|xx:{}", card, board);
                if let Some(probs) = strategy.get(&key) {
                    let is_pair = card == board;
                    rows.push(json!({
                        "key": key,
                        "card": card,
                        "is_pair": is_pair,
                        "probs": [
                            {"action": "Check", "prob": probs[0]},
                            {"action": "Bet", "prob": probs[1]},
                        ]
                    }));
                }
            }
            flop.push(json!({"board": board, "rows": rows}));
        }

        json!({
            "exploitability": exploit,
            "num_info_sets": solver.nodes.len(),
            "preflop": {
                "opening": opening,
                "facing_open": facing_open,
                "facing_bet": facing_bet,
            },
            "flop": flop,
        })
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(result))
}

// --- Equity Calculator ---

#[derive(Deserialize)]
pub struct EquityRequest {
    pub hand1: [String; 2],
    pub hand2: [String; 2],
    #[serde(default)]
    pub board: Vec<String>,
}

pub async fn calc_equity(
    Json(req): Json<EquityRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let hand1_cards: Result<Vec<Card>, String> = req.hand1.iter().map(|s| parse_card(s)).collect();
    let hand2_cards: Result<Vec<Card>, String> = req.hand2.iter().map(|s| parse_card(s)).collect();
    let board_cards: Result<Vec<Card>, String> = req.board.iter().map(|s| parse_card(s)).collect();

    let hand1 = hand1_cards
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;
    let hand2 = hand2_cards
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;
    let board = board_cards
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    // Validate no duplicate cards
    let mut all_cards = vec![];
    all_cards.extend_from_slice(&hand1);
    all_cards.extend_from_slice(&hand2);
    all_cards.extend_from_slice(&board);
    for i in 0..all_cards.len() {
        for j in (i + 1)..all_cards.len() {
            if all_cards[i].0 == all_cards[j].0 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "Duplicate cards detected"})),
                ));
            }
        }
    }

    let h1 = [hand1[0], hand1[1]];
    let h2 = [hand2[0], hand2[1]];

    let result = tokio::task::spawn_blocking(move || {
        let name1 = hand_class_name(h1[0], h1[1]);
        let name2 = hand_class_name(h2[0], h2[1]);
        let eq = equity_exact(h1, h2, &board);

        let total = eq.total as f64;
        json!({
            "hand1_name": name1,
            "hand2_name": name2,
            "wins": eq.wins,
            "ties": eq.ties,
            "losses": eq.losses,
            "total": eq.total,
            "win_pct": eq.wins as f64 / total * 100.0,
            "tie_pct": eq.ties as f64 / total * 100.0,
            "lose_pct": eq.losses as f64 / total * 100.0,
            "equity": eq.equity() * 100.0,
        })
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(result))
}

// --- Push/Fold ---

#[derive(Deserialize)]
pub struct PushFoldRequest {
    #[serde(default = "default_stacks")]
    pub stacks: Vec<f64>,
    #[serde(default = "default_mc_samples")]
    pub mc_samples: usize,
    #[serde(default = "default_iterations")]
    pub iterations: usize,
}

fn default_stacks() -> Vec<f64> {
    vec![5.0, 10.0, 15.0, 20.0]
}
fn default_mc_samples() -> usize {
    2_000_000
}
fn default_iterations() -> usize {
    500_000
}

fn freq_to_grid(freqs: &[f64; NUM_CLASSES]) -> Vec<Vec<f64>> {
    let mut grid = vec![vec![0.0; 13]; 13];
    for row in (0..13).rev() {
        for col in (0..13).rev() {
            let class = if row == col {
                row
            } else if col > row {
                91 + (col * (col - 1) / 2 + row)
            } else {
                13 + (row * (row - 1) / 2 + col)
            };
            grid[12 - row][12 - col] = freqs[class];
        }
    }
    grid
}

fn range_stats(freqs: &[f64; NUM_CLASSES]) -> (f64, f64) {
    let combos = |class: usize| -> f64 {
        if class < 13 {
            6.0
        } else if class < 91 {
            4.0
        } else {
            12.0
        }
    };
    let total_combos: f64 = (0..NUM_CLASSES).map(|c| combos(c) * freqs[c]).sum();
    let total_hands: f64 = (0..NUM_CLASSES).map(combos).sum();
    (total_combos / total_hands * 100.0, total_combos)
}

pub async fn start_pushfold(
    State(state): State<AppState>,
    Json(req): Json<PushFoldRequest>,
) -> Json<Value> {
    let job_id = uuid::Uuid::new_v4().to_string();

    state.jobs.lock().unwrap().insert(
        job_id.clone(),
        JobStatus::Running {
            progress: "Starting...".into(),
            pct: 0.0,
            current_step: 0,
            total_steps: 0,
        },
    );

    let jobs = state.jobs.clone();
    let cache = state.push_fold_cache.clone();
    let jid = job_id.clone();

    tokio::task::spawn_blocking(move || {
        // Step 1: Get or compute equity matrix
        let update_progress = |msg: &str, pct: f64, step: usize, total: usize| {
            if let Ok(mut j) = jobs.lock() {
                j.insert(
                    jid.clone(),
                    JobStatus::Running {
                        progress: msg.to_string(),
                        pct,
                        current_step: step,
                        total_steps: total,
                    },
                );
            }
        };

        update_progress("Computing equity matrix...", -1.0, 0, req.stacks.len() + 1);

        let data = {
            let cached = cache.lock().unwrap().take();
            if let Some(c) = cached {
                if c.samples >= req.mc_samples {
                    let data = PushFoldData::new(
                        c.data.equity.clone(),
                        c.data.weights.clone(),
                    );
                    *cache.lock().unwrap() = Some(c);
                    data
                } else {
                    let data = PushFoldData::compute(req.mc_samples);
                    *cache.lock().unwrap() = Some(PushFoldDataCache {
                        data: PushFoldData::new(
                            data.equity.clone(),
                            data.weights.clone(),
                        ),
                        samples: req.mc_samples,
                    });
                    data
                }
            } else {
                let data = PushFoldData::compute(req.mc_samples);
                *cache.lock().unwrap() = Some(PushFoldDataCache {
                    data: PushFoldData::new(
                        data.equity.clone(),
                        data.weights.clone(),
                    ),
                    samples: req.mc_samples,
                });
                data
            }
        };

        // Step 2: Solve each stack
        let labels: Vec<String> = (0..13)
            .rev()
            .map(|r| {
                ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"][r].to_string()
            })
            .collect();

        // Build hand name grid (upper-right = suited, lower-left = offsuit)
        let mut hand_names = vec![vec![String::new(); 13]; 13];
        for row in (0..13).rev() {
            for col in (0..13).rev() {
                let class = if row == col {
                    row
                } else if col > row {
                    91 + (col * (col - 1) / 2 + row)
                } else {
                    13 + (row * (row - 1) / 2 + col)
                };
                hand_names[12 - row][12 - col] = class_index_to_name(class);
            }
        }

        let mut results = Vec::new();
        let num_stacks = req.stacks.len();
        let iters = req.iterations;
        for (i, &stack) in req.stacks.iter().enumerate() {
            let base_pct = 20.0 + (i as f64 / num_stacks as f64) * 80.0;
            let step_width = 80.0 / num_stacks as f64;
            update_progress(
                &format!("Solving stack {:.0}bb ({}/{})...", stack, i + 1, num_stacks),
                base_pct,
                i + 1,
                num_stacks + 1,
            );

            let game = PushFoldGame::new(
                stack,
                PushFoldData::new(data.equity.clone(), data.weights.clone()),
            );

            let config = TrainerConfig {
                iterations: iters,
                use_cfr_plus: false,
                use_chance_sampling: true,
                print_interval: 0,
            };

            let solver = train_with_callback(&game, &config, |iter, total| {
                let iter_pct = base_pct + step_width * (iter as f64 / total as f64);
                update_progress(
                    &format!("Solving stack {:.0}bb ({}/{}) - iter {}/{}",
                        stack, i + 1, num_stacks, iter, total),
                    iter_pct,
                    i + 1,
                    num_stacks + 1,
                );
            });
            let exploit = solver.exploitability(&game);
            let strategy = Strategy::from_solver(&solver);

            let push = extract_push_range(&strategy);
            let call = extract_call_range(&strategy);

            let (push_pct, push_combos) = range_stats(&push);
            let (call_pct, call_combos) = range_stats(&call);

            results.push(json!({
                "stack_bb": stack,
                "exploitability": exploit,
                "push_range": {
                    "grid": freq_to_grid(&push),
                    "pct": push_pct,
                    "combos": push_combos,
                },
                "call_range": {
                    "grid": freq_to_grid(&call),
                    "pct": call_pct,
                    "combos": call_combos,
                },
            }));
        }

        if let Ok(mut j) = jobs.lock() {
            j.insert(
                jid,
                JobStatus::Completed {
                    result: json!({
                        "labels": labels,
                        "hand_names": hand_names,
                        "results": results,
                    }),
                },
            );
        }
    });

    Json(json!({
        "job_id": job_id,
        "status": "running",
    }))
}

pub async fn pushfold_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    job_status_response(&state, &job_id)
}

fn job_status_response(
    state: &AppState,
    job_id: &str,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let jobs = state.jobs.lock().unwrap();
    match jobs.get(job_id) {
        Some(JobStatus::Running { progress, pct, current_step, total_steps }) => Ok(Json(json!({
            "job_id": job_id,
            "status": "running",
            "progress": progress,
            "progress_pct": pct,
            "current_step": current_step,
            "total_steps": total_steps,
        }))),
        Some(JobStatus::Completed { result }) => Ok(Json(json!({
            "job_id": job_id,
            "status": "completed",
            "result": result,
        }))),
        Some(JobStatus::Failed { error }) => Ok(Json(json!({
            "job_id": job_id,
            "status": "failed",
            "error": error,
        }))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Job not found"})),
        )),
    }
}

// --- Range vs Range Equity ---

#[derive(Deserialize)]
pub struct RangeEquityRequest {
    pub range1: Vec<f64>,
    pub range2: Vec<f64>,
    #[serde(default)]
    pub board: Vec<String>,
    pub mc_samples: Option<usize>,
}

fn vec_to_169(v: &[f64]) -> Result<[f64; NUM_CLASSES], String> {
    if v.len() != NUM_CLASSES {
        return Err(format!("Range must have {} elements, got {}", NUM_CLASSES, v.len()));
    }
    let mut arr = [0.0f64; NUM_CLASSES];
    arr.copy_from_slice(v);
    Ok(arr)
}

pub async fn calc_range_equity(
    State(state): State<AppState>,
    Json(req): Json<RangeEquityRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let range1 = vec_to_169(&req.range1)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;
    let range2 = vec_to_169(&req.range2)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    let board_cards: Result<Vec<Card>, String> = req.board.iter().map(|s| parse_card(s)).collect();
    let board = board_cards
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    if !board.is_empty() && board.len() != 3 && board.len() != 4 && board.len() != 5 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Board must have 0, 3, 4, or 5 cards"})),
        ));
    }

    // If preflop (no board), use async job pattern
    if board.is_empty() {
        return start_range_equity_job(state, range1, range2, req.mc_samples.unwrap_or(500_000));
    }

    let result = tokio::task::spawn_blocking(move || {
        let res = range_vs_range_equity(&range1, &range2, &board);
        build_range_equity_json(&res, &range1, &range2)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    Ok(Json(result))
}

fn build_range_equity_json(
    res: &gto_eval::RangeEquityResult,
    range1: &[f64; NUM_CLASSES],
    range2: &[f64; NUM_CLASSES],
) -> Value {
    let labels: Vec<String> = (0..13)
        .rev()
        .map(|r| ["2","3","4","5","6","7","8","9","T","J","Q","K","A"][r].to_string())
        .collect();

    let mut hand_names = vec![vec![String::new(); 13]; 13];
    for row in (0..13).rev() {
        for col in (0..13).rev() {
            let class = if row == col { row }
            else if col > row { 91 + (col * (col - 1) / 2 + row) }
            else { 13 + (row * (row - 1) / 2 + col) };
            hand_names[12 - row][12 - col] = re_class_name(class);
        }
    }

    let eq_grid = freq_to_grid_f64(&res.class_equity);
    let weight_grid = freq_to_grid_f64(&res.class_weight);

    let (r1_pct, r1_combos) = re_range_stats(range1);
    let (r2_pct, r2_combos) = re_range_stats(range2);

    json!({
        "equity": res.equity * 100.0,
        "total_matchups": res.total_matchups,
        "equity_grid": eq_grid,
        "weight_grid": weight_grid,
        "labels": labels,
        "hand_names": hand_names,
        "range1_pct": r1_pct,
        "range1_combos": r1_combos,
        "range2_pct": r2_pct,
        "range2_combos": r2_combos,
    })
}

fn freq_to_grid_f64(freqs: &[f64; NUM_CLASSES]) -> Vec<Vec<f64>> {
    let mut grid = vec![vec![0.0; 13]; 13];
    for row in (0..13).rev() {
        for col in (0..13).rev() {
            let class = if row == col { row }
            else if col > row { 91 + (col * (col - 1) / 2 + row) }
            else { 13 + (row * (row - 1) / 2 + col) };
            grid[12 - row][12 - col] = freqs[class];
        }
    }
    grid
}

fn start_range_equity_job(
    state: AppState,
    range1: [f64; NUM_CLASSES],
    range2: [f64; NUM_CLASSES],
    mc_samples: usize,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let job_id = uuid::Uuid::new_v4().to_string();
    state.jobs.lock().unwrap().insert(
        job_id.clone(),
        JobStatus::Running {
            progress: "Computing range vs range equity (preflop MC)...".into(),
            pct: -1.0,
            current_step: 0,
            total_steps: 1,
        },
    );

    let jobs = state.jobs.clone();
    let jid = job_id.clone();

    tokio::task::spawn_blocking(move || {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::from_entropy();
        let res = range_vs_range_monte_carlo(&range1, &range2, mc_samples, &mut rng);
        let result = build_range_equity_json(&res, &range1, &range2);

        if let Ok(mut j) = jobs.lock() {
            j.insert(jid, JobStatus::Completed { result });
        }
    });

    Ok(Json(json!({
        "job_id": job_id,
        "status": "running",
    })))
}

pub async fn range_equity_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    job_status_response(&state, &job_id)
}

// --- Postflop Review ---

#[derive(Deserialize)]
pub struct PostflopReviewRequest {
    pub hero_hand: [String; 2],
    pub board: Vec<String>,
    pub pot: f64,
    pub effective_stack: f64,
    pub villain_bet: f64,
    pub villain_range: Vec<f64>,
}

pub async fn calc_postflop_review(
    Json(req): Json<PostflopReviewRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let hero: Result<Vec<Card>, String> = req.hero_hand.iter().map(|s| parse_card(s)).collect();
    let hero = hero.map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;
    let board: Result<Vec<Card>, String> = req.board.iter().map(|s| parse_card(s)).collect();
    let board = board.map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    if board.len() < 3 || board.len() > 5 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Board must have 3-5 cards"}))));
    }

    let villain_range = vec_to_169(&req.villain_range)
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    let pot = req.pot;
    let eff_stack = req.effective_stack;
    let villain_bet = req.villain_bet;
    let hero_cards = [hero[0], hero[1]];

    let result = tokio::task::spawn_blocking(move || {
        let hero_name = hand_class_name(hero_cards[0], hero_cards[1]);

        // Equity vs range
        let equity = hand_vs_range_equity(hero_cards, &villain_range, &board);

        // Board texture
        let tex = board_texture(&board);

        // Hand category on current board
        let hand_cat = if board.len() >= 3 {
            let mut cards7 = [Card(0); 7];
            cards7[0] = hero_cards[0];
            cards7[1] = hero_cards[1];
            for (i, &c) in board.iter().enumerate() {
                cards7[2 + i] = c;
            }
            // Pad with dummy if < 5 board cards (evaluate needs 7)
            if board.len() < 5 {
                let mut all = vec![hero_cards[0], hero_cards[1]];
                all.extend_from_slice(&board);
                if all.len() >= 5 {
                    let s = gto_eval::evaluate_5(&[all[0], all[1], all[2], all[3], all[4]]);
                    category_name(s).to_string()
                } else {
                    String::new()
                }
            } else {
                let s = gto_eval::evaluate_7(&cards7);
                category_name(s).to_string()
            }
        } else {
            String::new()
        };

        // EV calculations
        let pot_after_bet = pot + villain_bet;
        let cost_to_call = villain_bet;
        let pot_odds = if pot_after_bet + cost_to_call > 0.0 {
            cost_to_call / (pot_after_bet + cost_to_call) * 100.0
        } else {
            0.0
        };

        let ev_fold = 0.0;
        let ev_call = equity * (pot_after_bet + cost_to_call) - cost_to_call;

        // Raise EV (simple model: assume villain folds X% or calls)
        let raise_sizes: Vec<f64> = if eff_stack > villain_bet {
            let small_raise = (villain_bet * 2.5).min(eff_stack);
            let big_raise = eff_stack; // all-in
            if (small_raise - big_raise).abs() < 1.0 {
                vec![big_raise]
            } else {
                vec![small_raise, big_raise]
            }
        } else {
            vec![]
        };

        let raise_evs: Vec<Value> = raise_sizes.iter().map(|&size| {
            // Fold equity needed: how often villain must fold for raise to be +EV
            let risk = size;
            let fold_equity_needed = if pot_after_bet + risk > 0.0 {
                (risk - equity * (pot_after_bet + risk)) / (pot_after_bet + risk) * 100.0
            } else {
                100.0
            };
            // EV if villain always calls (worst case for bluff)
            let ev_if_called = equity * (pot_after_bet + size * 2.0) - size;
            json!({
                "size": size,
                "ev_if_called": ev_if_called,
                "fold_equity_needed": fold_equity_needed.max(0.0),
            })
        }).collect();

        // Count outs (only on flop/turn)
        let outs = if board.len() <= 4 {
            gto_eval::range_equity::count_outs(hero_cards, &villain_range, &board)
        } else {
            0
        };

        // Recommendation
        let recommendation = if villain_bet <= 0.0 {
            if equity > 0.6 { "bet" } else { "check" }
        } else if ev_call > 0.0 {
            "call"
        } else {
            "fold"
        };

        json!({
            "hero_hand_name": hero_name,
            "hero_equity": equity * 100.0,
            "pot_odds": pot_odds,
            "ev_call": ev_call,
            "ev_fold": ev_fold,
            "ev_raise": raise_evs,
            "recommendation": recommendation,
            "hand_category": hand_cat,
            "outs": outs,
            "board_texture": {
                "paired": tex.paired,
                "two_tone": tex.two_tone,
                "monotone": tex.monotone,
                "straight_possible": tex.straight_possible,
                "high_card": tex.high_card,
            }
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))))?;

    Ok(Json(result))
}

// --- Preflop GTO ---

#[derive(Deserialize)]
pub struct PreflopRequest {
    #[serde(default = "default_preflop_stack")]
    pub stack_bb: f64,
    #[serde(default = "default_mc_samples")]
    pub mc_samples: usize,
    #[serde(default = "default_preflop_iterations")]
    pub iterations: usize,
    /// Optional position filter: "UTG", "HJ", "CO", "BTN", "SB", or null for all.
    pub position: Option<String>,
    /// Optional defender filter: "BB", "SB", "BTN", "CO", "HJ", "all", or null (= BB).
    pub defender: Option<String>,
    /// Tree depth limit: 1=Open only, 2=up to 3bet, 3=full (4bet). Default 3.
    pub max_raises: Option<usize>,
}

fn default_preflop_stack() -> f64 {
    100.0
}
fn default_preflop_iterations() -> usize {
    2_000_000
}

fn parse_position(s: &str) -> Option<Position> {
    match s.to_uppercase().as_str() {
        "UTG" => Some(Position::UTG),
        "HJ" => Some(Position::HJ),
        "CO" => Some(Position::CO),
        "BTN" => Some(Position::BTN),
        "SB" => Some(Position::SB),
        _ => None,
    }
}

fn solve_single_matchup<F: FnMut(usize, usize)>(
    data: &PushFoldData,
    stack_bb: f64,
    opener: Position,
    defender: Position,
    iterations: usize,
    max_raises: Option<usize>,
    on_progress: F,
) -> Value {
    let mut config = PreflopConfig::for_matchup(stack_bb, opener, defender);
    if let Some(mr) = max_raises {
        config = config.with_max_raises(mr);
    }
    let game = PreflopGame::new(
        config.clone(),
        PushFoldData::new(data.equity.clone(), data.weights.clone()),
    );

    let tc = TrainerConfig {
        iterations,
        use_cfr_plus: true,
        use_chance_sampling: true,
        print_interval: 0,
    };

    let solver = train_with_callback(&game, &tc, on_progress);
    let exploit = solver.exploitability(&game);
    let strategy = Strategy::from_solver(&solver);

    let strat_sets = extract_preflop_strategies(&strategy, &config);

    let decision_points: Vec<Value> = strat_sets
        .iter()
        .map(|set| {
            let actions: Vec<Value> = set
                .action_names
                .iter()
                .enumerate()
                .map(|(ai, name)| {
                    let grid = freq_to_grid(&set.freqs[ai]);
                    let (pct, combos) = range_stats(&set.freqs[ai]);
                    json!({
                        "name": name,
                        "grid": grid,
                        "pct": pct,
                        "combos": combos,
                    })
                })
                .collect();
            json!({
                "label": set.label,
                "history": set.history_prefix,
                "actions": actions,
            })
        })
        .collect();

    json!({
        "decision_points": decision_points,
        "exploitability": exploit,
    })
}

pub async fn start_preflop(
    State(state): State<AppState>,
    Json(req): Json<PreflopRequest>,
) -> Json<Value> {
    let job_id = uuid::Uuid::new_v4().to_string();

    state.jobs.lock().unwrap().insert(
        job_id.clone(),
        JobStatus::Running {
            progress: "Starting...".into(),
            pct: 0.0,
            current_step: 0,
            total_steps: 0,
        },
    );

    let jobs = state.jobs.clone();
    let cache = state.push_fold_cache.clone();
    let jid = job_id.clone();

    // Determine which matchups (opener, defender) to solve
    let matchups: Vec<(Position, Position)> = {
        let opener_opt = req.position.as_deref().and_then(|s| {
            if s.eq_ignore_ascii_case("all") { None } else { parse_position(s) }
        });
        let defender_str = req.defender.as_deref().unwrap_or("BB");
        let defender_all = defender_str.eq_ignore_ascii_case("all");
        let defender_opt = if defender_all { None } else { parse_position(defender_str) };

        match (opener_opt, defender_opt, defender_all) {
            // Specific opener, specific defender
            (Some(o), Some(d), _) => vec![(o, d)],
            // Specific opener, all defenders
            (Some(o), None, true) => o.defenders().iter().map(|&d| (o, d)).collect(),
            // Specific opener, default BB
            (Some(o), None, false) => vec![(o, Position::BB)],
            // All openers, specific defender
            (None, Some(d), _) => Position::ALL.iter()
                .filter(|&&p| p.defenders().contains(&d))
                .map(|&p| (p, d))
                .collect(),
            // All openers, all defenders
            (None, None, true) => Position::all_matchups(),
            // All openers, default BB
            (None, None, false) => Position::ALL.iter().map(|&p| (p, Position::BB)).collect(),
        }
    };

    tokio::task::spawn_blocking(move || {
        let num_matchups = matchups.len();
        let update_progress = |msg: &str, pct: f64, step: usize, total: usize| {
            if let Ok(mut j) = jobs.lock() {
                j.insert(
                    jid.clone(),
                    JobStatus::Running {
                        progress: msg.to_string(),
                        pct,
                        current_step: step,
                        total_steps: total,
                    },
                );
            }
        };

        update_progress("Computing equity matrix...", -1.0, 0, num_matchups + 1);

        let data = {
            let cached = cache.lock().unwrap().take();
            if let Some(c) = cached {
                if c.samples >= req.mc_samples {
                    let data = PushFoldData::new(
                        c.data.equity.clone(),
                        c.data.weights.clone(),
                    );
                    *cache.lock().unwrap() = Some(c);
                    data
                } else {
                    let data = PushFoldData::compute(req.mc_samples);
                    *cache.lock().unwrap() = Some(PushFoldDataCache {
                        data: PushFoldData::new(
                            data.equity.clone(),
                            data.weights.clone(),
                        ),
                        samples: req.mc_samples,
                    });
                    data
                }
            } else {
                let data = PushFoldData::compute(req.mc_samples);
                *cache.lock().unwrap() = Some(PushFoldDataCache {
                    data: PushFoldData::new(
                        data.equity.clone(),
                        data.weights.clone(),
                    ),
                    samples: req.mc_samples,
                });
                data
            }
        };

        let labels: Vec<String> = (0..13)
            .rev()
            .map(|r| {
                ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"][r].to_string()
            })
            .collect();

        let mut hand_names = vec![vec![String::new(); 13]; 13];
        for row in (0..13).rev() {
            for col in (0..13).rev() {
                let class = if row == col {
                    row
                } else if col > row {
                    91 + (col * (col - 1) / 2 + row)
                } else {
                    13 + (row * (row - 1) / 2 + col)
                };
                hand_names[12 - row][12 - col] = class_index_to_name(class);
            }
        }

        // Group matchups by opener
        let mut pos_results = serde_json::Map::new();
        let iters = req.iterations;
        let stack_bb = req.stack_bb;
        let max_raises = req.max_raises;
        for (i, &(opener, defender)) in matchups.iter().enumerate() {
            let base_pct = 15.0 + (i as f64 / num_matchups as f64) * 85.0;
            let step_width = 85.0 / num_matchups as f64;
            update_progress(
                &format!(
                    "Solving {} vs {} {:.0}bb ({}/{})...",
                    opener.name(),
                    defender.name(),
                    stack_bb,
                    i + 1,
                    num_matchups,
                ),
                base_pct,
                i + 1,
                num_matchups + 1,
            );
            let result = solve_single_matchup(&data, stack_bb, opener, defender, iters, max_raises, |iter, total| {
                let iter_pct = base_pct + step_width * (iter as f64 / total as f64);
                update_progress(
                    &format!("Solving {} vs {} {:.0}bb ({}/{}) - iter {}/{}",
                        opener.name(), defender.name(), stack_bb, i + 1, num_matchups, iter, total),
                    iter_pct,
                    i + 1,
                    num_matchups + 1,
                );
            });

            let opener_entry = pos_results
                .entry(opener.name().to_string())
                .or_insert_with(|| json!({}));
            if let Value::Object(ref mut map) = opener_entry {
                map.insert(defender.name().to_string(), result);
            }
        }

        if let Ok(mut j) = jobs.lock() {
            j.insert(
                jid,
                JobStatus::Completed {
                    result: json!({
                        "stack_bb": req.stack_bb,
                        "labels": labels,
                        "hand_names": hand_names,
                        "positions": pos_results,
                    }),
                },
            );
        }
    });

    Json(json!({
        "job_id": job_id,
        "status": "running",
    }))
}

pub async fn preflop_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    job_status_response(&state, &job_id)
}

// --- Range Presets ---

pub async fn get_range_presets() -> Json<Value> {
    Json(json!({ "presets": build_presets() }))
}

fn build_presets() -> Value {
    // Standard opening ranges as [f64; 169] arrays
    // Class order: 0-12 pairs (22-AA), 13-90 suited, 91-168 offsuit
    // We'll define presets by listing included hand names, then convert

    let presets = vec![
        ("utg_open", "UTG Open (15%)", "UTGオープン (15%)",
         &["AA","KK","QQ","JJ","TT","99","88","77",
           "AKs","AQs","AJs","ATs","A5s","A4s",
           "KQs","KJs","KTs","QJs","QTs","JTs","T9s","98s",
           "AKo","AQo"][..]),
        ("mp_open", "MP Open (20%)", "MPオープン (20%)",
         &["AA","KK","QQ","JJ","TT","99","88","77","66",
           "AKs","AQs","AJs","ATs","A9s","A5s","A4s","A3s",
           "KQs","KJs","KTs","K9s","QJs","QTs","Q9s","JTs","J9s","T9s","98s","87s",
           "AKo","AQo","AJo","KQo"][..]),
        ("co_open", "CO Open (27%)", "COオープン (27%)",
         &["AA","KK","QQ","JJ","TT","99","88","77","66","55",
           "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
           "KQs","KJs","KTs","K9s","K8s","QJs","QTs","Q9s","JTs","J9s","J8s","T9s","T8s","98s","87s","76s",
           "AKo","AQo","AJo","ATo","KQo","KJo","QJo"][..]),
        ("btn_open", "BTN Open (40%)", "BTNオープン (40%)",
         &["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
           "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
           "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s",
           "QJs","QTs","Q9s","Q8s","JTs","J9s","J8s","T9s","T8s","T7s","98s","97s","87s","86s","76s","75s","65s","54s",
           "AKo","AQo","AJo","ATo","A9o","KQo","KJo","KTo","QJo","QTo","JTo"][..]),
        ("sb_open", "SB Open (45%)", "SBオープン (45%)",
         &["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
           "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
           "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s","K4s","K3s",
           "QJs","QTs","Q9s","Q8s","Q7s","JTs","J9s","J8s","J7s","T9s","T8s","T7s","98s","97s","96s","87s","86s","76s","75s","65s","64s","54s","53s",
           "AKo","AQo","AJo","ATo","A9o","A8o","KQo","KJo","KTo","K9o","QJo","QTo","JTo","J9o","T9o"][..]),
        ("bb_defend", "BB Defend vs 2.5x (38%)", "BBディフェンス vs 2.5x (38%)",
         &["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
           "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
           "KQs","KJs","KTs","K9s","K8s","K7s","K6s",
           "QJs","QTs","Q9s","Q8s","Q7s","JTs","J9s","J8s","T9s","T8s","98s","97s","87s","86s","76s","75s","65s","54s",
           "AKo","AQo","AJo","ATo","A9o","KQo","KJo","KTo","QJo","QTo","JTo"][..]),
        ("top10", "Top 10%", "トップ10%",
         &["AA","KK","QQ","JJ","TT","99",
           "AKs","AQs","AJs","ATs","KQs","KJs","QJs","JTs",
           "AKo","AQo"][..]),
        ("top20", "Top 20%", "トップ20%",
         &["AA","KK","QQ","JJ","TT","99","88","77",
           "AKs","AQs","AJs","ATs","A9s","A5s","A4s",
           "KQs","KJs","KTs","K9s","QJs","QTs","JTs","T9s","98s",
           "AKo","AQo","AJo","ATo","KQo","KJo"][..]),
        ("top50", "Top 50%", "トップ50%",
         &["AA","KK","QQ","JJ","TT","99","88","77","66","55","44","33","22",
           "AKs","AQs","AJs","ATs","A9s","A8s","A7s","A6s","A5s","A4s","A3s","A2s",
           "KQs","KJs","KTs","K9s","K8s","K7s","K6s","K5s","K4s","K3s","K2s",
           "QJs","QTs","Q9s","Q8s","Q7s","Q6s","Q5s",
           "JTs","J9s","J8s","J7s","J6s","T9s","T8s","T7s","98s","97s","96s","87s","86s","85s","76s","75s","65s","64s","54s","53s","43s",
           "AKo","AQo","AJo","ATo","A9o","A8o","A7o","A6o","A5o","A4o",
           "KQo","KJo","KTo","K9o","K8o","QJo","QTo","Q9o","JTo","J9o","T9o","T8o","98o","97o","87o","76o"][..]),
    ];

    let mut result = serde_json::Map::new();
    for (key, name_en, name_ja, hands) in presets {
        let range = hands_to_range(hands);
        result.insert(key.to_string(), json!({
            "name_en": name_en,
            "name_ja": name_ja,
            "range": range,
        }));
    }
    Value::Object(result)
}

fn hands_to_range(hand_names: &[&str]) -> Vec<f64> {
    let mut range = vec![0.0f64; NUM_CLASSES];
    for &name in hand_names {
        if let Some(idx) = name_to_class_index(name) {
            range[idx] = 1.0;
        }
    }
    range
}

fn name_to_class_index(name: &str) -> Option<usize> {
    let bytes = name.as_bytes();
    if bytes.len() < 2 {
        return None;
    }

    let rank_of = |b: u8| -> Option<u8> {
        match b {
            b'2' => Some(0), b'3' => Some(1), b'4' => Some(2), b'5' => Some(3),
            b'6' => Some(4), b'7' => Some(5), b'8' => Some(6), b'9' => Some(7),
            b'T' => Some(8), b'J' => Some(9), b'Q' => Some(10), b'K' => Some(11),
            b'A' => Some(12), _ => None,
        }
    };

    let r1 = rank_of(bytes[0])?;
    let r2 = rank_of(bytes[1])?;
    let high = r1.max(r2);
    let low = r1.min(r2);

    if high == low {
        // Pair
        Some(high as usize)
    } else if bytes.len() >= 3 && bytes[2] == b's' {
        // Suited
        let idx = (high as usize) * (high as usize - 1) / 2 + low as usize;
        Some(13 + idx)
    } else {
        // Offsuit (or no suffix = offsuit)
        let idx = (high as usize) * (high as usize - 1) / 2 + low as usize;
        Some(91 + idx)
    }
}

// --- Postflop Solver ---

#[derive(Deserialize)]
pub struct PostflopSolveRequest {
    pub flop: [String; 3],
    pub pot: u32,
    pub effective_stack: u32,
    pub oop_range: Vec<f64>,
    pub ip_range: Vec<f64>,
    pub oop_bet_sizes: Option<Vec<Vec<u32>>>,
    pub ip_bet_sizes: Option<Vec<Vec<u32>>>,
    pub oop_raise_sizes: Option<Vec<Vec<u32>>>,
    pub ip_raise_sizes: Option<Vec<Vec<u32>>>,
    pub max_raises: Option<usize>,
    pub iterations: Option<usize>,
}

pub async fn start_postflop_solve(
    State(state): State<AppState>,
    Json(req): Json<PostflopSolveRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Parse flop cards
    let flop_cards: Result<Vec<Card>, String> = req.flop.iter().map(|s| parse_card(s)).collect();
    let flop = flop_cards
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(json!({"error": e}))))?;

    if flop.len() != 3 {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": "Flop must have exactly 3 cards"}))));
    }

    // Validate ranges
    if req.oop_range.len() != NUM_CLASSES || req.ip_range.len() != NUM_CLASSES {
        return Err((StatusCode::BAD_REQUEST, Json(json!({"error": format!("Ranges must have {} elements", NUM_CLASSES)}))));
    }

    let flop_arr = [flop[0], flop[1], flop[2]];
    let iterations = req.iterations.unwrap_or(5000).min(50000);
    let max_raises = req.max_raises.unwrap_or(2);

    let to_3 = |v: &Option<Vec<Vec<u32>>>, def: Vec<u32>| -> [Vec<u32>; 3] {
        match v {
            Some(sizes) if sizes.len() >= 3 => [sizes[0].clone(), sizes[1].clone(), sizes[2].clone()],
            Some(sizes) if sizes.len() == 1 => [sizes[0].clone(), sizes[0].clone(), sizes[0].clone()],
            _ => [def.clone(), def.clone(), def.clone()],
        }
    };

    let bet_sizes = BetSizeConfig {
        oop_bet_sizes: to_3(&req.oop_bet_sizes, vec![33, 75]),
        ip_bet_sizes: to_3(&req.ip_bet_sizes, vec![33, 75]),
        oop_raise_sizes: to_3(&req.oop_raise_sizes, vec![100]),
        ip_raise_sizes: to_3(&req.ip_raise_sizes, vec![100]),
        max_raises_per_street: max_raises,
    };

    let config = PostflopConfig {
        flop: flop_arr,
        pot: req.pot,
        effective_stack: req.effective_stack,
        bet_sizes,
        oop_range: req.oop_range.clone(),
        ip_range: req.ip_range.clone(),
        num_buckets: 10,
    };

    let job_id = uuid::Uuid::new_v4().to_string();
    state.jobs.lock().unwrap().insert(
        job_id.clone(),
        JobStatus::Running {
            progress: "Starting postflop solve...".into(),
            pct: 0.0,
            current_step: 0,
            total_steps: 4,
        },
    );

    let jobs = state.jobs.clone();
    let jid = job_id.clone();
    let flop_strs = req.flop.clone();

    tokio::task::spawn_blocking(move || {
        let update_progress = |msg: &str, pct: f64, step: usize| {
            if let Ok(mut j) = jobs.lock() {
                j.insert(
                    jid.clone(),
                    JobStatus::Running {
                        progress: msg.to_string(),
                        pct,
                        current_step: step,
                        total_steps: 4,
                    },
                );
            }
        };

        // Step 1: Build game
        update_progress("Building game...", 10.0, 1);
        let game = PostflopGame::new(config.clone());

        // Step 2: Train
        update_progress("Building game tree...", 25.0, 2);
        let trainer_config = TrainerConfig {
            iterations,
            use_cfr_plus: true,
            use_chance_sampling: true,
            print_interval: 0,
        };

        let solver = train_with_callback(&game, &trainer_config, |iter, total| {
            let pct = 25.0 + 70.0 * (iter as f64 / total as f64);
            update_progress(
                &format!("MCCFR iteration {}/{}...", iter, total),
                pct,
                3,
            );
        });

        // Step 3: Extract strategies
        update_progress("Extracting strategies...", 95.0, 4);
        let strat_sets = extract_postflop_strategies(&game, &solver);

        let strategies: Vec<Value> = strat_sets.iter().map(|set| {
            json!({
                "label": set.label,
                "key": set.key_suffix,
                "actions": set.action_names,
                "grids": set.grids,
            })
        }).collect();

        if let Ok(mut j) = jobs.lock() {
            j.insert(
                jid,
                JobStatus::Completed {
                    result: json!({
                        "flop": flop_strs,
                        "strategies": strategies,
                        "iterations": iterations,
                        "num_info_sets": solver.nodes.len(),
                    }),
                },
            );
        }
    });

    Ok(Json(json!({
        "job_id": job_id,
        "status": "running",
    })))
}

pub async fn postflop_solve_status(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    job_status_response(&state, &job_id)
}
