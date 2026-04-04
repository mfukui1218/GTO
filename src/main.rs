use gto_cfr::{train, Strategy, TrainerConfig};
use gto_core::Card;
use gto_eval::{equity_exact, hand_class_name};
use gto_games::preflop::{
    extract_preflop_strategies, display_preflop_chart, Position, PreflopConfig, PreflopGame,
};
use gto_games::push_fold::{
    display_chart, extract_call_range, extract_push_range, PushFoldData, PushFoldGame,
};
use gto_games::{KuhnPoker, LeducHoldem};

mod web;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let game_name = args.get(1).map(|s| s.as_str()).unwrap_or("kuhn");

    match game_name {
        "kuhn" => run_kuhn(),
        "leduc" => run_leduc(),
        "equity" => run_equity(),
        "pushfold" => run_push_fold(),
        "preflop" => {
            let mode = args.get(2).map(|s| s.as_str()).unwrap_or("balanced");
            run_preflop(mode);
        }
        "web" => {
            let port = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080u16);
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(web::server::run_server(port));
        }
        _ => {
            eprintln!("Usage: gto-solver [kuhn|leduc|equity|pushfold|preflop [fast|balanced|accurate]|web [port]]");
            std::process::exit(1);
        }
    }
}

fn c(s: &str) -> Card {
    let bytes = s.as_bytes();
    let rank = match bytes[0] {
        b'2' => 0, b'3' => 1, b'4' => 2, b'5' => 3, b'6' => 4,
        b'7' => 5, b'8' => 6, b'9' => 7, b'T' => 8, b'J' => 9,
        b'Q' => 10, b'K' => 11, b'A' => 12,
        _ => panic!("bad rank"),
    };
    let suit = match bytes[1] {
        b'c' => 0, b'd' => 1, b'h' => 2, b's' => 3,
        _ => panic!("bad suit"),
    };
    Card::new(rank, suit)
}

fn run_kuhn() {
    println!("=== Phase 1: Kuhn Poker ===");
    println!();

    let game = KuhnPoker;
    let config = TrainerConfig {
        iterations: 100_000,
        use_cfr_plus: false,
        use_chance_sampling: false,
        print_interval: 20_000,
    };

    let solver = train(&game, &config);
    let strategy = Strategy::from_solver(&solver);
    let exploit = solver.exploitability(&game);

    println!();
    println!("{} info sets, exploitability = {:.6}", solver.nodes.len(), exploit);
    println!();

    println!("Player 0 (first to act):       Check   Bet");
    for card in &["J", "Q", "K"] {
        if let Some(p) = strategy.get(card) {
            println!("  {:<12}                  {:.4}   {:.4}", card, p[0], p[1]);
        }
    }

    println!("Player 0 (facing bet):         Fold    Call");
    for card in &["J", "Q", "K"] {
        let key = format!("{}|xb1", card);
        if let Some(p) = strategy.get(&key) {
            println!("  {:<12}                  {:.4}   {:.4}", key, p[0], p[1]);
        }
    }

    println!("Player 1 (after check):        Check   Bet");
    for card in &["J", "Q", "K"] {
        let key = format!("{}|x", card);
        if let Some(p) = strategy.get(&key) {
            println!("  {:<12}                  {:.4}   {:.4}", key, p[0], p[1]);
        }
    }

    println!("Player 1 (facing bet):         Fold    Call");
    for card in &["J", "Q", "K"] {
        let key = format!("{}|b1", card);
        if let Some(p) = strategy.get(&key) {
            println!("  {:<12}                  {:.4}   {:.4}", key, p[0], p[1]);
        }
    }

    println!();
    println!("Nash game value for P0: -1/18 = {:.6}", -1.0 / 18.0);
}

fn run_leduc() {
    println!("=== Phase 2: Leduc Hold'em ===");
    println!();

    let game = LeducHoldem;
    let config = TrainerConfig {
        iterations: 100_000,
        use_cfr_plus: false,
        use_chance_sampling: false,
        print_interval: 20_000,
    };

    let solver = train(&game, &config);
    let strategy = Strategy::from_solver(&solver);
    let exploit = solver.exploitability(&game);

    println!();
    println!(
        "{} info sets, exploitability = {:.6}",
        solver.nodes.len(),
        exploit
    );
    println!();

    println!("--- Preflop (bet size = 2) ---");
    println!();
    println!("Opening (P0, first to act):    Check   Bet");
    for card in &["J", "Q", "K"] {
        if let Some(p) = strategy.get(*card) {
            println!("  {:<12}                  {:.4}   {:.4}", card, p[0], p[1]);
        }
    }

    println!("Facing open (P1):              Check   Bet");
    for card in &["J", "Q", "K"] {
        let key = format!("{}|x", card);
        if let Some(p) = strategy.get(&key) {
            println!("  {:<12}                  {:.4}   {:.4}", key, p[0], p[1]);
        }
    }

    println!("Facing bet (P1):               Fold    Call    Raise");
    for card in &["J", "Q", "K"] {
        let key = format!("{}|b2", card);
        if let Some(p) = strategy.get(&key) {
            println!(
                "  {:<12}                  {:.4}   {:.4}   {:.4}",
                key, p[0], p[1], p[2]
            );
        }
    }

    println!();
    println!("--- Flop after check-check (bet size = 4) ---");
    println!();
    for board in &["J", "Q", "K"] {
        println!("Board: {}                       Check   Bet", board);
        for card in &["J", "Q", "K"] {
            let key = format!("{}|xx:{}", card, board);
            if let Some(p) = strategy.get(&key) {
                let label = if card == board {
                    format!("{}* (pair)", card)
                } else {
                    card.to_string()
                };
                println!("  P0 {:<10}                 {:.4}   {:.4}", label, p[0], p[1]);
            }
        }
        println!();
    }
}

fn run_equity() {
    println!("=== Phase 3: Hand Evaluator & Equity ===");
    println!();

    let matchups: Vec<([&str; 2], [&str; 2])> = vec![
        (["As", "Ah"], ["Ks", "Kh"]), // AA vs KK
        (["As", "Ks"], ["Qh", "Qd"]), // AKs vs QQ
        (["As", "Ks"], ["Ad", "Kd"]), // AKs vs AKs
        (["As", "Ah"], ["7d", "2c"]), // AA vs 72o
        (["Jc", "Tc"], ["9h", "9d"]), // JTs vs 99
        (["As", "Qh"], ["Kd", "Jd"]), // AQo vs KJs
    ];

    println!("{:<14} {:<14} {:>8} {:>8} {:>8} {:>10}", "Hand 1", "Hand 2", "Win%", "Tie%", "Lose%", "Equity");
    println!("{}", "-".repeat(68));

    for (h1, h2) in &matchups {
        let hand1 = [c(h1[0]), c(h1[1])];
        let hand2 = [c(h2[0]), c(h2[1])];

        let name1 = hand_class_name(hand1[0], hand1[1]);
        let name2 = hand_class_name(hand2[0], hand2[1]);

        let result = equity_exact(hand1, hand2, &[]);

        let win_pct = result.wins as f64 / result.total as f64 * 100.0;
        let tie_pct = result.ties as f64 / result.total as f64 * 100.0;
        let lose_pct = result.losses as f64 / result.total as f64 * 100.0;
        let equity = result.equity() * 100.0;

        println!(
            "{:<14} {:<14} {:>7.2}% {:>7.2}% {:>7.2}% {:>8.2}%",
            name1, name2, win_pct, tie_pct, lose_pct, equity
        );
    }
}

fn run_push_fold() {
    println!("=== Phase 4: Push/Fold Solver ===");
    println!();
    println!("Computing equity matrix (Monte Carlo, 2M samples)...");

    let data = PushFoldData::compute(2_000_000);

    for &stack in &[5.0, 10.0, 15.0, 20.0] {
        println!();
        println!("=== Stack: {:.0}bb ===", stack);
        println!();

        let game = PushFoldGame::new(
            stack,
            PushFoldData::new(data.equity.clone(), data.weights.clone()),
        );

        let config = TrainerConfig {
            iterations: 500_000,
            use_cfr_plus: false,
            use_chance_sampling: true,
            print_interval: 0,
        };

        let solver = train(&game, &config);
        let exploit = solver.exploitability(&game);
        let strategy = Strategy::from_solver(&solver);

        println!("Exploitability: {:.4} bb", exploit);
        println!();

        let push = extract_push_range(&strategy);
        let call = extract_call_range(&strategy);

        display_chart(&format!("SB Push Range ({:.0}bb)", stack), &push);
        println!();
        display_chart(&format!("BB Call Range ({:.0}bb)", stack), &call);
    }
}

fn run_preflop(mode: &str) {
    let (iterations, chance_sampling, mc_samples, mode_label) = match mode {
        "fast" => (2_000_000, true, 500_000, "Fast (MCCFR+ 2M, ~5s/matchup)"),
        "accurate" => (5_000, false, 2_000_000, "Accurate (Vanilla CFR+ 5K, ~3min/matchup)"),
        _ => (500_000, true, 1_000_000, "Balanced (MCCFR+ 500K, ~2s/matchup)"),
    };

    println!("=== Preflop GTO Solver (6-max) ===");
    println!("Mode: {}", mode_label);
    println!();
    println!("Computing equity matrix (Monte Carlo, {}K samples)...", mc_samples / 1000);

    let data = PushFoldData::compute(mc_samples);

    for &stack in &[25.0, 50.0, 100.0] {
        println!();
        println!("========== Stack: {:.0}bb ==========", stack);

        for &pos in &Position::ALL {
            println!();
            println!("----- {} vs BB -----", pos.name());
            println!();

            let config = PreflopConfig::for_position(stack, pos);
            let game = PreflopGame::new(
                config.clone(),
                PushFoldData::new(data.equity.clone(), data.weights.clone()),
            );

            let tc = TrainerConfig {
                iterations,
                use_cfr_plus: true,
                use_chance_sampling: chance_sampling,
                print_interval: 0,
            };

            let solver = train(&game, &tc);
            let exploit = solver.exploitability(&game);
            let strategy = Strategy::from_solver(&solver);

            println!("Exploitability: {:.4} bb", exploit);
            println!();

            let strat_sets = extract_preflop_strategies(&strategy, &config);
            for set in &strat_sets {
                for (ai, action_name) in set.action_names.iter().enumerate() {
                    display_preflop_chart(&set.label, action_name, &set.freqs[ai]);
                    println!();
                }
            }
        }
    }
}
