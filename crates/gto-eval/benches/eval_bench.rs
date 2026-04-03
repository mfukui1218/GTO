use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use gto_core::Card;
use gto_eval::{evaluate_5, evaluate_7, equity_exact, range_vs_range_equity, NUM_CLASSES};

fn bench_evaluate_5(c: &mut Criterion) {
    let hand = [
        Card::new(12, 0), // Ac
        Card::new(11, 0), // Kc
        Card::new(10, 0), // Qc
        Card::new(9, 0),  // Jc
        Card::new(8, 0),  // Tc
    ];

    c.bench_function("evaluate_5", |b| {
        b.iter(|| evaluate_5(black_box(&hand)))
    });
}

fn bench_evaluate_7(c: &mut Criterion) {
    let cards = [
        Card::new(12, 0), // Ac
        Card::new(11, 1), // Kd
        Card::new(10, 2), // Qh
        Card::new(9, 3),  // Js
        Card::new(7, 0),  // 9c
        Card::new(5, 1),  // 7d
        Card::new(3, 2),  // 5h
    ];

    c.bench_function("evaluate_7 (LUT)", |b| {
        b.iter(|| evaluate_7(black_box(&cards)))
    });
}

fn bench_evaluate_7_batch(c: &mut Criterion) {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    let mut deck: Vec<Card> = (0..52).map(Card).collect();

    let mut hands = Vec::with_capacity(1000);
    for _ in 0..1000 {
        deck.shuffle(&mut rng);
        hands.push([deck[0], deck[1], deck[2], deck[3], deck[4], deck[5], deck[6]]);
    }

    c.bench_function("evaluate_7 x1000", |b| {
        b.iter(|| {
            let mut sum = 0u32;
            for hand in &hands {
                sum = sum.wrapping_add(evaluate_7(black_box(hand)));
            }
            sum
        })
    });
}

fn bench_equity_exact(c: &mut Criterion) {
    let hand1 = [Card::new(12, 0), Card::new(12, 1)]; // AcAd
    let hand2 = [Card::new(11, 0), Card::new(11, 1)]; // KcKd

    let mut group = c.benchmark_group("equity_exact");

    // River (0 cards to deal)
    let river_board = [
        Card::new(0, 2), Card::new(2, 3), Card::new(5, 0),
        Card::new(7, 1), Card::new(9, 2),
    ];
    group.bench_function("river", |b| {
        b.iter(|| equity_exact(black_box(hand1), black_box(hand2), black_box(&river_board)))
    });

    // Turn (1 card to deal)
    let turn_board = [
        Card::new(0, 2), Card::new(2, 3), Card::new(5, 0), Card::new(7, 1),
    ];
    group.bench_function("turn", |b| {
        b.iter(|| equity_exact(black_box(hand1), black_box(hand2), black_box(&turn_board)))
    });

    // Flop (2 cards to deal)
    let flop_board = [Card::new(0, 2), Card::new(2, 3), Card::new(5, 0)];
    group.bench_function("flop", |b| {
        b.iter(|| equity_exact(black_box(hand1), black_box(hand2), black_box(&flop_board)))
    });

    group.finish();
}

fn bench_range_vs_range(c: &mut Criterion) {
    // AA vs KK on a river board
    let mut range1 = [0.0f64; NUM_CLASSES];
    let mut range2 = [0.0f64; NUM_CLASSES];
    range1[12] = 1.0; // AA
    range2[11] = 1.0; // KK
    let board = [
        Card::new(0, 2), Card::new(2, 3), Card::new(5, 0),
        Card::new(7, 1), Card::new(9, 2),
    ];

    c.bench_function("range_vs_range (AA vs KK, river)", |b| {
        b.iter(|| range_vs_range_equity(black_box(&range1), black_box(&range2), black_box(&board)))
    });

    // Wider ranges on flop
    let mut wide1 = [0.0f64; NUM_CLASSES];
    let mut wide2 = [0.0f64; NUM_CLASSES];
    // Top 20% range (roughly pairs + broadway)
    for i in 6..13 { wide1[i] = 1.0; } // 88+
    for i in 70..91 { wide1[i] = 1.0; } // suited broadway
    for i in 6..13 { wide2[i] = 1.0; }
    for i in 70..91 { wide2[i] = 1.0; }
    let flop = [Card::new(0, 2), Card::new(2, 3), Card::new(5, 0)];

    c.bench_function("range_vs_range (20% vs 20%, flop, rayon)", |b| {
        b.iter(|| range_vs_range_equity(black_box(&wide1), black_box(&wide2), black_box(&flop)))
    });
}

criterion_group!(
    benches,
    bench_evaluate_5,
    bench_evaluate_7,
    bench_evaluate_7_batch,
    bench_equity_exact,
    bench_range_vs_range,
);
criterion_main!(benches);
