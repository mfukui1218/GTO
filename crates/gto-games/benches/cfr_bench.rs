use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gto_cfr::{train, TrainerConfig};
use gto_games::push_fold::{PushFoldData, PushFoldGame, NUM_CLASSES};
use gto_games::KuhnPoker;
use gto_games::LeducHoldem;

fn bench_kuhn_cfr(c: &mut Criterion) {
    let mut group = c.benchmark_group("kuhn_cfr");
    group.sample_size(10);

    group.bench_function("1000 iterations", |b| {
        b.iter(|| {
            let game = KuhnPoker;
            let config = TrainerConfig {
                iterations: 1000,
                use_cfr_plus: false,
                use_chance_sampling: false,
                print_interval: 0,
            };
            train(black_box(&game), black_box(&config))
        })
    });

    group.bench_function("1000 iterations (CFR+)", |b| {
        b.iter(|| {
            let game = KuhnPoker;
            let config = TrainerConfig {
                iterations: 1000,
                use_cfr_plus: true,
                use_chance_sampling: false,
                print_interval: 0,
            };
            train(black_box(&game), black_box(&config))
        })
    });

    group.finish();
}

fn bench_leduc_cfr(c: &mut Criterion) {
    let mut group = c.benchmark_group("leduc_cfr");
    group.sample_size(10);

    group.bench_function("500 iterations (MCCFR+)", |b| {
        b.iter(|| {
            let game = LeducHoldem;
            let config = TrainerConfig {
                iterations: 500,
                use_cfr_plus: true,
                use_chance_sampling: true,
                print_interval: 0,
            };
            train(black_box(&game), black_box(&config))
        })
    });

    group.finish();
}

fn bench_pushfold_data_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("pushfold_data");
    group.sample_size(10);

    group.bench_function("compute 100K samples (rayon)", |b| {
        b.iter(|| PushFoldData::compute(black_box(100_000)))
    });

    group.finish();
}

fn bench_pushfold_cfr(c: &mut Criterion) {
    let mut group = c.benchmark_group("pushfold_cfr");
    group.sample_size(10);

    // Pre-compute data to isolate CFR timing
    let data = PushFoldData::compute(100_000);

    group.bench_function("1000 MCCFR+ iterations", |b| {
        b.iter(|| {
            let game = PushFoldGame::new(10.0, PushFoldData::new(
                data.equity.clone(),
                data.weights.clone(),
            ));
            let config = TrainerConfig {
                iterations: 1000,
                use_cfr_plus: true,
                use_chance_sampling: true,
                print_interval: 0,
            };
            train(black_box(&game), black_box(&config))
        })
    });

    group.finish();
}

fn bench_exploitability(c: &mut Criterion) {
    let mut group = c.benchmark_group("exploitability");
    group.sample_size(10);

    // Train Kuhn solver first
    let game = KuhnPoker;
    let config = TrainerConfig {
        iterations: 1000,
        use_cfr_plus: true,
        use_chance_sampling: false,
        print_interval: 0,
    };
    let solver = train(&game, &config);

    group.bench_function("kuhn (rayon parallel BR)", |b| {
        b.iter(|| solver.exploitability(black_box(&game)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_kuhn_cfr,
    bench_leduc_cfr,
    bench_pushfold_data_compute,
    bench_pushfold_cfr,
    bench_exploitability,
);
criterion_main!(benches);
