//! Benchmarks for charter-ta analyzer.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use charter_core::{Candle, Timeframe};
use charter_ta::{
    AnalyzerConfig, DefaultAnalyzer, Analyzer, MultiTimeframeAnalyzer, TimeframeConfig,
    TimeframeData, FeatureExtractor, aggregate_candles,
};

fn make_candle(time: f64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Candle {
    Candle::new(time, open, high, low, close, volume)
}

fn generate_random_candles(count: usize) -> Vec<Candle> {
    let mut candles = Vec::with_capacity(count);
    let mut price = 100.0_f32;

    for i in 0..count {
        // Create a somewhat realistic price movement with trends
        let trend = (i as f32 * 0.01).sin() * 10.0;
        let volatility = (i as f32 * 0.1).sin() * 2.0;

        let open = price;
        let change = trend * 0.01 + volatility;
        let close = (open + change).max(1.0);
        let high = open.max(close) + (volatility.abs() * 0.5);
        let low = open.min(close) - (volatility.abs() * 0.5).max(0.1);

        candles.push(make_candle(
            i as f64 * 60.0, // 1-minute intervals
            open,
            high.max(open).max(close),
            low.min(open).min(close),
            close,
            100.0 + (i % 100) as f32,
        ));

        price = close;
    }

    candles
}

fn bench_reverse_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_pass");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let candles = generate_random_candles(*size);
        let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 3, 0.001)]);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &candles, |b, candles| {
            b.iter(|| {
                let mut analyzer = DefaultAnalyzer::new(config.clone());
                analyzer.update(0, black_box(candles), 100.0)
            });
        });
    }

    group.finish();
}

fn bench_level_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_queries");

    // Setup: create analyzer with levels
    let candles = generate_random_candles(1000);
    let config = AnalyzerConfig::new(vec![TimeframeConfig::new(Timeframe::Hour1, 2, 0.001)]);
    let mut analyzer = DefaultAnalyzer::new(config);
    analyzer.update(0, &candles, 100.0);

    let state = analyzer.state();
    let tf_state = state.get_timeframe(0).unwrap();

    group.bench_function("closest_resistance", |b| {
        b.iter(|| {
            tf_state.level_index.closest_resistance_above(black_box(100.0))
        });
    });

    group.bench_function("closest_support", |b| {
        b.iter(|| {
            tf_state.level_index.closest_support_below(black_box(100.0))
        });
    });

    group.bench_function("closest_n_resistance_3", |b| {
        b.iter(|| {
            tf_state.level_index.closest_n_resistance_above(black_box(100.0), 3)
        });
    });

    group.bench_function("closest_n_support_3", |b| {
        b.iter(|| {
            tf_state.level_index.closest_n_support_below(black_box(100.0), 3)
        });
    });

    group.finish();
}

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");

    // Setup: create analyzer with data
    let candles = generate_random_candles(1000);
    let config = AnalyzerConfig::new(vec![
        TimeframeConfig::new(Timeframe::Hour1, 2, 0.001),
        TimeframeConfig::new(Timeframe::Min30, 2, 0.001),
    ]);
    let mut analyzer = DefaultAnalyzer::new(config);
    analyzer.update(0, &candles, 100.0);
    analyzer.update(1, &candles, 100.0);

    group.bench_function("extract_features", |b| {
        b.iter(|| {
            analyzer.extract_features()
        });
    });

    group.finish();
}

fn bench_candle_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("candle_aggregation");

    for size in [100, 500, 1000, 5000].iter() {
        let candles = generate_random_candles(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("1m_to_1h", size),
            &candles,
            |b, candles| {
                b.iter(|| aggregate_candles(black_box(candles), Timeframe::Hour1));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("1m_to_5m", size),
            &candles,
            |b, candles| {
                b.iter(|| aggregate_candles(black_box(candles), Timeframe::Min5));
            },
        );
    }

    group.finish();
}

fn bench_multi_timeframe(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_timeframe");

    let candles = generate_random_candles(1000);
    let h1_candles = aggregate_candles(&candles, Timeframe::Hour1);
    let m30_candles = aggregate_candles(&candles, Timeframe::Min30);

    let config = AnalyzerConfig::new(vec![
        TimeframeConfig::new(Timeframe::Hour1, 2, 0.001),
        TimeframeConfig::new(Timeframe::Min30, 2, 0.001),
    ]);

    group.bench_function("update_two_timeframes", |b| {
        b.iter(|| {
            let mut analyzer = MultiTimeframeAnalyzer::new(config.clone());
            let data = vec![
                TimeframeData {
                    timeframe_idx: 0,
                    candles: black_box(&h1_candles),
                    current_price: 100.0,
                },
                TimeframeData {
                    timeframe_idx: 1,
                    candles: black_box(&m30_candles),
                    current_price: 100.0,
                },
            ];
            analyzer.update(&data)
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_reverse_pass,
    bench_level_queries,
    bench_feature_extraction,
    bench_candle_aggregation,
    bench_multi_timeframe,
);
criterion_main!(benches);
