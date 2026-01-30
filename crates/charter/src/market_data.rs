//! CPU-side state for market data (volume profile & depth heatmap).

use charter_core::Candle;
use charter_data::TradeData;
use charter_render::{
    DepthHeatmapCellGpu, DepthHeatmapParams, VolumeProfileBucketGpu, VolumeProfileParams,
    MAX_DEPTH_LEVELS, MAX_VOLUME_PROFILE_BUCKETS,
};

/// Raw trade stored for re-bucketing when the view changes.
#[derive(Debug, Clone)]
struct RawTrade {
    price: f32,
    quantity: f32,
    is_buy: bool,
}

/// CPU-side market data state for volume profile and depth heatmap.
pub struct MarketDataState {
    // --- Volume profile ---
    /// Raw trades kept for re-bucketing when view changes.
    trades: Vec<RawTrade>,

    // --- Depth heatmap ---
    /// Latest depth snapshot: (price, bid_qty, ask_qty) for each level.
    latest_depth: Vec<(f32, f32, f32)>,
}

impl Default for MarketDataState {
    fn default() -> Self {
        Self {
            trades: Vec::new(),
            latest_depth: Vec::new(),
        }
    }
}

impl MarketDataState {
    /// Store a batch of trades for volume profile.
    pub fn process_trades(&mut self, trades: &[TradeData]) {
        for t in trades {
            if t.price > 0.0 && t.price.is_finite() && t.quantity > 0.0 && t.quantity.is_finite() {
                self.trades.push(RawTrade {
                    price: t.price,
                    quantity: t.quantity,
                    is_buy: t.is_buy,
                });
            }
        }
        // Cap stored trades to avoid unbounded growth
        const MAX_TRADES: usize = 500_000;
        if self.trades.len() > MAX_TRADES {
            let drain = self.trades.len() - MAX_TRADES;
            self.trades.drain(..drain);
        }
    }

    /// Process a depth snapshot — keeps only the latest.
    pub fn process_depth(
        &mut self,
        bids: &[(f32, f32)],
        asks: &[(f32, f32)],
        _timestamp: i64,
    ) {
        self.latest_depth.clear();

        // Store each bid level
        for &(price, qty) in bids {
            if price > 0.0 && qty >= 0.0 {
                self.latest_depth.push((price, qty, 0.0));
            }
        }

        // Store each ask level
        for &(price, qty) in asks {
            if price > 0.0 && qty >= 0.0 {
                // Check if there's already a bid at this price
                if let Some(entry) = self.latest_depth.iter_mut().find(|(p, _, _)| (*p - price).abs() < 0.001) {
                    entry.2 = qty;
                } else {
                    self.latest_depth.push((price, 0.0, qty));
                }
            }
        }

        // Sort by price
        self.latest_depth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Build GPU data for volume profile.
    ///
    /// Uses raw trade data if available, otherwise falls back to candle data.
    /// Buckets span the visible price range [y_min, y_max].
    pub fn build_volume_profile_gpu(
        &self,
        x_right: f32,
        y_min: f32,
        y_max: f32,
        visible_x_width: f32,
        candles: &[Candle],
    ) -> (Vec<VolumeProfileBucketGpu>, VolumeProfileParams) {
        let price_range = y_max - y_min;
        let empty_params = VolumeProfileParams {
            bucket_count: 0,
            max_volume: 1.0,
            profile_width: 0.0,
            y_min,
            y_max,
            bucket_height: 1.0,
            x_right,
            visible: 0,
        };

        if price_range <= 0.0 {
            return (vec![], empty_params);
        }

        let bucket_count = MAX_VOLUME_PROFILE_BUCKETS.min(128);
        let bucket_size = price_range / bucket_count as f32;
        let mut buy_volumes = vec![0.0f32; bucket_count];
        let mut sell_volumes = vec![0.0f32; bucket_count];

        if !self.trades.is_empty() {
            // Build from tick-level trade data
            for trade in &self.trades {
                if trade.price < y_min || trade.price > y_max {
                    continue;
                }
                let idx = ((trade.price - y_min) / bucket_size).floor() as usize;
                if idx >= bucket_count {
                    continue;
                }
                if trade.is_buy {
                    buy_volumes[idx] += trade.quantity;
                } else {
                    sell_volumes[idx] += trade.quantity;
                }
            }
        } else if !candles.is_empty() {
            // Fallback: distribute candle volume across high-low range
            for candle in candles {
                let c_low = candle.low.max(y_min);
                let c_high = candle.high.min(y_max);
                if c_low >= c_high {
                    continue;
                }
                let first_bucket = ((c_low - y_min) / bucket_size).floor() as usize;
                let last_bucket = ((c_high - y_min) / bucket_size).floor() as usize;
                let span = (last_bucket - first_bucket + 1).max(1);
                let vol_per_bucket = candle.volume / span as f32;
                let is_bullish = candle.close >= candle.open;

                for idx in first_bucket..=last_bucket.min(bucket_count - 1) {
                    if is_bullish {
                        buy_volumes[idx] += vol_per_bucket;
                    } else {
                        sell_volumes[idx] += vol_per_bucket;
                    }
                }
            }
        } else {
            return (vec![], empty_params);
        }

        let mut max_vol = 0.0f32;
        for i in 0..bucket_count {
            let total = buy_volumes[i] + sell_volumes[i];
            if total > max_vol {
                max_vol = total;
            }
        }

        // Only emit non-empty buckets
        let buckets: Vec<VolumeProfileBucketGpu> = (0..bucket_count)
            .filter(|&i| buy_volumes[i] > 0.0 || sell_volumes[i] > 0.0)
            .map(|i| VolumeProfileBucketGpu {
                price: y_min + (i as f32 + 0.5) * bucket_size,
                buy_volume: buy_volumes[i],
                sell_volume: sell_volumes[i],
                _padding: 0.0,
            })
            .collect();

        let profile_width = visible_x_width * 0.15;

        let params = VolumeProfileParams {
            bucket_count: buckets.len() as u32,
            max_volume: max_vol.max(1.0),
            profile_width,
            y_min,
            y_max,
            bucket_height: bucket_size,
            x_right,
            visible: if buckets.is_empty() { 0 } else { 1 },
        };

        (buckets, params)
    }

    /// Build GPU data for depth sidebar from the latest depth snapshot.
    ///
    /// Renders a two-sided horizontal bar chart (bid bars left, ask bars right)
    /// pinned to the right edge of the chart. Each level renders at its exact
    /// price so individual order book levels are visible.
    pub fn build_depth_heatmap_gpu(
        &self,
        x_right: f32,
        visible_x_width: f32,
        y_min: f32,
        y_max: f32,
    ) -> (Vec<DepthHeatmapCellGpu>, DepthHeatmapParams) {
        let empty_params = DepthHeatmapParams {
            level_count: 0,
            _pad0: 0,
            max_quantity: 1.0,
            half_width: 1.0,
            _pad1: 0.0,
            _pad2: 0.0,
            x_center: 0.0,
            visible: 0,
        };

        if self.latest_depth.is_empty() || y_max <= y_min {
            return (vec![], empty_params);
        }

        // Filter depth levels to visible price range
        let visible: Vec<&(f32, f32, f32)> = self
            .latest_depth
            .iter()
            .filter(|(p, _, _)| *p >= y_min && *p <= y_max)
            .collect();

        if visible.is_empty() {
            return (vec![], empty_params);
        }

        // Each side of the sidebar = 5% of visible width
        let half_width = visible_x_width * 0.05;
        // Center divider: asks extend to x_right
        let x_center = x_right - half_width;

        // Compute bar height from the minimum gap between adjacent levels.
        // Enforce a minimum so bars are visible (~3px on a 1000px chart).
        // When levels are denser than this minimum, overlapping bars with alpha
        // blending create a natural intensity effect in the cluster.
        let price_range = y_max - y_min;
        let min_visible_height = price_range / 300.0;

        let bar_height = if visible.len() >= 2 {
            let mut min_gap = f32::MAX;
            for i in 1..visible.len() {
                let gap = visible[i].0 - visible[i - 1].0;
                if gap > 0.0 && gap < min_gap {
                    min_gap = gap;
                }
            }
            let gap_based = if min_gap == f32::MAX {
                price_range / 100.0
            } else {
                min_gap * 0.85
            };
            gap_based.max(min_visible_height)
        } else {
            price_range / 100.0
        };

        // Find max quantity for normalization (current snapshot only)
        let mut max_qty = 0.0f32;
        for &&(_, bid_q, ask_q) in &visible {
            max_qty = max_qty.max(bid_q).max(ask_q);
        }
        let max_qty = max_qty.max(0.001);

        // Emit one cell per depth level at its exact price
        let mut cells = Vec::new();
        for &&(price, bid_q, ask_q) in &visible {
            if bid_q > 0.0 || ask_q > 0.0 {
                cells.push(DepthHeatmapCellGpu {
                    price,
                    bar_height,
                    bid_quantity: bid_q,
                    ask_quantity: ask_q,
                });
                if cells.len() >= MAX_DEPTH_LEVELS {
                    break;
                }
            }
        }

        let params = DepthHeatmapParams {
            level_count: cells.len() as u32,
            _pad0: 0,
            max_quantity: max_qty,
            half_width,
            _pad1: 0.0,
            _pad2: 0.0,
            x_center,
            visible: if cells.is_empty() { 0 } else { 1 },
        };

        (cells, params)
    }

    /// Number of stored raw trades.
    pub fn trades_count(&self) -> usize {
        self.trades.len()
    }

    /// Check if there is any data to render.
    pub fn has_volume_data(&self) -> bool {
        !self.trades.is_empty()
    }

    pub fn has_depth_data(&self) -> bool {
        !self.latest_depth.is_empty()
    }

    /// Clear all market data state.
    pub fn clear(&mut self) {
        self.trades.clear();
        self.latest_depth.clear();
    }
}

/// GPU buffers for market data visualization.
pub struct MarketDataGpuBuffers {
    // Volume profile
    pub vp_bucket_buffer: wgpu::Buffer,
    pub vp_params_buffer: wgpu::Buffer,
    pub vp_bind_group: wgpu::BindGroup,
    pub vp_bucket_count: u32,
    // Depth heatmap
    pub dh_cell_buffer: wgpu::Buffer,
    pub dh_params_buffer: wgpu::Buffer,
    pub dh_bind_group: wgpu::BindGroup,
    pub dh_cell_count: u32,
}

impl MarketDataGpuBuffers {
    /// Create initial GPU buffers.
    pub fn new(
        device: &wgpu::Device,
        renderer: &charter_render::ChartRenderer,
    ) -> Self {
        let vp_bucket_buffer = renderer.volume_profile_pipeline.create_bucket_buffer(device);
        let vp_params_buffer = renderer.volume_profile_pipeline.create_params_buffer(device);
        let vp_bind_group = renderer.volume_profile_pipeline.create_bind_group(
            device,
            &vp_bucket_buffer,
            &vp_params_buffer,
        );

        let dh_cell_buffer = renderer.depth_heatmap_pipeline.create_cell_buffer(device);
        let dh_params_buffer = renderer.depth_heatmap_pipeline.create_params_buffer(device);
        let dh_bind_group = renderer.depth_heatmap_pipeline.create_bind_group(
            device,
            &dh_cell_buffer,
            &dh_params_buffer,
        );

        Self {
            vp_bucket_buffer,
            vp_params_buffer,
            vp_bind_group,
            vp_bucket_count: 0,
            dh_cell_buffer,
            dh_params_buffer,
            dh_bind_group,
            dh_cell_count: 0,
        }
    }
}
