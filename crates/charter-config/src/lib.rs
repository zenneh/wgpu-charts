//! Configuration management for charter.
//!
//! Loads configuration from TOML files with support for per-timeframe TA parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Configuration errors.
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    ReadError(#[from] std::io::Error),
    #[error("Failed to parse config file: {0}")]
    ParseError(#[from] toml::de::Error),
}

/// Root configuration structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub general: GeneralConfig,
    pub api: ApiConfig,
    pub ta: TaConfig,
    pub sync: SyncConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            api: ApiConfig::default(),
            ta: TaConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}

impl Config {
    /// Load configuration from a file path.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from default locations.
    ///
    /// Searches in order:
    /// 1. `./config.toml`
    /// 2. `~/.config/charter/config.toml`
    ///
    /// Returns default config if no file found.
    pub fn load_default() -> Self {
        // Try current directory first
        if let Ok(config) = Self::load("config.toml") {
            return config;
        }

        // Try user config directory
        if let Some(config_dir) = dirs::config_dir() {
            let config_path = config_dir.join("charter").join("config.toml");
            if let Ok(config) = Self::load(&config_path) {
                return config;
            }
        }

        // Return defaults
        Self::default()
    }

    /// Save configuration to a file path.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Get the default config file path.
    pub fn default_path() -> PathBuf {
        PathBuf::from("config.toml")
    }

    /// Get TA analysis config for a specific timeframe.
    /// Falls back to default if timeframe not configured.
    pub fn ta_analysis_for_timeframe(&self, timeframe: &str) -> TaAnalysisConfig {
        self.ta.timeframes
            .get(timeframe)
            .map(|tf| self.ta.default.merge(tf))
            .unwrap_or_else(|| self.ta.default.clone())
    }
}

/// General application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GeneralConfig {
    /// Default trading symbol to load on startup.
    pub default_symbol: String,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            default_symbol: "BTCUSDT".to_string(),
        }
    }
}

/// API configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ApiConfig {
    /// MEXC WebSocket URL.
    pub ws_url: String,
    /// Number of days of historical data to fetch.
    pub history_days: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            ws_url: "wss://wbs-api.mexc.com/ws".to_string(),
            history_days: 90,
        }
    }
}

/// Sync configuration for historical data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    /// Whether sync is enabled by default on startup.
    pub enabled: bool,
    /// Path to the DuckDB database file.
    /// Defaults to ~/.local/share/charter/candles.duckdb
    pub db_path: Option<PathBuf>,
    /// Delay between batch fetches in milliseconds.
    pub batch_delay_ms: u64,
    /// Number of days of historical data to sync.
    pub sync_days: u32,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            db_path: None, // Will use default path
            batch_delay_ms: 100,
            sync_days: 365, // 1 year by default
        }
    }
}

impl SyncConfig {
    /// Get the database path, using default if not specified.
    pub fn get_db_path(&self) -> PathBuf {
        self.db_path.clone().unwrap_or_else(|| {
            dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("charter")
                .join("candles.duckdb")
        })
    }
}

/// Technical Analysis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaConfig {
    /// Default TA analysis parameters.
    pub default: TaAnalysisConfig,
    /// Per-timeframe overrides.
    #[serde(default)]
    pub timeframes: HashMap<String, TaAnalysisOverride>,
    /// Display settings.
    pub display: TaDisplayConfig,
}

impl Default for TaConfig {
    fn default() -> Self {
        let mut timeframes = HashMap::new();

        // Set sensible defaults per timeframe
        timeframes.insert("1m".to_string(), TaAnalysisOverride {
            min_range_candles: Some(3),
            ..Default::default()
        });
        timeframes.insert("3m".to_string(), TaAnalysisOverride {
            min_range_candles: Some(3),
            ..Default::default()
        });
        timeframes.insert("5m".to_string(), TaAnalysisOverride {
            min_range_candles: Some(3),
            ..Default::default()
        });
        timeframes.insert("30m".to_string(), TaAnalysisOverride {
            min_range_candles: Some(4),
            ..Default::default()
        });
        timeframes.insert("1h".to_string(), TaAnalysisOverride {
            min_range_candles: Some(4),
            ..Default::default()
        });
        timeframes.insert("3h".to_string(), TaAnalysisOverride {
            min_range_candles: Some(4),
            ..Default::default()
        });
        timeframes.insert("5h".to_string(), TaAnalysisOverride {
            min_range_candles: Some(4),
            ..Default::default()
        });
        timeframes.insert("10h".to_string(), TaAnalysisOverride {
            min_range_candles: Some(5),
            ..Default::default()
        });
        timeframes.insert("1d".to_string(), TaAnalysisOverride {
            min_range_candles: Some(5),
            ..Default::default()
        });
        timeframes.insert("1w".to_string(), TaAnalysisOverride {
            min_range_candles: Some(5),
            ..Default::default()
        });
        timeframes.insert("3w".to_string(), TaAnalysisOverride {
            min_range_candles: Some(5),
            ..Default::default()
        });
        timeframes.insert("1M".to_string(), TaAnalysisOverride {
            min_range_candles: Some(5),
            ..Default::default()
        });

        Self {
            default: TaAnalysisConfig::default(),
            timeframes,
            display: TaDisplayConfig::default(),
        }
    }
}

/// TA analysis parameters (full config with all fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaAnalysisConfig {
    /// Threshold for doji detection (body_ratio < threshold = doji).
    pub doji_threshold: f32,
    /// Minimum candles required to form a valid range.
    pub min_range_candles: usize,
    /// Tolerance for level interactions (in price units).
    pub level_tolerance: f32,
    /// Whether to create greedy hold levels.
    pub create_greedy_levels: bool,
}

impl Default for TaAnalysisConfig {
    fn default() -> Self {
        Self {
            doji_threshold: 0.001,
            min_range_candles: 3,
            level_tolerance: 0.0,
            create_greedy_levels: true,
        }
    }
}

impl TaAnalysisConfig {
    /// Merge with an override, using override values where present.
    pub fn merge(&self, override_config: &TaAnalysisOverride) -> Self {
        Self {
            doji_threshold: override_config.doji_threshold.unwrap_or(self.doji_threshold),
            min_range_candles: override_config.min_range_candles.unwrap_or(self.min_range_candles),
            level_tolerance: override_config.level_tolerance.unwrap_or(self.level_tolerance),
            create_greedy_levels: override_config.create_greedy_levels.unwrap_or(self.create_greedy_levels),
        }
    }
}

/// TA analysis override (all fields optional for partial overrides).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TaAnalysisOverride {
    pub doji_threshold: Option<f32>,
    pub min_range_candles: Option<usize>,
    pub level_tolerance: Option<f32>,
    pub create_greedy_levels: Option<bool>,
}

/// TA display configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TaDisplayConfig {
    /// Show TA overlay on startup.
    pub show_ta: bool,
    /// Show range underlines.
    pub show_ranges: bool,
    /// Show hold levels.
    pub show_hold_levels: bool,
    /// Show greedy levels.
    pub show_greedy_levels: bool,
    /// Show active levels.
    pub show_active_levels: bool,
    /// Show hit levels (touched but not broken).
    pub show_hit_levels: bool,
    /// Show broken levels.
    pub show_broken_levels: bool,
    /// Show trendlines.
    pub show_trends: bool,
    /// Show active trends.
    pub show_active_trends: bool,
    /// Show hit trends.
    pub show_hit_trends: bool,
    /// Show broken trends.
    pub show_broken_trends: bool,
}

impl Default for TaDisplayConfig {
    fn default() -> Self {
        Self {
            show_ta: false,
            show_ranges: true,
            show_hold_levels: true,
            show_greedy_levels: false,
            show_active_levels: true,
            show_hit_levels: true,
            show_broken_levels: false,
            show_trends: true,
            show_active_trends: true,
            show_hit_trends: true,
            show_broken_trends: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.general.default_symbol, "BTCUSDT");
        assert_eq!(config.ta.default.min_range_candles, 3);
    }

    #[test]
    fn test_timeframe_override() {
        let config = Config::default();

        // 1m should have min_range_candles = 3
        let ta_1m = config.ta_analysis_for_timeframe("1m");
        assert_eq!(ta_1m.min_range_candles, 3);

        // 1d should have min_range_candles = 5
        let ta_1d = config.ta_analysis_for_timeframe("1d");
        assert_eq!(ta_1d.min_range_candles, 5);

        // Unknown timeframe should use default
        let ta_unknown = config.ta_analysis_for_timeframe("unknown");
        assert_eq!(ta_unknown.min_range_candles, 3);
    }

    #[test]
    fn test_parse_toml() {
        let toml = r#"
[general]
default_symbol = "ETHUSDT"

[ta.default]
min_range_candles = 4

[ta.timeframes.1m]
min_range_candles = 2

[ta.display]
show_ta = true
"#;
        let config: Config = toml::from_str(toml).unwrap();
        assert_eq!(config.general.default_symbol, "ETHUSDT");
        assert_eq!(config.ta.default.min_range_candles, 4);
        assert_eq!(config.ta_analysis_for_timeframe("1m").min_range_candles, 2);
        assert!(config.ta.display.show_ta);
    }
}
