//! Signal generation from C-Mamba predictions.

use ndarray::{Array1, Array2};

/// Trading signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    Long,
    Short,
    Hold,
}

/// Signal generator from C-Mamba predictions.
#[derive(Debug)]
pub struct SignalGenerator {
    pub long_threshold: f64,
    pub short_threshold: f64,
    pub top_n: usize,
    pub bottom_n: usize,
}

impl Default for SignalGenerator {
    fn default() -> Self {
        Self {
            long_threshold: 0.0,
            short_threshold: 0.0,
            top_n: 3,
            bottom_n: 0,
        }
    }
}

impl SignalGenerator {
    pub fn new(long_threshold: f64, short_threshold: f64, top_n: usize, bottom_n: usize) -> Self {
        Self {
            long_threshold,
            short_threshold,
            top_n,
            bottom_n,
        }
    }

    /// Generate signals from predictions.
    ///
    /// # Arguments
    /// * `predictions` - Predicted returns [pred_len, n_channels]
    ///
    /// # Returns
    /// Vector of signals for each channel
    pub fn generate(&self, predictions: &Array2<f64>) -> Vec<Signal> {
        let n_channels = predictions.ncols();

        // Sum predictions across prediction horizon
        let expected_returns: Vec<f64> = (0..n_channels)
            .map(|i| predictions.column(i).sum())
            .collect();

        // Rank channels
        let mut ranked: Vec<(usize, f64)> = expected_returns
            .iter()
            .enumerate()
            .map(|(i, &r)| (i, r))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Generate signals
        let mut signals = vec![Signal::Hold; n_channels];

        // Long top N above threshold
        for i in 0..self.top_n.min(n_channels) {
            let (idx, ret) = ranked[i];
            if ret > self.long_threshold {
                signals[idx] = Signal::Long;
            }
        }

        // Short bottom N below threshold
        for i in 0..self.bottom_n.min(n_channels) {
            let (idx, ret) = ranked[n_channels - 1 - i];
            if ret < self.short_threshold {
                signals[idx] = Signal::Short;
            }
        }

        signals
    }

    /// Generate position weights from predictions.
    pub fn generate_weights(&self, predictions: &Array2<f64>) -> Array1<f64> {
        let n_channels = predictions.ncols();
        let signals = self.generate(predictions);

        let mut weights = Array1::zeros(n_channels);

        // Count positions
        let long_count = signals.iter().filter(|&&s| s == Signal::Long).count();
        let short_count = signals.iter().filter(|&&s| s == Signal::Short).count();

        // Assign equal weights
        for (i, signal) in signals.iter().enumerate() {
            match signal {
                Signal::Long if long_count > 0 => {
                    weights[i] = 1.0 / long_count as f64;
                }
                Signal::Short if short_count > 0 => {
                    weights[i] = -1.0 / short_count as f64;
                }
                _ => {}
            }
        }

        weights
    }
}

/// Analyze correlation regime from channel attention.
#[derive(Debug)]
pub struct CorrelationAnalyzer;

impl CorrelationAnalyzer {
    /// Detect correlation regime.
    ///
    /// Returns the average correlation (high values indicate stress regime).
    pub fn detect_regime(correlation_matrix: &Array2<f64>) -> CorrelationRegime {
        let n = correlation_matrix.nrows();
        if n < 2 {
            return CorrelationRegime::Unknown;
        }

        // Calculate average off-diagonal correlation
        let mut sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                sum += correlation_matrix[[i, j]];
                count += 1;
            }
        }

        let avg_corr = if count > 0 { sum / count as f64 } else { 0.0 };

        if avg_corr > 0.8 {
            CorrelationRegime::HighStress
        } else if avg_corr > 0.5 {
            CorrelationRegime::Elevated
        } else if avg_corr > 0.2 {
            CorrelationRegime::Normal
        } else {
            CorrelationRegime::Dispersed
        }
    }

    /// Find top correlated pairs.
    pub fn top_correlations(
        correlation_matrix: &Array2<f64>,
        symbols: &[String],
        top_n: usize,
    ) -> Vec<(String, String, f64)> {
        let n = correlation_matrix.nrows();
        let mut pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j, correlation_matrix[[i, j]]));
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        pairs
            .into_iter()
            .take(top_n)
            .map(|(i, j, corr)| (symbols[i].clone(), symbols[j].clone(), corr))
            .collect()
    }
}

/// Correlation regime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CorrelationRegime {
    HighStress,  // > 0.8 average correlation
    Elevated,    // 0.5 - 0.8
    Normal,      // 0.2 - 0.5
    Dispersed,   // < 0.2
    Unknown,
}

impl CorrelationRegime {
    pub fn description(&self) -> &'static str {
        match self {
            Self::HighStress => "High stress - assets moving together, reduce risk",
            Self::Elevated => "Elevated correlation - be cautious",
            Self::Normal => "Normal conditions - diversification works",
            Self::Dispersed => "Dispersed - strong diversification potential",
            Self::Unknown => "Unknown regime",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_signal_generation() {
        let generator = SignalGenerator::default();

        let predictions = array![
            [0.01, 0.02, -0.01, 0.005, -0.02],
            [0.01, 0.015, -0.015, 0.003, -0.01],
        ];

        let signals = generator.generate(&predictions);

        assert_eq!(signals.len(), 5);
        // Top 3 should be Long (indices 1, 0, 3)
        assert_eq!(signals[1], Signal::Long);
        assert_eq!(signals[0], Signal::Long);
    }

    #[test]
    fn test_correlation_regime() {
        let high_corr = Array2::from_shape_fn((5, 5), |(i, j)| {
            if i == j { 1.0 } else { 0.9 }
        });

        let regime = CorrelationAnalyzer::detect_regime(&high_corr);
        assert_eq!(regime, CorrelationRegime::HighStress);
    }
}
