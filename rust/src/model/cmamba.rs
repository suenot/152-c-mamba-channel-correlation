//! C-Mamba: Channel Correlation Enhanced State Space Model
//!
//! Main model implementation for multivariate time series forecasting
//! with cross-channel dependency modeling.

use ndarray::{Array1, Array2, Array3, Array4, Axis};
use rand::Rng;
use anyhow::Result;

use super::mamba::MambaBlock;
use super::gdd_mlp::GddMlp;

/// Configuration for C-Mamba model.
#[derive(Debug, Clone)]
pub struct CMambaConfig {
    /// Input sequence length
    pub seq_len: usize,
    /// Prediction length
    pub pred_len: usize,
    /// Number of channels (assets)
    pub n_channels: usize,
    /// Model dimension
    pub d_model: usize,
    /// SSM state dimension
    pub d_state: usize,
    /// Convolution kernel size
    pub d_conv: usize,
    /// Number of M-Mamba layers
    pub n_layers: usize,
    /// Patch length
    pub patch_len: usize,
    /// Patch stride
    pub stride: usize,
    /// MLP expansion factor
    pub expansion_factor: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Channel mixup alpha
    pub channel_mixup_alpha: f64,
}

impl Default for CMambaConfig {
    fn default() -> Self {
        Self {
            seq_len: 60,
            pred_len: 5,
            n_channels: 10,
            d_model: 64,
            d_state: 16,
            d_conv: 4,
            n_layers: 2,
            patch_len: 12,
            stride: 6,
            expansion_factor: 2,
            dropout: 0.1,
            channel_mixup_alpha: 0.2,
        }
    }
}

/// Output from C-Mamba model.
#[derive(Debug)]
pub struct CMambaOutput {
    /// Predictions [batch, pred_len, n_channels]
    pub predictions: Array3<f64>,
    /// Channel attention matrix (optional)
    pub channel_attention: Option<Array3<f64>>,
}

/// Patch embedding layer.
#[derive(Debug)]
pub struct PatchEmbedding {
    patch_len: usize,
    stride: usize,
    d_model: usize,
    w_embed: Array2<f64>,
}

impl PatchEmbedding {
    pub fn new(patch_len: usize, stride: usize, d_model: usize) -> Self {
        let mut rng = rand::thread_rng();
        let w_embed = Array2::from_shape_fn((patch_len, d_model), |_| {
            rng.gen::<f64>() * 0.01
        });

        Self {
            patch_len,
            stride,
            d_model,
            w_embed,
        }
    }

    /// Create patch embeddings.
    ///
    /// Input: [batch, seq_len, n_channels]
    /// Output: [batch, n_channels, n_patches, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> (Array4<f64>, usize) {
        let (batch_size, seq_len, n_channels) = x.dim();
        let n_patches = (seq_len - self.patch_len) / self.stride + 1;

        let mut output = Array4::zeros((batch_size, n_channels, n_patches, self.d_model));

        for b in 0..batch_size {
            for c in 0..n_channels {
                for p in 0..n_patches {
                    let start = p * self.stride;
                    let end = start + self.patch_len;

                    // Extract patch and project
                    for (i, t) in (start..end).enumerate() {
                        let val = x[[b, t, c]];
                        for d in 0..self.d_model {
                            output[[b, c, p, d]] += val * self.w_embed[[i, d]];
                        }
                    }
                }
            }
        }

        (output, n_patches)
    }
}

/// Channel Mixup for data augmentation.
#[derive(Debug)]
pub struct ChannelMixup {
    alpha: f64,
}

impl ChannelMixup {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }

    /// Apply channel mixup during training.
    pub fn apply(&self, x: &Array3<f64>, training: bool) -> Array3<f64> {
        if !training || self.alpha <= 0.0 {
            return x.clone();
        }

        let mut rng = rand::thread_rng();
        let (batch_size, seq_len, n_channels) = x.dim();

        // Sample lambda from Beta distribution (simplified to uniform for demo)
        let lambda = rng.gen::<f64>() * self.alpha;

        // Random permutation
        let mut perm: Vec<usize> = (0..n_channels).collect();
        for i in (1..n_channels).rev() {
            let j = rng.gen_range(0..=i);
            perm.swap(i, j);
        }

        // Mix channels
        let mut mixed = Array3::zeros((batch_size, seq_len, n_channels));
        for b in 0..batch_size {
            for t in 0..seq_len {
                for c in 0..n_channels {
                    mixed[[b, t, c]] = lambda * x[[b, t, c]] +
                        (1.0 - lambda) * x[[b, t, perm[c]]];
                }
            }
        }

        mixed
    }
}

/// C-Mamba: Channel Correlation Enhanced State Space Model.
///
/// Main model for multivariate time series forecasting with
/// cross-channel dependency modeling.
#[derive(Debug)]
pub struct CMamba {
    pub config: CMambaConfig,
    patch_embed: PatchEmbedding,
    mamba_blocks: Vec<MambaBlock>,
    gdd_mlp: GddMlp,
    channel_mixup: ChannelMixup,
    w_out: Array2<f64>,
    n_patches: usize,
    is_training: bool,
}

impl CMamba {
    /// Create a new C-Mamba model.
    pub fn new(config: CMambaConfig) -> Self {
        let n_patches = (config.seq_len - config.patch_len) / config.stride + 1;

        let patch_embed = PatchEmbedding::new(
            config.patch_len,
            config.stride,
            config.d_model,
        );

        let mamba_blocks: Vec<MambaBlock> = (0..config.n_layers)
            .map(|_| MambaBlock::new(config.d_model, config.d_state, config.d_conv))
            .collect();

        let gdd_mlp = GddMlp::new(
            config.n_channels,
            config.d_model,
            config.expansion_factor,
        );

        let channel_mixup = ChannelMixup::new(config.channel_mixup_alpha);

        let mut rng = rand::thread_rng();
        let w_out = Array2::from_shape_fn(
            (config.d_model * n_patches, config.pred_len),
            |_| rng.gen::<f64>() * 0.01,
        );

        Self {
            config,
            patch_embed,
            mamba_blocks,
            gdd_mlp,
            channel_mixup,
            w_out,
            n_patches,
            is_training: false,
        }
    }

    /// Set model to training mode.
    pub fn train(&mut self) {
        self.is_training = true;
    }

    /// Set model to evaluation mode.
    pub fn eval(&mut self) {
        self.is_training = false;
    }

    /// Forward pass through C-Mamba.
    ///
    /// Input: [batch, seq_len, n_channels]
    /// Output: CMambaOutput with predictions [batch, pred_len, n_channels]
    pub fn forward(&self, x: &Array3<f64>) -> CMambaOutput {
        let (batch_size, _seq_len, n_channels) = x.dim();

        // Channel mixup (training only)
        let x = if self.is_training {
            self.channel_mixup.apply(x, true)
        } else {
            x.clone()
        };

        // Patch embedding: [batch, n_channels, n_patches, d_model]
        let (mut x, _n_patches) = self.patch_embed.forward(&x);

        // M-Mamba blocks (process each channel independently)
        for block in &self.mamba_blocks {
            x = self.apply_mamba_block(block, &x);
        }

        // GDD-MLP for cross-channel dependencies
        let (x, attention) = self.gdd_mlp.forward(&x, true);

        // Flatten and project
        // [batch, n_channels, n_patches * d_model]
        let x_flat = x.into_shape((batch_size, n_channels, self.n_patches * self.config.d_model))
            .unwrap();

        // Output projection: [batch, n_channels, pred_len]
        let mut predictions = Array3::zeros((batch_size, n_channels, self.config.pred_len));
        for b in 0..batch_size {
            for c in 0..n_channels {
                for p in 0..self.config.pred_len {
                    for d in 0..(self.n_patches * self.config.d_model) {
                        predictions[[b, c, p]] += x_flat[[b, c, d]] * self.w_out[[d, p]];
                    }
                }
            }
        }

        // Transpose to [batch, pred_len, n_channels]
        let predictions = predictions.permuted_axes([0, 2, 1]).to_owned();

        CMambaOutput {
            predictions,
            channel_attention: attention,
        }
    }

    fn apply_mamba_block(&self, block: &MambaBlock, x: &Array4<f64>) -> Array4<f64> {
        let (batch_size, n_channels, n_patches, d_model) = x.dim();

        // Reshape to [batch * n_channels, n_patches, d_model]
        let x_reshaped = x.clone()
            .into_shape((batch_size * n_channels, n_patches, d_model))
            .unwrap();

        // Apply Mamba block
        let out = block.forward(&x_reshaped);

        // Reshape back
        out.into_shape((batch_size, n_channels, n_patches, d_model)).unwrap()
    }

    /// Generate predictions.
    pub fn predict(&self, x: &Array3<f64>) -> Result<Array3<f64>> {
        let output = self.forward(x);
        Ok(output.predictions)
    }

    /// Get channel correlation matrix.
    pub fn get_channel_correlation(&self, x: &Array3<f64>) -> Option<Array3<f64>> {
        let output = self.forward(x);
        output.channel_attention
    }

    /// Analyze dependencies.
    pub fn analyze_dependencies(&self, x: &Array3<f64>) -> AnalysisResult {
        let output = self.forward(x);

        let correlation_matrix = output.channel_attention.as_ref().map(|a| {
            // Average over batch
            a.mean_axis(Axis(0)).unwrap()
        });

        AnalysisResult {
            correlation_matrix,
            predictions: output.predictions,
        }
    }
}

/// Result from dependency analysis.
#[derive(Debug)]
pub struct AnalysisResult {
    pub correlation_matrix: Option<Array2<f64>>,
    pub predictions: Array3<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmamba_creation() {
        let config = CMambaConfig::default();
        let model = CMamba::new(config);

        assert_eq!(model.config.seq_len, 60);
        assert_eq!(model.config.pred_len, 5);
        assert_eq!(model.config.n_channels, 10);
    }

    #[test]
    fn test_patch_embedding() {
        let embed = PatchEmbedding::new(12, 6, 64);

        let x = Array3::zeros((1, 60, 10));
        let (out, n_patches) = embed.forward(&x);

        assert_eq!(n_patches, 9);
        assert_eq!(out.dim(), (1, 10, 9, 64));
    }
}
