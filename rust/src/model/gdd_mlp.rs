//! GDD-MLP: Global-local Dual-branch MLP
//!
//! Captures cross-channel dependencies through parallel global and local branches.

use ndarray::{Array1, Array2, Array3, Array4};
use rand::Rng;

/// Global-local Dual-branch MLP.
#[derive(Debug)]
pub struct GddMlp {
    n_channels: usize,
    d_model: usize,
    expansion: usize,
    // Global branch
    w_global_1: Array2<f64>,
    w_global_2: Array2<f64>,
    // Local branch
    w_local_1: Array2<f64>,
    w_local_2: Array2<f64>,
    // Channel attention
    w_attn: Array2<f64>,
}

impl GddMlp {
    pub fn new(n_channels: usize, d_model: usize, expansion: usize) -> Self {
        let mut rng = rand::thread_rng();

        let w_global_1 = Array2::from_shape_fn(
            (n_channels, n_channels * expansion),
            |_| rng.gen::<f64>() * 0.01,
        );
        let w_global_2 = Array2::from_shape_fn(
            (n_channels * expansion, n_channels),
            |_| rng.gen::<f64>() * 0.01,
        );

        let w_local_1 = Array2::from_shape_fn(
            (d_model, d_model * expansion),
            |_| rng.gen::<f64>() * 0.01,
        );
        let w_local_2 = Array2::from_shape_fn(
            (d_model * expansion, d_model),
            |_| rng.gen::<f64>() * 0.01,
        );

        let w_attn = Array2::from_shape_fn(
            (n_channels, n_channels),
            |_| rng.gen::<f64>() * 0.01,
        );

        Self {
            n_channels,
            d_model,
            expansion,
            w_global_1,
            w_global_2,
            w_local_1,
            w_local_2,
            w_attn,
        }
    }

    /// Forward pass through GDD-MLP.
    ///
    /// Input: [batch, n_channels, n_patches, d_model]
    /// Output: ([batch, n_channels, n_patches, d_model], Optional attention)
    pub fn forward(
        &self,
        x: &Array4<f64>,
        return_attention: bool,
    ) -> (Array4<f64>, Option<Array3<f64>>) {
        let (batch_size, n_channels, n_patches, d_model) = x.dim();

        // Global branch: aggregate and process channels
        let global_feat = self.global_pool(x);  // [batch, n_channels]
        let global_out = self.mlp_global(&global_feat);  // [batch, n_channels]

        // Local branch: process each position
        let local_out = self.mlp_local(x);  // [batch, n_channels, n_patches, d_model]

        // Channel attention
        let attention = self.compute_attention(&global_feat);  // [batch, n_channels]

        // Combine global and local
        let mut output = Array4::zeros((batch_size, n_channels, n_patches, d_model));
        for b in 0..batch_size {
            for c in 0..n_channels {
                for p in 0..n_patches {
                    for d in 0..d_model {
                        output[[b, c, p, d]] = local_out[[b, c, p, d]] +
                            global_out[[b, c]] * attention[[b, c]];
                    }
                }
            }
        }

        // Compute attention matrix if requested
        let attn_matrix = if return_attention {
            Some(self.compute_attention_matrix(&attention))
        } else {
            None
        };

        (output, attn_matrix)
    }

    fn global_pool(&self, x: &Array4<f64>) -> Array2<f64> {
        let (batch_size, n_channels, n_patches, d_model) = x.dim();
        let mut output = Array2::zeros((batch_size, n_channels));

        for b in 0..batch_size {
            for c in 0..n_channels {
                let mut sum = 0.0;
                for p in 0..n_patches {
                    for d in 0..d_model {
                        sum += x[[b, c, p, d]];
                    }
                }
                output[[b, c]] = sum / (n_patches * d_model) as f64;
            }
        }

        output
    }

    fn mlp_global(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch_size, n_channels) = x.dim();
        let hidden_dim = n_channels * self.expansion;

        // First layer with GELU
        let mut hidden = Array2::zeros((batch_size, hidden_dim));
        for b in 0..batch_size {
            for h in 0..hidden_dim {
                let mut sum = 0.0;
                for c in 0..n_channels {
                    sum += x[[b, c]] * self.w_global_1[[c, h]];
                }
                hidden[[b, h]] = gelu(sum);
            }
        }

        // Second layer
        let mut output = Array2::zeros((batch_size, n_channels));
        for b in 0..batch_size {
            for c in 0..n_channels {
                let mut sum = 0.0;
                for h in 0..hidden_dim {
                    sum += hidden[[b, h]] * self.w_global_2[[h, c]];
                }
                output[[b, c]] = sum;
            }
        }

        output
    }

    fn mlp_local(&self, x: &Array4<f64>) -> Array4<f64> {
        let (batch_size, n_channels, n_patches, d_model) = x.dim();
        let hidden_dim = d_model * self.expansion;

        let mut output = Array4::zeros((batch_size, n_channels, n_patches, d_model));

        for b in 0..batch_size {
            for c in 0..n_channels {
                for p in 0..n_patches {
                    // First layer with GELU
                    let mut hidden = Array1::zeros(hidden_dim);
                    for h in 0..hidden_dim {
                        let mut sum = 0.0;
                        for d in 0..d_model {
                            sum += x[[b, c, p, d]] * self.w_local_1[[d, h]];
                        }
                        hidden[h] = gelu(sum);
                    }

                    // Second layer
                    for d in 0..d_model {
                        let mut sum = 0.0;
                        for h in 0..hidden_dim {
                            sum += hidden[h] * self.w_local_2[[h, d]];
                        }
                        output[[b, c, p, d]] = sum;
                    }
                }
            }
        }

        output
    }

    fn compute_attention(&self, x: &Array2<f64>) -> Array2<f64> {
        let (batch_size, n_channels) = x.dim();
        let mut output = Array2::zeros((batch_size, n_channels));

        for b in 0..batch_size {
            // Compute attention scores
            let mut scores = Array1::zeros(n_channels);
            for c in 0..n_channels {
                let mut sum = 0.0;
                for c2 in 0..n_channels {
                    sum += x[[b, c2]] * self.w_attn[[c2, c]];
                }
                scores[c] = sum;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut exp_sum = 0.0;
            for c in 0..n_channels {
                scores[c] = (scores[c] - max_score).exp();
                exp_sum += scores[c];
            }
            for c in 0..n_channels {
                output[[b, c]] = scores[c] / exp_sum;
            }
        }

        output
    }

    fn compute_attention_matrix(&self, attention: &Array2<f64>) -> Array3<f64> {
        let (batch_size, n_channels) = attention.dim();
        let mut matrix = Array3::zeros((batch_size, n_channels, n_channels));

        for b in 0..batch_size {
            for i in 0..n_channels {
                for j in 0..n_channels {
                    matrix[[b, i, j]] = attention[[b, i]] * attention[[b, j]];
                }
            }
        }

        matrix
    }
}

/// GELU activation function.
fn gelu(x: f64) -> f64 {
    let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_pi * (x + 0.044715 * x.powi(3))).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdd_mlp() {
        let mlp = GddMlp::new(10, 64, 2);
        let x = Array4::zeros((1, 10, 5, 64));
        let (out, attn) = mlp.forward(&x, true);

        assert_eq!(out.dim(), (1, 10, 5, 64));
        assert!(attn.is_some());
        assert_eq!(attn.unwrap().dim(), (1, 10, 10));
    }
}
