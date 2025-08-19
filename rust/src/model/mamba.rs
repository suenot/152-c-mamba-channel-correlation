//! Mamba SSM: Selective State Space Model
//!
//! Implementation of the core Mamba SSM with bidirectional processing.

use ndarray::{Array1, Array2, Array3};
use rand::Rng;

/// Selective State Space Model.
#[derive(Debug)]
pub struct SelectiveSsm {
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    a: Array1<f64>,
    w_b: Array2<f64>,
    w_c: Array2<f64>,
    w_delta: Array2<f64>,
}

impl SelectiveSsm {
    pub fn new(d_model: usize, d_state: usize, d_conv: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize A with negative values (stability)
        let a = Array1::from_shape_fn(d_state, |_| -rng.gen::<f64>().abs());

        // Projection matrices
        let w_b = Array2::from_shape_fn((d_model, d_state), |_| rng.gen::<f64>() * 0.01);
        let w_c = Array2::from_shape_fn((d_state, d_model), |_| rng.gen::<f64>() * 0.01);
        let w_delta = Array2::from_shape_fn((d_model, 1), |_| rng.gen::<f64>() * 0.01);

        Self {
            d_model,
            d_state,
            d_conv,
            a,
            w_b,
            w_c,
            w_delta,
        }
    }

    /// Forward pass through SSM.
    ///
    /// Input: [batch, seq_len, d_model]
    /// Output: [batch, seq_len, d_model]
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            let mut h: Array1<f64> = Array1::zeros(self.d_state);

            for t in 0..seq_len {
                // Get input slice
                let x_t: Array1<f64> = x.slice(ndarray::s![b, t, ..]).to_owned();

                // Compute delta (step size)
                let mut delta = 0.0;
                for d in 0..self.d_model {
                    delta += x_t[d] * self.w_delta[[d, 0]];
                }
                delta = sigmoid(delta);

                // Compute B * x
                let mut bx: Array1<f64> = Array1::zeros(self.d_state);
                for s in 0..self.d_state {
                    for d in 0..self.d_model {
                        bx[s] += self.w_b[[d, s]] * x_t[d];
                    }
                }

                // State update: h = exp(delta * A) * h + delta * B * x
                for s in 0..self.d_state {
                    let a_bar = (delta * self.a[s]).exp();
                    h[s] = a_bar * h[s] + delta * bx[s];
                }

                // Output: y = C * h
                for d in 0..d_model {
                    let mut y: f64 = 0.0;
                    for s in 0..self.d_state {
                        y += h[s] * self.w_c[[s, d]];
                    }
                    output[[b, t, d]] = y;
                }
            }
        }

        output
    }
}

/// Bidirectional Mamba block for time series.
#[derive(Debug)]
pub struct MambaBlock {
    d_model: usize,
    ssm_forward: SelectiveSsm,
    ssm_backward: SelectiveSsm,
    gamma: Array1<f64>,
    beta: Array1<f64>,
}

impl MambaBlock {
    pub fn new(d_model: usize, d_state: usize, d_conv: usize) -> Self {
        Self {
            d_model,
            ssm_forward: SelectiveSsm::new(d_model, d_state, d_conv),
            ssm_backward: SelectiveSsm::new(d_model, d_state, d_conv),
            gamma: Array1::ones(d_model),
            beta: Array1::zeros(d_model),
        }
    }

    /// Forward pass with bidirectional processing.
    pub fn forward(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();

        // Forward SSM
        let fwd_out = self.ssm_forward.forward(x);

        // Backward SSM (reverse, process, reverse back)
        let x_rev = reverse_sequence(x);
        let bwd_out_rev = self.ssm_backward.forward(&x_rev);
        let bwd_out = reverse_sequence(&bwd_out_rev);

        // Combine with residual and layer norm
        let mut output = Array3::zeros((batch_size, seq_len, d_model));
        for b in 0..batch_size {
            for t in 0..seq_len {
                for d in 0..d_model {
                    output[[b, t, d]] = x[[b, t, d]] + fwd_out[[b, t, d]] + bwd_out[[b, t, d]];
                }
            }
        }

        // Layer normalization
        self.layer_norm(&output)
    }

    fn layer_norm(&self, x: &Array3<f64>) -> Array3<f64> {
        let (batch_size, seq_len, d_model) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for t in 0..seq_len {
                // Calculate mean and std
                let mut mean = 0.0;
                for d in 0..d_model {
                    mean += x[[b, t, d]];
                }
                mean /= d_model as f64;

                let mut var = 0.0;
                for d in 0..d_model {
                    var += (x[[b, t, d]] - mean).powi(2);
                }
                var /= d_model as f64;
                let std = (var + 1e-6).sqrt();

                // Normalize
                for d in 0..d_model {
                    output[[b, t, d]] = self.gamma[d] * (x[[b, t, d]] - mean) / std + self.beta[d];
                }
            }
        }

        output
    }
}

/// Sigmoid activation function.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Reverse sequence along time dimension.
fn reverse_sequence(x: &Array3<f64>) -> Array3<f64> {
    let (batch_size, seq_len, d_model) = x.dim();
    let mut output = Array3::zeros((batch_size, seq_len, d_model));

    for b in 0..batch_size {
        for t in 0..seq_len {
            for d in 0..d_model {
                output[[b, t, d]] = x[[b, seq_len - 1 - t, d]];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_forward() {
        let ssm = SelectiveSsm::new(64, 16, 4);
        let x = Array3::zeros((1, 10, 64));
        let out = ssm.forward(&x);
        assert_eq!(out.dim(), (1, 10, 64));
    }

    #[test]
    fn test_mamba_block() {
        let block = MambaBlock::new(64, 16, 4);
        let x = Array3::zeros((1, 10, 64));
        let out = block.forward(&x);
        assert_eq!(out.dim(), (1, 10, 64));
    }
}
