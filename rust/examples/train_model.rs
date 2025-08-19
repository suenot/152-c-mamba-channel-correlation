//! Example: Train C-Mamba model.

use cmamba_trading::model::cmamba::{CMamba, CMambaConfig};
use ndarray::Array3;

fn main() -> anyhow::Result<()> {
    println!("C-Mamba Model Training Example");
    println!("===============================\n");

    // Configure model
    let config = CMambaConfig {
        seq_len: 60,
        pred_len: 5,
        n_channels: 10,
        d_model: 64,
        d_state: 16,
        n_layers: 2,
        patch_len: 12,
        stride: 6,
        ..CMambaConfig::default()
    };

    println!("Model Configuration:");
    println!("  - Sequence length: {}", config.seq_len);
    println!("  - Prediction length: {}", config.pred_len);
    println!("  - Number of channels: {}", config.n_channels);
    println!("  - Model dimension: {}", config.d_model);
    println!("  - SSM state dimension: {}", config.d_state);
    println!("  - Number of layers: {}", config.n_layers);
    println!();

    // Create model
    let model = CMamba::new(config);
    println!("Model created successfully!\n");

    // Generate sample data (in real usage, load from Bybit)
    let batch_size = 32;
    let seq_len = 60;
    let n_channels = 10;

    println!("Generating sample data...");
    let sample_data = Array3::from_shape_fn(
        (batch_size, seq_len, n_channels),
        |(_, t, c)| {
            // Simulate correlated price movements
            let trend = (t as f64) * 0.001;
            let noise = rand::random::<f64>() * 0.01;
            let correlation = (c as f64) * 0.0001;
            trend + noise + correlation
        },
    );

    println!("  Shape: {:?}", sample_data.dim());

    // Forward pass
    println!("\nRunning forward pass...");
    let output = model.forward(&sample_data);

    println!("  Predictions shape: {:?}", output.predictions.dim());

    if let Some(ref attention) = output.channel_attention {
        println!("  Channel attention shape: {:?}", attention.dim());

        // Show sample correlations
        println!("\nSample channel correlations (first batch):");
        for i in 0..3 {
            for j in 0..3 {
                print!("  {:.3}", attention[[0, i, j]]);
            }
            println!();
        }
    }

    // Analyze dependencies
    println!("\nAnalyzing dependencies...");
    let analysis = model.analyze_dependencies(&sample_data);

    if let Some(ref corr) = analysis.correlation_matrix {
        println!("  Average correlation: {:.3}",
            corr.iter().sum::<f64>() / (corr.len() as f64));
    }

    println!("\nTraining example complete!");
    println!("In production, you would:");
    println!("  1. Load real data from Bybit");
    println!("  2. Split into train/val/test");
    println!("  3. Train with backpropagation");
    println!("  4. Evaluate on test set");

    Ok(())
}
