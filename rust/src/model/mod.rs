//! Model module for C-Mamba implementation.
//!
//! Contains:
//! - CMamba: Main model implementation
//! - Mamba SSM: Selective State Space Model
//! - GDD-MLP: Global-local Dual-branch MLP

pub mod cmamba;
pub mod mamba;
pub mod gdd_mlp;

pub use cmamba::{CMamba, CMambaConfig};
