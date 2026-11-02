//! mini_diffusion: a small diffusion model, built by hand (see PLAN.md).
//!
//! The foundation, copied from tiny-llm-in-rust and tested here:
//! - `tensor`: row-major matrices and their products
//! - `tape`: reverse-mode automatic differentiation on whole matrices
//! - `blas`: optional Apple Accelerate for the large products
//! - `rng`: a seedable random number generator, normal draws included
//! - `safetensors`: reading and writing model files
//!
//! A library as well as a binary, so each stage's code is reachable from
//! its tests and from the command line alike.

pub mod blas;
pub mod classify;
pub mod data2d;
pub mod ddpm;
pub mod dit;
pub mod flow;
pub mod image;
pub mod inside;
pub mod metrics;
pub mod mlp;
pub mod optim;
pub mod rng;
pub mod safetensors;
pub mod style;
pub mod svg;
pub mod tape;
pub mod tensor;
pub mod text;
pub mod textmodel;
pub mod train2d;
pub mod trainimg;
