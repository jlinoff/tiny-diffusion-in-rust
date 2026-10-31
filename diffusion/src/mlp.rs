//! The stage-2 network: a small MLP (multilayer perceptron) that maps a
//! point and a time to a vector of the same size as the point: a velocity
//! for flow matching, or a noise estimate for DDPM.
//!
//! ```text
//! input  = [x₁, x₂, t, sin(2πt), cos(2πt), sin(4πt), cos(4πt), …]   (the time embedding)
//! h      = GELU(input·W_in + b_in)
//! h      = GELU(h·W_k + b_k)        for each of `depth` − 1 more hidden layers
//! output = h·W_out + b_out          (W_out starts at zero: the first prediction is 0)
//! ```
//!
//! Generic over the float type like the tape: f32 to train, f64 for the
//! gradient check. `forward` builds the computation on a tape for training;
//! `predict` computes the same numbers directly, without one, for
//! generation, and a test requires the two to agree.

use crate::rng::Rng;
use crate::safetensors::{File, Tensor};
use crate::tape::{gelu, Tape, Var};
use crate::tensor::{Float, Mat};
use std::collections::BTreeMap;

/// The MLP's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpConfig {
    /// The size of each example: 2 for points in a plane.
    pub dim: usize,
    /// Neurons per hidden layer.
    pub width: usize,
    /// Hidden layers.
    pub depth: usize,
    /// Sine-cosine pairs in the time embedding.
    pub freqs: usize,
}

impl MlpConfig {
    /// Columns of the input: the point, t itself, and the sine-cosine pairs.
    pub fn input_cols(&self) -> usize {
        self.dim + 1 + 2 * self.freqs
    }

    pub fn parameters(&self) -> usize {
        let (i, w, d) = (self.input_cols(), self.width, self.dim);
        (i * w + w) + (self.depth - 1) * (w * w + w) + (w * d + d)
    }
}

/// The time embedding, one row per t: [t, sin(2πt), cos(2πt), sin(4πt),
/// cos(4πt), …], `freqs` pairs. Neighbouring times get similar rows, and the
/// higher frequencies let the network change quickly with t where it must.
pub fn time_features(t: &[f32], freqs: usize) -> Mat<f32> {
    let cols = 1 + 2 * freqs;
    let mut m = Mat::zeros(t.len(), cols);
    for (r, &tr) in t.iter().enumerate() {
        let row = m.row_mut(r);
        row[0] = tr;
        for k in 0..freqs {
            let a = 2.0 * std::f32::consts::PI * (k + 1) as f32 * tr;
            row[1 + 2 * k] = a.sin();
            row[2 + 2 * k] = a.cos();
        }
    }
    m
}

/// The network's input: each point's coordinates followed by its time
/// embedding.
pub fn input(x: &Mat<f32>, t: &[f32], freqs: usize) -> Mat<f32> {
    let tf = time_features(t, freqs);
    let mut m = Mat::zeros(x.rows, x.cols + tf.cols);
    for r in 0..x.rows {
        let row = m.row_mut(r);
        row[..x.cols].copy_from_slice(x.row(r));
        row[x.cols..].copy_from_slice(tf.row(r));
    }
    m
}

/// The weights, in a fixed order: W_in, b_in, then each hidden layer's W and
/// b, then W_out, b_out.
pub struct Mlp<T> {
    pub cfg: MlpConfig,
    pub params: Vec<Mat<T>>,
}

impl<T: Float> Mlp<T> {
    /// Random weights: each W normal with standard deviation 1/√(inputs), so
    /// every layer starts with outputs of about the same size as its inputs;
    /// biases zero; W_out zero, so the untrained network predicts 0.
    pub fn new(cfg: MlpConfig, seed: u64) -> Mlp<T> {
        let mut rng = Rng::new(seed);
        let mut normal = |rows: usize, cols: usize| {
            let s = 1.0 / (rows as f64).sqrt();
            Mat::from_vec(
                rows,
                cols,
                (0..rows * cols)
                    .map(|_| T::from_f64(s * rng.normal()))
                    .collect(),
            )
        };
        let (i, w, d) = (cfg.input_cols(), cfg.width, cfg.dim);
        let mut params = vec![normal(i, w), Mat::zeros(1, w)];
        for _ in 1..cfg.depth {
            params.push(normal(w, w));
            params.push(Mat::zeros(1, w));
        }
        params.push(Mat::zeros(w, d));
        params.push(Mat::zeros(1, d));
        Mlp { cfg, params }
    }

    /// Build the forward pass on a tape. Returns the output and the
    /// parameters' variables (to read their gradients after `backward`).
    pub fn forward(&self, tape: &mut Tape<T>, input: Mat<T>) -> (Var, Vec<Var>) {
        let vars: Vec<Var> = self.params.iter().map(|p| tape.leaf(p.clone())).collect();
        let mut h = tape.leaf(input);
        let layers = vars.len() / 2;
        for l in 0..layers {
            let z = tape.matmul(h, vars[2 * l]);
            h = tape.add_row(z, vars[2 * l + 1]);
            if l + 1 < layers {
                h = tape.gelu(h);
            }
        }
        (h, vars)
    }

    /// The same computation as `forward`, without a tape: for generation.
    pub fn predict_input(&self, input: &Mat<T>) -> Mat<T> {
        let layers = self.params.len() / 2;
        let mut h = input.clone();
        for l in 0..layers {
            let mut z = h.matmul(&self.params[2 * l]);
            let b = &self.params[2 * l + 1];
            for r in 0..z.rows {
                for (x, &y) in z.row_mut(r).iter_mut().zip(&b.data) {
                    *x += y;
                }
            }
            h = if l + 1 < layers { z.map(gelu) } else { z };
        }
        h
    }
}

impl Mlp<f32> {
    /// The prediction for points x at time t (the same t for every point).
    pub fn predict(&self, x: &Mat<f32>, t: f32) -> Mat<f32> {
        self.predict_input(&input(x, &vec![t; x.rows], self.cfg.freqs))
    }

    pub fn names(&self) -> Vec<String> {
        let layers = self.params.len() / 2;
        (0..layers)
            .flat_map(|l| {
                let p = match l {
                    0 => "in".to_string(),
                    l if l + 1 == layers => "out".to_string(),
                    l => format!("hidden{l}"),
                };
                [format!("{p}.w"), format!("{p}.b")]
            })
            .collect()
    }

    /// The weights as named tensors, and the shape as metadata.
    pub fn to_file(&self, mut metadata: BTreeMap<String, String>) -> File {
        let c = self.cfg;
        for (k, v) in [
            ("dim", c.dim),
            ("width", c.width),
            ("depth", c.depth),
            ("freqs", c.freqs),
        ] {
            metadata.insert(k.into(), v.to_string());
        }
        let tensors = self
            .names()
            .iter()
            .zip(&self.params)
            .map(|(n, p)| Tensor::new(n, &[p.rows, p.cols], p.data.clone()))
            .collect();
        File { tensors, metadata }
    }

    /// The model in a file written by `to_file`, checked against the shape
    /// its metadata states.
    pub fn from_file(f: &File) -> Result<Mlp<f32>, String> {
        let num = |k: &str| -> Result<usize, String> {
            f.metadata
                .get(k)
                .ok_or(format!("the model file has no {k:?} in its metadata"))?
                .parse()
                .map_err(|_| format!("the model file's {k:?} isn't a number"))
        };
        let cfg = MlpConfig {
            dim: num("dim")?,
            width: num("width")?,
            depth: num("depth")?,
            freqs: num("freqs")?,
        };
        let mut m = Mlp::<f32>::new(cfg, 0);
        for (name, p) in m.names().iter().zip(m.params.iter_mut()) {
            let t = f
                .tensors
                .iter()
                .find(|t| &t.name == name)
                .ok_or(format!("the model file has no tensor {name}"))?;
            if t.shape != [p.rows, p.cols] {
                return Err(format!(
                    "{name} is {:?}, the metadata implies {:?}",
                    t.shape,
                    [p.rows, p.cols]
                ));
            }
            p.data.clone_from(&t.data);
        }
        Ok(m)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // coordinate loops read more plainly
mod tests {
    use super::*;

    fn cfg() -> MlpConfig {
        MlpConfig {
            dim: 2,
            width: 8,
            depth: 3,
            freqs: 2,
        }
    }

    #[test]
    fn parameter_count_matches_the_formula() {
        let m = Mlp::<f32>::new(cfg(), 1);
        let total: usize = m.params.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, cfg().parameters());
        assert_eq!(cfg().input_cols(), 7);
    }

    #[test]
    fn time_features_by_hand() {
        let f = time_features(&[0.25], 2);
        let want = [0.25, 1.0, 0.0, 0.0, -1.0]; // sin/cos of π/2, then of π
        for (a, b) in f.data.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{:?}", f.data);
        }
    }

    /// Every weight's gradient, through the whole network and the loss,
    /// against finite differences in f64.
    #[test]
    fn gradients_match_finite_differences() {
        let mut m = Mlp::<f64>::new(cfg(), 2);
        // Nonzero output weights, so every layer gets a gradient.
        let mut rng = Rng::new(3);
        for p in m.params.iter_mut() {
            for x in p.data.iter_mut() {
                *x += 0.3 * rng.normal();
            }
        }
        let inp = Mat::from_vec(5, 7, (0..35).map(|_| rng.normal()).collect());
        let target = Mat::from_vec(5, 2, (0..10).map(|_| rng.normal()).collect());
        let loss_of = |m: &Mlp<f64>| {
            let mut t = Tape::new();
            let (out, _) = m.forward(&mut t, inp.clone());
            let l = t.mse(out, &target);
            t.value(l).data[0]
        };
        let mut t = Tape::new();
        let (out, vars) = m.forward(&mut t, inp.clone());
        let l = t.mse(out, &target);
        t.backward(l);
        let grads: Vec<Mat<f64>> = vars.iter().map(|&v| t.take_grad(v)).collect();
        let eps = 1e-6;
        let mut worst: f64 = 0.0;
        for k in 0..m.params.len() {
            for i in 0..m.params[k].data.len() {
                m.params[k].data[i] += eps;
                let up = loss_of(&m);
                m.params[k].data[i] -= 2.0 * eps;
                let down = loss_of(&m);
                m.params[k].data[i] += eps;
                let numeric = (up - down) / (2.0 * eps);
                let a = grads[k].data[i];
                worst = worst.max((numeric - a).abs() / (numeric.abs() + a.abs() + 1e-3));
            }
        }
        assert!(worst < 1e-6, "{worst}");
    }

    #[test]
    fn predict_equals_the_tape_forward_pass() {
        let mut m = Mlp::<f32>::new(cfg(), 4);
        let mut rng = Rng::new(5);
        for x in m.params.last_mut().unwrap().data.iter_mut() {
            *x = rng.normal() as f32;
        }
        let x = Mat::from_vec(6, 2, (0..12).map(|_| rng.normal() as f32).collect());
        let mut t = Tape::new();
        let (out, _) = m.forward(&mut t, input(&x, &[0.3; 6], 2));
        assert_eq!(t.value(out), &m.predict(&x, 0.3));
    }

    #[test]
    fn untrained_network_predicts_zero() {
        let m = Mlp::<f32>::new(cfg(), 6);
        let x = Mat::from_vec(2, 2, vec![1.0, -1.0, 0.5, 2.0]);
        assert!(m.predict(&x, 0.5).data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn file_round_trip_and_shape_check() {
        let m = Mlp::<f32>::new(cfg(), 7);
        let f = m.to_file(BTreeMap::new());
        let back = Mlp::from_file(&f).unwrap();
        assert_eq!(back.cfg, m.cfg);
        assert_eq!(back.params, m.params);
        let mut bad = m.to_file(BTreeMap::new());
        bad.metadata.insert("width".into(), "9".into());
        assert!(Mlp::from_file(&bad).err().unwrap().contains("in.w"));
    }
}
