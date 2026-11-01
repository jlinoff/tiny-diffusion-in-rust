//! The DiT, the diffusion transformer of stage 3 (Peebles and Xie 2023):
//! a transformer whose tokens are an image's patches, predicting a velocity
//! for every pixel (PLAN.md, Terms: The networks).
//!
//! ```text
//! patches (B·T × p·p·c) ─ W_patch ─► + position embedding ─► x   (B·T × d)
//! t ─ time embedding ─ Linear ─ SiLU ─ Linear ─► c                (B × d)
//! each block:
//!     m = Linear(SiLU(c))  → shift₁, scale₁, gate₁, shift₂, scale₂, gate₂
//!     x += gate₁ ⊙ Attention(LN(x) ⊙ (1 + scale₁) + shift₁)     (no causal mask)
//!     x += gate₂ ⊙ FFN(LN(x) ⊙ (1 + scale₂) + shift₂)
//! final: LN(x) ⊙ (1 + scale) + shift ─ W_out ─► velocity per patch (B·T × p·p·c)
//! ```
//!
//! The time enters every block through its layer normalizations' scale and
//! shift (adaptive layer normalization, adaLN), computed per example, and
//! through a gate on each sublayer's output. The layers computing them
//! start at zero ("adaLN-Zero"), so every gate starts at 0 and every block
//! starts as the identity: an untrained DiT predicts exactly 0, and training
//! opens each block gradually. The image's size, channels and patch size are
//! part of the configuration, never constants (Decision 5).

use crate::image::{check_patch, patch_count, patchify, unpatchify, Shape};
use crate::mlp::time_features;
use crate::rng::Rng;
use crate::safetensors::{File, Tensor};
use crate::tape::{Tape, Var};
use crate::tensor::{Float, Mat};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DitConfig {
    pub image: Shape,
    /// Patch side, in pixels: it must divide the height and width.
    pub patch: usize,
    /// d, the length of every token's vector.
    pub width: usize,
    pub blocks: usize,
    pub heads: usize,
    /// Sine-cosine pairs in the time embedding.
    pub freqs: usize,
}

impl DitConfig {
    pub fn tokens(&self) -> usize {
        patch_count(self.image, self.patch)
    }

    pub fn patch_values(&self) -> usize {
        self.patch * self.patch * self.image.channels
    }

    pub fn check(&self) -> Result<(), String> {
        check_patch(self.image, self.patch)?;
        if self.heads == 0 || !self.width.is_multiple_of(self.heads) {
            return Err(format!(
                "{} heads don't divide width {}",
                self.heads, self.width
            ));
        }
        if self.blocks == 0 {
            return Err("at least one block".into());
        }
        Ok(())
    }

    /// Every weight's name and shape, in order.
    pub fn layout(&self) -> Vec<(String, usize, usize)> {
        let (d, pv, tf) = (self.width, self.patch_values(), 1 + 2 * self.freqs);
        let mut v = vec![
            ("patch.w".to_string(), pv, d),
            ("patch.b".to_string(), 1, d),
            ("pos".to_string(), self.tokens(), d),
            ("time.w1".to_string(), tf, d),
            ("time.b1".to_string(), 1, d),
            ("time.w2".to_string(), d, d),
            ("time.b2".to_string(), 1, d),
        ];
        for l in 0..self.blocks {
            for (n, r, c) in [
                ("ada.w", d, 6 * d),
                ("ada.b", 1, 6 * d),
                ("qkv.w", d, 3 * d),
                ("qkv.b", 1, 3 * d),
                ("proj.w", d, d),
                ("proj.b", 1, d),
                ("ffn.w1", d, 4 * d),
                ("ffn.b1", 1, 4 * d),
                ("ffn.w2", 4 * d, d),
                ("ffn.b2", 1, d),
            ] {
                v.push((format!("block{l}.{n}"), r, c));
            }
        }
        v.push(("final.ada.w".into(), d, 2 * d));
        v.push(("final.ada.b".into(), 1, 2 * d));
        v.push(("out.w".into(), d, pv));
        v.push(("out.b".into(), 1, pv));
        v
    }

    pub fn parameters(&self) -> usize {
        self.layout().iter().map(|(_, r, c)| r * c).sum()
    }
}

pub struct Dit<T> {
    pub cfg: DitConfig,
    pub params: Vec<Mat<T>>,
}

impl<T: Float> Dit<T> {
    /// Random weights, normal with standard deviation 1/√(inputs); biases
    /// zero; position embeddings small; the adaLN layers and the output
    /// layer zero (adaLN-Zero, above).
    pub fn new(cfg: DitConfig, seed: u64) -> Dit<T> {
        let mut rng = Rng::new(seed);
        let params = cfg
            .layout()
            .iter()
            .map(|(name, r, c)| {
                let zero = name.contains(".b") || name.contains("ada") || name.starts_with("out.");
                let std = if name == "pos" {
                    0.02
                } else {
                    1.0 / (*r as f64).sqrt()
                };
                if zero {
                    Mat::zeros(*r, *c)
                } else {
                    Mat::from_vec(
                        *r,
                        *c,
                        (0..r * c)
                            .map(|_| T::from_f64(std * rng.normal()))
                            .collect(),
                    )
                }
            })
            .collect();
        Dit { cfg, params }
    }

    /// The forward pass on a tape: `patches` is (B·T) × (p·p·c), the noisy
    /// images cut into patches; `t` one time per image. Returns the
    /// predicted velocity per patch, and the parameters' variables.
    pub fn forward(&self, tape: &mut Tape<T>, patches: Mat<T>, t: &[f32]) -> (Var, Vec<Var>) {
        let c = self.cfg;
        let (d, seq, b) = (c.width, c.tokens(), t.len());
        assert_eq!(patches.rows, b * seq, "one time per image");
        let p: Vec<Var> = self.params.iter().map(|m| tape.leaf(m.clone())).collect();
        let lin = |tape: &mut Tape<T>, x: Var, w: usize| {
            let z = tape.matmul(x, p[w]);
            tape.add_row(z, p[w + 1])
        };
        let ones = tape.leaf(Mat::from_vec(1, d, vec![T::one(); d]));
        let zeros = tape.leaf(Mat::zeros(1, d));
        // Tokens: patches projected to d, plus each position's embedding.
        let inp = tape.leaf(patches);
        let x0 = lin(tape, inp, 0);
        let positions: Vec<usize> = (0..b).flat_map(|_| 0..seq).collect();
        let pos = tape.gather(p[2], &positions);
        let mut x = tape.add(x0, pos);
        // The time: sines and cosines, then a small MLP.
        let tf = time_features(t, c.freqs).convert::<T>();
        let tl = tape.leaf(tf);
        let h = lin(tape, tl, 3);
        let h = tape.silu(h);
        let cond = lin(tape, h, 5);
        let cond = tape.silu(cond);
        for l in 0..c.blocks {
            let w = 7 + 10 * l;
            let m = lin(tape, cond, w);
            let part = |tape: &mut Tape<T>, k: usize| tape.slice_cols(m, k * d, d);
            let (sh1, sc1, g1, sh2, sc2, g2) = (
                part(tape, 0),
                part(tape, 1),
                part(tape, 2),
                part(tape, 3),
                part(tape, 4),
                part(tape, 5),
            );
            // Attention, every patch seeing every patch.
            let n = tape.layer_norm(x, ones, zeros);
            let n = tape.modulate(n, sc1, sh1, seq);
            let qkv = lin(tape, n, w + 2);
            let (q, k, v) = (
                tape.slice_cols(qkv, 0, d),
                tape.slice_cols(qkv, d, d),
                tape.slice_cols(qkv, 2 * d, d),
            );
            let a = tape.attention_full(q, k, v, seq, c.heads);
            let a = lin(tape, a, w + 4);
            let a = tape.gate(a, g1, seq);
            x = tape.add(x, a);
            // The feed-forward network, each patch on its own.
            let n = tape.layer_norm(x, ones, zeros);
            let n = tape.modulate(n, sc2, sh2, seq);
            let f = lin(tape, n, w + 6);
            let f = tape.gelu(f);
            let f = lin(tape, f, w + 8);
            let f = tape.gate(f, g2, seq);
            x = tape.add(x, f);
        }
        let w = 7 + 10 * c.blocks;
        let m = lin(tape, cond, w);
        let (sh, sc) = (tape.slice_cols(m, 0, d), tape.slice_cols(m, d, d));
        let n = tape.layer_norm(x, ones, zeros);
        let n = tape.modulate(n, sc, sh, seq);
        let out = lin(tape, n, w + 2);
        (out, p)
    }
}

impl Dit<f32> {
    /// The predicted velocity for images x (n × h·w·c), all at time t.
    pub fn predict(&self, x: &Mat<f32>, t: f32) -> Mat<f32> {
        let mut tape = Tape::new();
        let (out, _) = self.forward(
            &mut tape,
            patchify(x, self.cfg.image, self.cfg.patch),
            &vec![t; x.rows],
        );
        unpatchify(tape.value(out), self.cfg.image, self.cfg.patch)
    }

    pub fn to_file(&self, mut metadata: BTreeMap<String, String>) -> File {
        let c = self.cfg;
        for (k, v) in [
            ("height", c.image.height),
            ("width_px", c.image.width),
            ("channels", c.image.channels),
            ("patch", c.patch),
            ("width", c.width),
            ("blocks", c.blocks),
            ("heads", c.heads),
            ("freqs", c.freqs),
        ] {
            metadata.insert(k.into(), v.to_string());
        }
        metadata.insert("kind".into(), "dit".into());
        let tensors = c
            .layout()
            .iter()
            .zip(&self.params)
            .map(|((n, r, cl), p)| Tensor::new(n, &[*r, *cl], p.data.clone()))
            .collect();
        File { tensors, metadata }
    }

    pub fn from_file(f: &File) -> Result<Dit<f32>, String> {
        let num = |k: &str| -> Result<usize, String> {
            f.metadata
                .get(k)
                .ok_or(format!("the model file has no {k:?} in its metadata"))?
                .parse()
                .map_err(|_| format!("the model file's {k:?} isn't a number"))
        };
        if f.metadata.get("kind").map(String::as_str) != Some("dit") {
            return Err("not a DiT model file".into());
        }
        let cfg = DitConfig {
            image: Shape {
                height: num("height")?,
                width: num("width_px")?,
                channels: num("channels")?,
            },
            patch: num("patch")?,
            width: num("width")?,
            blocks: num("blocks")?,
            heads: num("heads")?,
            freqs: num("freqs")?,
        };
        cfg.check()?;
        let mut params = Vec::new();
        for (name, r, c) in cfg.layout() {
            let t = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or(format!("the model file has no tensor {name}"))?;
            if t.shape != [r, c] {
                return Err(format!(
                    "{name} is {:?}, the metadata implies {:?}",
                    t.shape,
                    [r, c]
                ));
            }
            params.push(Mat::from_vec(r, c, t.data.clone()));
        }
        Ok(Dit { cfg, params })
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // index loops over parameters and their gradients
mod tests {
    use super::*;

    fn cfg(h: usize, w: usize, ch: usize, patch: usize) -> DitConfig {
        DitConfig {
            image: Shape {
                height: h,
                width: w,
                channels: ch,
            },
            patch,
            width: 8,
            blocks: 2,
            heads: 2,
            freqs: 2,
        }
    }

    #[test]
    fn layout_and_parameter_count() {
        let c = cfg(28, 28, 1, 4);
        assert_eq!(c.tokens(), 49);
        let m = Dit::<f32>::new(c, 1);
        let total: usize = m.params.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, c.parameters());
        assert!(cfg(28, 28, 1, 5).check().is_err());
        let mut bad = c;
        bad.heads = 3;
        assert!(bad.check().is_err());
    }

    /// The parameter formulas of PLAN.md's "Where the weights are", term
    /// by term, against the layout: the defaults with both patch sizes,
    /// and odd sizes, so a change to the model shows up as a failure here
    /// and a reminder to update the table.
    #[test]
    fn parameter_formulas_in_plan_md() {
        let mut configs = vec![cfg(28, 28, 1, 4), cfg(16, 24, 3, 8)];
        for patch in [4, 7] {
            let mut c = cfg(28, 28, 1, patch);
            (c.width, c.blocks, c.heads, c.freqs) = (128, 6, 4, 8);
            configs.push(c);
        }
        for c in configs {
            let (d, b, k) = (c.width, c.blocks, c.freqs);
            let (v, n) = (c.patch_values(), c.tokens());
            let patch = (v + 1) * d;
            let pos = n * d;
            let time = (2 * k + 2) * d + d * d + d;
            let blocks = b * (18 * d * d + 15 * d);
            let last = 2 * d * d + 2 * d + (d + 1) * v;
            assert_eq!(patch + pos + time + blocks + last, c.parameters());
            let closed = b * (18 * d * d + 15 * d) + 3 * d * d + (2 * v + n + 2 * k + 6) * d + v;
            assert_eq!(closed, c.parameters());
        }
        let mut c = cfg(28, 28, 1, 4);
        (c.width, c.blocks, c.heads, c.freqs) = (128, 6, 4, 8);
        assert_eq!(c.parameters(), 1_843_344);
        c.patch = 7;
        assert_eq!(c.parameters(), 1_847_601);
        (c.patch, c.width) = (4, 256);
        assert_eq!(c.parameters(), 7_323_920);
    }

    /// adaLN-Zero: every gate and the output layer start at zero, so the
    /// untrained DiT predicts exactly 0.
    #[test]
    fn untrained_dit_predicts_zero() {
        let m = Dit::<f32>::new(cfg(8, 8, 1, 4), 2);
        let x = Mat::from_vec(2, 64, (0..128).map(|i| (i as f32 * 0.37).sin()).collect());
        assert!(m.predict(&x, 0.3).data.iter().all(|&v| v == 0.0));
    }

    /// Every weight's gradient through the whole DiT, against finite
    /// differences in f64, with the zero-initialized weights made nonzero so
    /// every path carries a gradient.
    #[test]
    fn gradients_match_finite_differences() {
        let c = cfg(4, 4, 1, 2); // 4 tokens of 4 values
        let mut m = Dit::<f64>::new(c, 3);
        let mut rng = Rng::new(4);
        for p in m.params.iter_mut() {
            for x in p.data.iter_mut() {
                *x += 0.2 * rng.normal();
            }
        }
        let inp = Mat::from_vec(8, 4, (0..32).map(|_| rng.normal()).collect()); // 2 images
        let target = Mat::from_vec(8, 4, (0..32).map(|_| rng.normal()).collect());
        let t = [0.2f32, 0.7];
        let loss_of = |m: &Dit<f64>| {
            let mut tp = Tape::new();
            let (o, _) = m.forward(&mut tp, inp.clone(), &t);
            let l = tp.mse(o, &target);
            tp.value(l).data[0]
        };
        let mut tp = Tape::new();
        let (o, vars) = m.forward(&mut tp, inp.clone(), &t);
        let l = tp.mse(o, &target);
        tp.backward(l);
        let grads: Vec<Mat<f64>> = vars.iter().map(|&v| tp.take_grad(v)).collect();
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

    /// The same code at several sizes, channel counts and patch sizes
    /// (Decision 5).
    #[test]
    fn runs_at_any_image_size() {
        for (h, w, ch, p) in [
            (16, 16, 1, 4),
            (28, 28, 1, 7),
            (32, 32, 3, 8),
            (12, 8, 3, 4),
        ] {
            let mut m = Dit::<f32>::new(cfg(h, w, ch, p), 5);
            for x in m.params.iter_mut().flat_map(|p| p.data.iter_mut()) {
                *x += 0.01;
            }
            let x = Mat::from_vec(2, h * w * ch, vec![0.5; 2 * h * w * ch]);
            let v = m.predict(&x, 0.5);
            assert_eq!((v.rows, v.cols), (2, h * w * ch));
            assert!(v.data.iter().all(|z| z.is_finite()));
        }
    }

    /// Each image is processed on its own: changing one image of a batch
    /// doesn't change another's prediction.
    #[test]
    fn images_in_a_batch_dont_mix() {
        let mut m = Dit::<f32>::new(cfg(8, 8, 1, 4), 6);
        let mut rng = Rng::new(7);
        for x in m.params.iter_mut().flat_map(|p| p.data.iter_mut()) {
            *x += 0.1 * rng.normal() as f32;
        }
        let mut x = Mat::from_vec(2, 64, (0..128).map(|_| rng.normal() as f32).collect());
        let a = m.predict(&x, 0.4);
        x.row_mut(1)[5] += 1.0;
        let b = m.predict(&x, 0.4);
        assert_eq!(a.row(0), b.row(0));
        assert_ne!(a.row(1), b.row(1));
    }

    #[test]
    fn file_round_trip() {
        let m = Dit::<f32>::new(cfg(16, 16, 3, 4), 8);
        let f = m.to_file(BTreeMap::new());
        let back = Dit::from_file(&f).unwrap();
        assert_eq!(back.cfg, m.cfg);
        assert_eq!(back.params, m.params);
        let mut bad = m.to_file(BTreeMap::new());
        bad.metadata.insert("patch".into(), "8".into());
        assert!(Dit::from_file(&bad).is_err());
    }
}
