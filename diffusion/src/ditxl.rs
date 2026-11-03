//! DiT-XL/2, the DiT paper's trained model, run through this project's
//! matrices (PLAN.md, stage 7): forward passes only, reading the weights
//! under diffusers' names.
//!
//! ```text
//! tokens = PatchEmbed(latent) + positions        (2 × 2 patches, fixed sine-cosine positions)
//! c      = TimeMLP(time features(t)) + class row (each block has its own copy, as diffusers stores it)
//! each block:  m = Linear(SiLU(c)) → shift, scale, gate for attention and for the MLP
//!              x += gate₁ · Attention(LN(x)·(1 + scale₁) + shift₁)
//!              x += gate₂ · MLP(LN(x)·(1 + scale₂) + shift₂)
//! out    = Linear(LN(x)·(1 + scale) + shift), shift and scale from c (block 0's copy)
//! ```
//!
//! Two small differences separate diffusers from the DiT repository the
//! model was trained with (`Reference`): diffusers divides the time
//! features' frequency exponents by 127 where the original divides by
//! 128, and its second layer norm in each block uses ε = 10⁻⁵, not 10⁻⁶.

use crate::safetensors::{File, Tensor};
use crate::tensor::Mat;

/// Whose arithmetic to follow where the two differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reference {
    /// diffusers' (`make dit-check` compares with it number for number).
    Diffusers,
    /// The DiT repository's, the one the model was trained with.
    Original,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DitXlConfig {
    pub width: usize,
    pub heads: usize,
    pub blocks: usize,
    pub patch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// The latent's side, in values (32 for 256 × 256 photographs).
    pub size: usize,
    /// Classes, not counting "no label" (whose index is `classes`).
    pub classes: usize,
}

impl DitXlConfig {
    /// From the configuration JSON diffusers writes (`config.json`, or the
    /// fixtures' metadata): the few numbers needed, found by name.
    pub fn from_json(json: &str) -> Result<DitXlConfig, String> {
        let num = |key: &str| -> Result<usize, String> {
            let at = json
                .find(&format!("\"{key}\""))
                .ok_or(format!("no {key:?} in the configuration"))?;
            let rest = &json[at + key.len() + 2..];
            let rest = rest
                .trim_start()
                .strip_prefix(':')
                .ok_or(format!("{key}: no value"))?;
            let digits: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits
                .parse()
                .map_err(|_| format!("{key} isn't a whole number"))
        };
        let heads = num("num_attention_heads")?;
        Ok(DitXlConfig {
            width: heads * num("attention_head_dim")?,
            heads,
            blocks: num("num_layers")?,
            patch: num("patch_size")?,
            in_channels: num("in_channels")?,
            out_channels: num("out_channels")?,
            size: num("sample_size")?,
            classes: num("num_embeds_ada_norm")?,
        })
    }

    pub fn tokens(&self) -> usize {
        (self.size / self.patch).pow(2)
    }
}

/// A linear layer: y = x·W + b, W stored as inputs × outputs.
pub struct Linear {
    pub w: Mat<f32>,
    pub b: Vec<f32>,
}

impl Linear {
    /// From PyTorch's [outputs, inputs] weight (and bias, if any).
    fn load(f: &Weights, name: &str) -> Result<Linear, String> {
        let w = f.get(&format!("{name}.weight"))?;
        if w.shape.len() != 2 {
            return Err(format!(
                "{name}.weight: expected a matrix, got {:?}",
                w.shape
            ));
        }
        let (out, inp) = (w.shape[0], w.shape[1]);
        let b = match f.find(&format!("{name}.bias")) {
            Some(t) => t.data.clone(),
            None => vec![0.0; out],
        };
        Ok(Linear {
            w: Mat::from_vec(out, inp, w.data.clone()).transpose(),
            b,
        })
    }

    /// Load by name from another model's weights (the decoder's).
    pub fn load_named(f: &Weights, name: &str) -> Result<Linear, String> {
        Linear::load(f, name)
    }

    pub fn forward(&self, x: &Mat<f32>) -> Mat<f32> {
        let mut y = x.matmul(&self.w);
        for r in 0..y.rows {
            for (v, b) in y.row_mut(r).iter_mut().zip(&self.b) {
                *v += b;
            }
        }
        y
    }
}

/// The tensors of a file, found by name.
pub struct Weights<'a> {
    file: &'a File,
}

impl<'a> Weights<'a> {
    pub fn new(file: &'a File) -> Weights<'a> {
        Weights { file }
    }

    pub fn find(&self, name: &str) -> Option<&'a Tensor> {
        self.file.tensors.iter().find(|t| t.name == name)
    }

    pub fn get(&self, name: &str) -> Result<&'a Tensor, String> {
        self.find(name)
            .ok_or(format!("the model file has no tensor {name}"))
    }
}

/// One block's weights.
struct Block {
    time1: Linear,
    time2: Linear,
    classes: Mat<f32>,
    ada: Linear,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    ff1: Linear,
    ff2: Linear,
}

pub struct DitXl {
    pub cfg: DitXlConfig,
    /// The patch embedding as a linear layer on each patch's
    /// in_channels·p·p values.
    patch: Linear,
    positions: Mat<f32>,
    blocks: Vec<Block>,
    final_ada: Linear,
    final_out: Linear,
}

impl DitXl {
    pub fn load(file: &File, cfg: DitXlConfig) -> Result<DitXl, String> {
        let f = Weights::new(file);
        let pw = f.get("pos_embed.proj.weight")?;
        let (d, c, p) = (cfg.width, cfg.in_channels, cfg.patch);
        if pw.shape != [d, c, p, p] {
            return Err(format!(
                "pos_embed.proj.weight is {:?}, expected {:?}",
                pw.shape,
                [d, c, p, p]
            ));
        }
        let patch = Linear {
            w: Mat::from_vec(d, c * p * p, pw.data.clone()).transpose(),
            b: f.get("pos_embed.proj.bias")?.data.clone(),
        };
        let mut blocks = Vec::new();
        for i in 0..cfg.blocks {
            let n = |s: &str| format!("transformer_blocks.{i}.{s}");
            let table = f.get(&n("norm1.emb.class_embedder.embedding_table.weight"))?;
            if table.shape != [cfg.classes + 1, d] {
                return Err(format!("block {i}'s class table is {:?}", table.shape));
            }
            blocks.push(Block {
                time1: Linear::load(&f, &n("norm1.emb.timestep_embedder.linear_1"))?,
                time2: Linear::load(&f, &n("norm1.emb.timestep_embedder.linear_2"))?,
                classes: Mat::from_vec(cfg.classes + 1, d, table.data.clone()),
                ada: Linear::load(&f, &n("norm1.linear"))?,
                q: Linear::load(&f, &n("attn1.to_q"))?,
                k: Linear::load(&f, &n("attn1.to_k"))?,
                v: Linear::load(&f, &n("attn1.to_v"))?,
                out: Linear::load(&f, &n("attn1.to_out.0"))?,
                ff1: Linear::load(&f, &n("ff.net.0.proj"))?,
                ff2: Linear::load(&f, &n("ff.net.2"))?,
            });
        }
        Ok(DitXl {
            positions: positions(d, cfg.size / p),
            patch,
            blocks,
            final_ada: Linear::load(&f, "proj_out_1")?,
            final_out: Linear::load(&f, "proj_out_2")?,
            cfg,
        })
    }

    /// The conditioning c for each example, from block `b`'s own copy of
    /// the time network and class table.
    fn conditioning(&self, b: usize, times: &[f32], labels: &[usize], r: Reference) -> Mat<f32> {
        let blk = &self.blocks[b];
        let feats = time_features(times, 256, r);
        let mut h = blk.time1.forward(&feats);
        silu(&mut h.data);
        let mut c = blk.time2.forward(&h);
        for (i, &l) in labels.iter().enumerate() {
            assert!(l <= self.cfg.classes, "class {l} past the table");
            for (v, e) in c.row_mut(i).iter_mut().zip(blk.classes.row(l)) {
                *v += e;
            }
        }
        c
    }

    /// The patch embedding plus positions: B·N token rows of width d.
    pub fn embed(&self, latents: &[f32], batch: usize) -> Mat<f32> {
        let (c, s, p) = (self.cfg.in_channels, self.cfg.size, self.cfg.patch);
        assert_eq!(latents.len(), batch * c * s * s, "latent size");
        let side = s / p;
        let n = side * side;
        let mut rows = Mat::zeros(batch * n, c * p * p);
        for e in 0..batch {
            let lat = &latents[e * c * s * s..(e + 1) * c * s * s];
            for py in 0..side {
                for px in 0..side {
                    let row = rows.row_mut(e * n + py * side + px);
                    for ch in 0..c {
                        for i in 0..p {
                            for j in 0..p {
                                row[(ch * p + i) * p + j] =
                                    lat[(ch * s + py * p + i) * s + px * p + j];
                            }
                        }
                    }
                }
            }
        }
        let mut x = self.patch.forward(&rows);
        for r in 0..x.rows {
            for (v, q) in x.row_mut(r).iter_mut().zip(self.positions.row(r % n)) {
                *v += q;
            }
        }
        x
    }

    /// One block on the token rows.
    pub fn block(&self, b: usize, x: &mut Mat<f32>, times: &[f32], labels: &[usize], r: Reference) {
        let (d, n) = (self.cfg.width, self.cfg.tokens());
        let batch = labels.len();
        let blk = &self.blocks[b];
        let mut c = self.conditioning(b, times, labels, r);
        silu(&mut c.data);
        let m = blk.ada.forward(&c); // batch × 6d
        let part = |e: usize, k: usize| &m.row(e)[k * d..(k + 1) * d];
        let mut h = layer_norm(x, 1e-6);
        modulate(&mut h, n, |e| part(e, 1), |e| part(e, 0));
        let a = blk.out.forward(&attention(
            &blk.q.forward(&h),
            &blk.k.forward(&h),
            &blk.v.forward(&h),
            batch,
            n,
            self.cfg.heads,
        ));
        gated_add(x, &a, n, |e| part(e, 2));
        let eps = match r {
            Reference::Diffusers => 1e-5,
            Reference::Original => 1e-6,
        };
        let mut h = layer_norm(x, eps);
        modulate(&mut h, n, |e| part(e, 4), |e| part(e, 3));
        let mut f = blk.ff1.forward(&h);
        for v in f.data.iter_mut() {
            *v = gelu_tanh(*v);
        }
        let f = blk.ff2.forward(&f);
        gated_add(x, &f, n, |e| part(e, 5));
    }

    /// The final layer and unpatchifying: for each example, out_channels
    /// × size × size values (ε, then the learned variance).
    pub fn finish(&self, x: &Mat<f32>, times: &[f32], labels: &[usize], r: Reference) -> Vec<f32> {
        let (d, n, p, s) = (
            self.cfg.width,
            self.cfg.tokens(),
            self.cfg.patch,
            self.cfg.size,
        );
        let oc = self.cfg.out_channels;
        let mut c = self.conditioning(0, times, labels, r);
        silu(&mut c.data);
        let m = self.final_ada.forward(&c);
        let mut h = layer_norm(x, 1e-6);
        modulate(&mut h, n, |e| &m.row(e)[d..2 * d], |e| &m.row(e)[..d]);
        let y = self.final_out.forward(&h); // B·N × (p·p·oc)
        let side = s / p;
        let batch = labels.len();
        let mut out = vec![0.0; batch * oc * s * s];
        for e in 0..batch {
            for py in 0..side {
                for px in 0..side {
                    let row = y.row(e * n + py * side + px);
                    for i in 0..p {
                        for j in 0..p {
                            for ch in 0..oc {
                                out[((e * oc + ch) * s + py * p + i) * s + px * p + j] =
                                    row[(i * p + j) * oc + ch];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The whole forward pass: latents (batch × in_channels × size ×
    /// size), a time and a class per example.
    pub fn forward(
        &self,
        latents: &[f32],
        times: &[f32],
        labels: &[usize],
        r: Reference,
    ) -> Vec<f32> {
        let mut x = self.embed(latents, labels.len());
        for b in 0..self.cfg.blocks {
            self.block(b, &mut x, times, labels, r);
        }
        self.finish(&x, times, labels, r)
    }
}

/// Generation with DDIM and classifier-free guidance (PLAN.md, stage 7):
/// from `noise` (one latent of in_channels × size × size per row of
/// `labels`), `steps` DDIM steps on DDPM's 1,000-step schedule, the guided
/// ε = ε_u + w·(ε_c − ε_u) on all in_channels channels. `on_step` sees,
/// after each step, the step's index and the model's current guess of the
/// finished latents (x̂₀). Returns the finished latents.
pub fn generate(
    model: &DitXl,
    noise: &[f32],
    labels: &[usize],
    guidance: f32,
    steps: usize,
    r: Reference,
    on_step: impl FnMut(usize, &[f32]),
) -> Vec<f32> {
    let all = model.cfg.in_channels;
    generate_guided(model, noise, labels, guidance, all, steps, r, on_step)
}

/// `generate`, with guidance applied to the first `guide_channels` of ε's
/// channels only; the rest take the conditional prediction alone. The DiT
/// repository's own sampler, by default, guides 3 of the 4 (PLAN.md,
/// stage 7).
#[allow(clippy::too_many_arguments)]
pub fn generate_guided(
    model: &DitXl,
    noise: &[f32],
    labels: &[usize],
    guidance: f32,
    guide_channels: usize,
    steps: usize,
    r: Reference,
    mut on_step: impl FnMut(usize, &[f32]),
) -> Vec<f32> {
    let sched = crate::ddpm::Schedule::linear(1000);
    let c = model.cfg.in_channels;
    let per = c * model.cfg.size * model.cfg.size;
    let batch = labels.len();
    assert_eq!(noise.len(), batch * per, "one latent per label");
    let n = sched.steps();
    // The schedule's steps visited, from N down; the last jump goes to 0.
    // Step i of the schedule is the model's time i − 1 (0 to 999).
    let mut seq: Vec<usize> = (0..steps).map(|k| n - k * n / steps).collect();
    seq.push(0);
    let mut x = noise.to_vec();
    let null = model.cfg.classes;
    for (k, w) in seq.windows(2).enumerate() {
        let (i, prev) = (w[0], w[1]);
        let t = (i - 1) as f32;
        let eps: Vec<f32> = if guidance == 1.0 {
            take_eps(&model.forward(&x, &vec![t; batch], labels, r), batch, per)
        } else {
            let mut both = x.clone();
            both.extend_from_slice(&x);
            let mut lab = labels.to_vec();
            lab.extend(std::iter::repeat_n(null, batch));
            let e = take_eps(
                &model.forward(&both, &vec![t; 2 * batch], &lab, r),
                2 * batch,
                per,
            );
            let (ec, eu) = e.split_at(batch * per);
            let area = model.cfg.size * model.cfg.size;
            ec.iter()
                .zip(eu)
                .enumerate()
                .map(|(j, (c, u))| {
                    if (j % per) / area < guide_channels {
                        u + guidance * (c - u)
                    } else {
                        *c
                    }
                })
                .collect()
        };
        let (ab, abp) = (sched.alpha_bar[i], sched.alpha_bar[prev]);
        let (sa, sb) = (ab.sqrt() as f32, (1.0 - ab).sqrt() as f32);
        let (spa, spb) = (abp.sqrt() as f32, (1.0 - abp).sqrt() as f32);
        let x0: Vec<f32> = x
            .iter()
            .zip(&eps)
            .map(|(xv, ev)| (xv - sb * ev) / sa)
            .collect();
        x = x0
            .iter()
            .zip(&eps)
            .map(|(x0v, ev)| spa * x0v + spb * ev)
            .collect();
        on_step(k, &x0);
    }
    x
}

/// ε from the model's output: the first `per` values of each example's
/// output (the rest is the learned variance, which DDIM doesn't use).
fn take_eps(out: &[f32], batch: usize, per: usize) -> Vec<f32> {
    let each = out.len() / batch;
    assert!(out.len() == batch * each && each >= per, "output shape");
    (0..batch)
        .flat_map(|e| out[e * each..e * each + per].iter().copied())
        .collect()
}

/// The sinusoidal time features,/// The sinusoidal time features, `dim` of them per time: cos then sin of
/// t·fᵢ, fᵢ = 10000^(−i/(half − shift)), shift 1 in diffusers, 0 in the
/// original.
pub fn time_features(times: &[f32], dim: usize, r: Reference) -> Mat<f32> {
    let half = dim / 2;
    let shift = match r {
        Reference::Diffusers => 1.0,
        Reference::Original => 0.0,
    };
    let mut m = Mat::zeros(times.len(), dim);
    for (row, &t) in times.iter().enumerate() {
        for i in 0..half {
            let f = (-(10000f32.ln()) * i as f32 / (half as f32 - shift)).exp();
            let a = t * f;
            m.data[row * dim + i] = a.cos();
            m.data[row * dim + half + i] = a.sin();
        }
    }
    m
}

/// The fixed 2-D sine-cosine positions of a side × side grid of patches,
/// row by row: the first half of each vector encodes the column, the
/// second the row, each as sines then cosines of pos·ωᵢ, ωᵢ =
/// 10000^(−i/(d/4)).
pub fn positions(d: usize, side: usize) -> Mat<f32> {
    let q = d / 4;
    let one = |pos: usize, out: &mut [f32]| {
        for i in 0..q {
            let w = 1.0 / 10000f64.powf(i as f64 / q as f64);
            let a = pos as f64 * w;
            out[i] = a.sin() as f32;
            out[q + i] = a.cos() as f32;
        }
    };
    let mut m = Mat::zeros(side * side, d);
    for r in 0..side {
        for c in 0..side {
            let row = m.row_mut(r * side + c);
            let (cols, rows) = row.split_at_mut(d / 2);
            one(c, cols);
            one(r, rows);
        }
    }
    m
}

/// Layer norm of each row, with no scale or shift of its own.
fn layer_norm(x: &Mat<f32>, eps: f32) -> Mat<f32> {
    let mut y = x.clone();
    for r in 0..x.rows {
        let row = x.row(r);
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inv = 1.0 / (var + eps).sqrt();
        for (o, v) in y.row_mut(r).iter_mut().zip(row) {
            *o = (v - mean) * inv;
        }
    }
    y
}

/// h ← h·(1 + scale) + shift, one scale and shift per example, `n` rows
/// per example.
fn modulate<'m>(
    h: &mut Mat<f32>,
    n: usize,
    scale: impl Fn(usize) -> &'m [f32],
    shift: impl Fn(usize) -> &'m [f32],
) {
    for r in 0..h.rows {
        let (sc, sh) = (scale(r / n), shift(r / n));
        for ((v, a), b) in h.row_mut(r).iter_mut().zip(sc).zip(sh) {
            *v = *v * (1.0 + a) + b;
        }
    }
}

/// x ← x + gate ⊙ y, one gate per example.
fn gated_add<'m>(x: &mut Mat<f32>, y: &Mat<f32>, n: usize, gate: impl Fn(usize) -> &'m [f32]) {
    for r in 0..x.rows {
        let g = gate(r / n);
        for ((v, a), gi) in x.row_mut(r).iter_mut().zip(y.row(r)).zip(g) {
            *v += gi * a;
        }
    }
}

/// Multi-head attention without a mask, each example's n rows on their
/// own.
fn attention(
    q: &Mat<f32>,
    k: &Mat<f32>,
    v: &Mat<f32>,
    batch: usize,
    n: usize,
    heads: usize,
) -> Mat<f32> {
    let d = q.cols;
    let hd = d / heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = Mat::zeros(q.rows, d);
    let take = |m: &Mat<f32>, e: usize, h: usize| {
        let mut s = Mat::zeros(n, hd);
        for i in 0..n {
            s.row_mut(i)
                .copy_from_slice(&m.row(e * n + i)[h * hd..(h + 1) * hd]);
        }
        s
    };
    for e in 0..batch {
        for h in 0..heads {
            let (qh, kh, vh) = (take(q, e, h), take(k, e, h), take(v, e, h));
            let mut w = qh.matmul(&kh.transpose());
            for i in 0..n {
                let row = w.row_mut(i);
                let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut sum = 0.0;
                for x in row.iter_mut() {
                    *x = ((*x - max) * scale).exp();
                    sum += *x;
                }
                for x in row.iter_mut() {
                    *x /= sum;
                }
            }
            let o = w.matmul(&vh);
            for i in 0..n {
                out.row_mut(e * n + i)[h * hd..(h + 1) * hd].copy_from_slice(o.row(i));
            }
        }
    }
    out
}

fn silu(x: &mut [f32]) {
    crate::conv::silu(x)
}

/// GELU's tanh form, as DiT uses.
fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x * x * x)).tanh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors;

    fn fixture() -> (File, DitXlConfig) {
        let f = safetensors::decode(include_bytes!("../tests/data/dit-tiny.safetensors")).unwrap();
        let cfg = DitXlConfig::from_json(&f.metadata["config"]).unwrap();
        (f, cfg)
    }

    fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: lengths");
        let scale = want.iter().fold(0f32, |a, b| a.max(b.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(
            worst <= tol * scale,
            "{what}: worst difference {worst}, largest value {scale}"
        );
    }

    /// The tiny model built by tools/stage7/fixtures.py in diffusers, with
    /// random weights: the same outputs, after the patch embedding, after
    /// each block, and at the end.
    #[test]
    fn matches_diffusers_on_the_tiny_fixture() {
        let (f, cfg) = fixture();
        assert_eq!(
            (
                cfg.width,
                cfg.heads,
                cfg.blocks,
                cfg.patch,
                cfg.size,
                cfg.classes,
                cfg.out_channels
            ),
            (16, 2, 2, 2, 8, 10, 8)
        );
        let m = DitXl::load(&f, cfg.clone()).unwrap();
        let w = Weights::new(&f);
        let latent = &w.get("input.latent").unwrap().data;
        let times = w.get("input.times").unwrap().data.clone();
        let labels: Vec<usize> = w
            .get("input.labels")
            .unwrap()
            .data
            .iter()
            .map(|&v| v as usize)
            .collect();
        assert_eq!(labels, vec![3, 10]);
        let r = Reference::Diffusers;
        let mut x = m.embed(latent, 2);
        close(
            &x.data,
            &w.get("inter.patches").unwrap().data,
            1e-5,
            "patches",
        );
        for b in 0..cfg.blocks {
            m.block(b, &mut x, &times, &labels, r);
            close(
                &x.data,
                &w.get(&format!("inter.block{b}")).unwrap().data,
                1e-4,
                "block",
            );
        }
        let out = m.finish(&x, &times, &labels, r);
        close(&out, &w.get("output").unwrap().data, 1e-4, "output");
        assert_eq!(out, m.forward(latent, &times, &labels, r));
    }

    /// Generation runs on the tiny model: guidance 1 needs no
    /// unconditional pass and equals the conditional model alone, the
    /// guess is reported once per step, and at guidance 0 the label
    /// doesn't matter.
    #[test]
    fn generation_with_guidance() {
        let (f, cfg) = fixture();
        let m = DitXl::load(&f, cfg).unwrap();
        let noise: Vec<f32> = {
            let mut r = crate::rng::Rng::new(3);
            (0..2 * 4 * 64).map(|_| r.normal() as f32).collect()
        };
        let mut seen = 0;
        let a = generate(&m, &noise, &[1, 2], 1.0, 5, Reference::Original, |k, x0| {
            assert_eq!((k, x0.len()), (seen, 2 * 4 * 64));
            seen += 1;
        });
        assert_eq!(seen, 5);
        assert!(a.iter().all(|v| v.is_finite()));
        let b = generate(&m, &noise, &[1, 2], 3.0, 5, Reference::Original, |_, _| {});
        assert_ne!(a, b);
        let u1 = generate(&m, &noise, &[1, 2], 0.0, 5, Reference::Original, |_, _| {});
        let u2 = generate(&m, &noise, &[7, 9], 0.0, 5, Reference::Original, |_, _| {});
        let diff = u1
            .iter()
            .zip(&u2)
            .fold(0f32, |x, (p, q)| x.max((p - q).abs()));
        assert!(diff < 1e-4, "{diff}");
    }

    /// The original arithmetic differs from diffusers', a little.
    #[test]
    fn the_two_references_differ_slightly() {
        let (f, cfg) = fixture();
        let m = DitXl::load(&f, cfg).unwrap();
        let w = Weights::new(&f);
        let latent = &w.get("input.latent").unwrap().data;
        let times = w.get("input.times").unwrap().data.clone();
        let a = m.forward(latent, &times, &[3, 10], Reference::Diffusers);
        let b = m.forward(latent, &times, &[3, 10], Reference::Original);
        let diff = a
            .iter()
            .zip(&b)
            .fold(0f32, |x, (p, q)| x.max((p - q).abs()));
        assert!(diff > 1e-4, "{diff}");
        // At t = 0 the time features agree; the frequencies differ above it.
        let (fa, fb) = (
            time_features(&[0.0, 500.0], 256, Reference::Diffusers),
            time_features(&[0.0, 500.0], 256, Reference::Original),
        );
        assert_eq!(fa.row(0), fb.row(0));
        assert_ne!(fa.row(1), fb.row(1));
        assert_eq!((fa.row(1)[0], fb.row(1)[0]), (500f32.cos(), 500f32.cos()));
    }

    /// The positions' closed form: position (row 0, column 0) is sin 0 =
    /// 0 and cos 0 = 1 in every pair; the column half changes along a
    /// row and the row half doesn't.
    #[test]
    fn positions_closed_form() {
        let p = positions(16, 4);
        assert_eq!(
            p.row(0),
            &[0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1.][..]
        );
        assert!((p.row(1)[0] - 1f32.sin()).abs() < 1e-7);
        assert_eq!(p.row(1)[8..], p.row(0)[8..]);
        assert!((p.row(4)[8] - 1f32.sin()).abs() < 1e-7);
        assert!(
            (p.row(4)[9] - 0.1f32.sin()).abs() < 1e-7,
            "ω₁ = 10000^(−1/4)"
        );
    }
}
