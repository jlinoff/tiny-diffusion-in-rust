//! The masked diffusion language model of stage 6 (PLAN.md): tiny-llm-in-
//! rust's GPT-2-style network with attention made bidirectional and one
//! more token, MASK, trained to recover masked tokens.
//!
//! ```text
//! x = token row + position row                      (one row per position)
//! each block:  x += Proj(Attention(LN(x)))          (every position sees all)
//!              x += W₂·GELU(W₁·LN(x))
//! logits = LN(x) · token tableᵀ                     (tied, over the V real tokens)
//! ```
//!
//! Generic over the float type, like the other networks: f32 to train, f64
//! for the gradient check.

use crate::optim::{clip_gradients, Adam};
use crate::rng::Rng;
use crate::safetensors::{File, Tensor};
use crate::tape::{Tape, Var};
use crate::tensor::{Float, Mat};
use crate::text::window_starts;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextConfig {
    /// V, the real tokens; MASK is token V.
    pub vocab: usize,
    /// d, each position's vector.
    pub width: usize,
    pub blocks: usize,
    pub heads: usize,
    /// The positions the model sees at once: 256, as tiny-llm-in-rust's.
    pub context: usize,
    /// The feed-forward network's inner width.
    pub ffn: usize,
}

impl TextConfig {
    pub fn mask(&self) -> usize {
        self.vocab
    }

    /// Every weight's name and shape, in order.
    pub fn layout(&self) -> Vec<(String, usize, usize)> {
        let (d, f) = (self.width, self.ffn);
        let mut v = vec![
            ("tok".to_string(), self.vocab + 1, d),
            ("pos".to_string(), self.context, d),
        ];
        for l in 0..self.blocks {
            for (n, r, c) in [
                ("ln1.g", 1, d),
                ("ln1.b", 1, d),
                ("qkv.w", d, 3 * d),
                ("qkv.b", 1, 3 * d),
                ("proj.w", d, d),
                ("proj.b", 1, d),
                ("ln2.g", 1, d),
                ("ln2.b", 1, d),
                ("ffn.w1", d, f),
                ("ffn.b1", 1, f),
                ("ffn.w2", f, d),
                ("ffn.b2", 1, d),
            ] {
                v.push((format!("block{l}.{n}"), r, c));
            }
        }
        v.push(("final.g".into(), 1, d));
        v.push(("final.b".into(), 1, d));
        v
    }

    pub fn parameters(&self) -> usize {
        self.layout().iter().map(|(_, r, c)| r * c).sum()
    }

    /// Which weights AdamW's decay applies to: matrices and embeddings, not
    /// biases or layer-norm scales.
    pub fn decayed(&self) -> Vec<bool> {
        self.layout()
            .iter()
            .map(|(n, _, _)| matches!(last_part(n), "tok" | "pos" | "w" | "w1" | "w2"))
            .collect()
    }
}

/// The part of a weight's name after its last dot: "w", "b1", "g", ….
fn last_part(name: &str) -> &str {
    name.rsplit('.').next().unwrap()
}

pub struct TextModel<T> {
    pub cfg: TextConfig,
    pub params: Vec<Mat<T>>,
}

impl<T: Float> TextModel<T> {
    /// GPT-2's initial values, as tiny-llm-in-rust's: matrices and
    /// embeddings normal(0, 0.02), the two that write into the residual
    /// stream scaled by 1/√(2·blocks), biases 0, layer-norm scales 1.
    pub fn new(cfg: TextConfig, seed: u64) -> TextModel<T> {
        let mut rng = Rng::new(seed);
        let res = 0.02 / ((2 * cfg.blocks) as f64).sqrt();
        let params = cfg
            .layout()
            .iter()
            .map(|(name, r, c)| {
                let part = last_part(name);
                if part == "g" {
                    Mat::from_vec(*r, *c, vec![T::one(); r * c])
                } else if matches!(part, "b" | "b1" | "b2") {
                    Mat::zeros(*r, *c)
                } else {
                    let s = if name.ends_with("proj.w") || name.ends_with("ffn.w2") {
                        res
                    } else {
                        0.02
                    };
                    Mat::from_vec(
                        *r,
                        *c,
                        (0..r * c).map(|_| T::from_f64(s * rng.normal())).collect(),
                    )
                }
            })
            .collect();
        TextModel { cfg, params }
    }

    /// The forward pass for `ids`, sequences of equal length one after
    /// another (B·T of them, T ≤ context): the logits over the V real
    /// tokens at every position, and the parameters' variables.
    pub fn forward(&self, tape: &mut Tape<T>, ids: &[usize], batch: usize) -> (Var, Vec<Var>) {
        let c = self.cfg;
        let seq = ids.len() / batch;
        assert!(
            seq * batch == ids.len() && seq <= c.context,
            "sequence shapes"
        );
        assert!(
            ids.iter().all(|&i| i <= c.vocab),
            "token ids run 0 to V (MASK)"
        );
        let p: Vec<Var> = self.params.iter().map(|m| tape.leaf(m.clone())).collect();
        let lin = |tape: &mut Tape<T>, x: Var, w: usize| {
            let z = tape.matmul(x, p[w]);
            tape.add_row(z, p[w + 1])
        };
        let tok = tape.gather(p[0], ids);
        let positions: Vec<usize> = (0..batch).flat_map(|_| 0..seq).collect();
        let pos = tape.gather(p[1], &positions);
        let mut x = tape.add(tok, pos);
        let d = c.width;
        for l in 0..c.blocks {
            let w = 2 + 12 * l;
            let n = tape.layer_norm(x, p[w], p[w + 1]);
            let qkv = lin(tape, n, w + 2);
            let (q, k, v) = (
                tape.slice_cols(qkv, 0, d),
                tape.slice_cols(qkv, d, d),
                tape.slice_cols(qkv, 2 * d, d),
            );
            let a = tape.attention_full(q, k, v, seq, c.heads);
            let a = lin(tape, a, w + 4);
            x = tape.add(x, a);
            let n = tape.layer_norm(x, p[w + 6], p[w + 7]);
            let f = lin(tape, n, w + 8);
            let f = tape.gelu(f);
            let f = lin(tape, f, w + 10);
            x = tape.add(x, f);
        }
        let last = 2 + 12 * c.blocks;
        let h = tape.layer_norm(x, p[last], p[last + 1]);
        // The output is tied to the token table, its V real rows.
        let real: Vec<usize> = (0..c.vocab).collect();
        let table = tape.gather(p[0], &real);
        (tape.matmul_nt(h, table), p)
    }
}

impl TextModel<f32> {
    /// The logits without keeping the tape.
    pub fn logits(&self, ids: &[usize], batch: usize) -> Mat<f32> {
        let mut tape = Tape::new();
        let (l, _) = self.forward(&mut tape, ids, batch);
        tape.value(l).clone()
    }

    pub fn to_file(&self, mut metadata: BTreeMap<String, String>) -> File {
        let c = self.cfg;
        for (k, v) in [
            ("vocab", c.vocab),
            ("width", c.width),
            ("blocks", c.blocks),
            ("heads", c.heads),
            ("context", c.context),
            ("ffn", c.ffn),
        ] {
            metadata.insert(k.into(), v.to_string());
        }
        metadata.insert("kind".into(), "masked-text".into());
        let tensors = c
            .layout()
            .iter()
            .zip(&self.params)
            .map(|((n, r, cl), p)| Tensor::new(n, &[*r, *cl], p.data.clone()))
            .collect();
        File { tensors, metadata }
    }

    pub fn from_file(f: &File) -> Result<TextModel<f32>, String> {
        if f.metadata.get("kind").map(String::as_str) != Some("masked-text") {
            return Err("not a masked text model (make text writes one)".into());
        }
        let num = |k: &str| -> Result<usize, String> {
            f.metadata
                .get(k)
                .ok_or(format!("the model file has no {k:?} in its metadata"))?
                .parse()
                .map_err(|_| format!("the model file's {k:?} isn't a number"))
        };
        let cfg = TextConfig {
            vocab: num("vocab")?,
            width: num("width")?,
            blocks: num("blocks")?,
            heads: num("heads")?,
            context: num("context")?,
            ffn: num("ffn")?,
        };
        let mut params = Vec::new();
        for (name, r, c) in cfg.layout() {
            let t = f
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or(format!("the model file has no tensor {name}"))?;
            if t.shape != [r, c] {
                return Err(format!("{name} is {:?}, expected {:?}", t.shape, [r, c]));
            }
            params.push(Mat::from_vec(r, c, t.data.clone()));
        }
        Ok(TextModel { cfg, params })
    }
}

/// Mask a window at time t: each token hidden with probability 1 − t.
/// Returns the inputs (MASK where hidden) and the targets (the original
/// token where hidden, None elsewhere). With `at_least_one`, a window that
/// drew no mask gets one, so every training window teaches something.
pub fn mask_window(
    window: &[u32],
    t: f64,
    mask: usize,
    rng: &mut Rng,
    at_least_one: bool,
) -> (Vec<usize>, Vec<Option<usize>>) {
    let mut inputs = Vec::with_capacity(window.len());
    let mut targets = Vec::with_capacity(window.len());
    for &tok in window {
        if rng.unit() < 1.0 - t {
            inputs.push(mask);
            targets.push(Some(tok as usize));
        } else {
            inputs.push(tok as usize);
            targets.push(None);
        }
    }
    if at_least_one && targets.iter().all(Option::is_none) {
        let i = rng.below(window.len());
        inputs[i] = mask;
        targets[i] = Some(window[i] as usize);
    }
    (inputs, targets)
}

/// The settings of a training run, tiny-llm-in-rust's [training] section's.
pub struct TextTraining {
    pub steps: u64,
    pub batch: usize,
    pub learning_rate: f32,
    pub warmup: u64,
    pub min_lr_fraction: f32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub beta2: f32,
    pub seed: u64,
}

impl TextTraining {
    pub fn lr(&self, step: u64) -> f32 {
        let peak = self.learning_rate;
        if step <= self.warmup {
            return peak * step as f32 / self.warmup.max(1) as f32;
        }
        let p = (step - self.warmup) as f32 / (self.steps - self.warmup).max(1) as f32;
        let m = self.min_lr_fraction;
        peak * (m + (1.0 - m) * 0.5 * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos()))
    }
}

/// Train on `data` (the training tokens): each step, tiny-llm-in-rust's
/// windows, each masked at its own random t, and the cross-entropy of the
/// masked positions' predictions. Returns each step's loss.
pub fn train(
    model: &mut TextModel<f32>,
    data: &[u32],
    c: &TextTraining,
    mut report: impl FnMut(u64, f32),
) -> Vec<f32> {
    let seq = model.cfg.context;
    let decay = model.cfg.decayed();
    let mut adam = Adam::new(&model.params);
    adam.beta2 = c.beta2;
    let mut losses = Vec::with_capacity(c.steps as usize);
    for step in 1..=c.steps {
        let starts = window_starts(data.len(), c.batch, seq, c.seed, step);
        let mut rng = Rng::stream(c.seed ^ 0x7E87, step);
        let (mut ids, mut targets) = (Vec::new(), Vec::new());
        for &s in &starts {
            let t = rng.unit();
            let (i, tg) = mask_window(&data[s..s + seq], t, model.cfg.mask(), &mut rng, true);
            ids.extend(i);
            targets.extend(tg);
        }
        let mut tape = Tape::new();
        let (logits, vars) = model.forward(&mut tape, &ids, c.batch);
        let loss = tape.cross_entropy(logits, &targets);
        let value = tape.value(loss).data[0];
        tape.backward(loss);
        let mut grads: Vec<Mat<f32>> = vars.iter().map(|&v| tape.take_grad(v)).collect();
        clip_gradients(&mut grads, c.grad_clip);
        adam.update_decayed(
            &mut model.params,
            &grads,
            c.lr(step),
            &decay,
            c.weight_decay,
        );
        losses.push(value);
        report(step, value);
    }
    losses
}

/// −log p(target) from one row of logits.
fn nll(row: &[f32], target: usize) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let total: f64 = row.iter().map(|&x| (x as f64 - max).exp()).sum();
    max + total.ln() - row[target] as f64
}

/// The diffusion bound on held-out windows, in nats per token (PLAN.md,
/// stage 6): for each window, `draws` values of t, stratified over (0, 1),
/// each masking the window and scoring the masked tokens; the sum of their
/// −log p, divided by 1 − t and by the window's length, averaged. Returns
/// the bound and its standard error over windows.
pub fn diffusion_bound(
    model: &TextModel<f32>,
    windows: &[&[u32]],
    draws: usize,
    seed: u64,
) -> (f64, f64) {
    let mask = model.cfg.mask();
    let mut per_window = Vec::with_capacity(windows.len());
    for (w_i, w) in windows.iter().enumerate() {
        let mut rng = Rng::stream(seed, w_i as u64);
        let (mut ids, mut targets, mut ts) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..draws {
            let t = (k as f64 + rng.unit()) / draws as f64;
            let (i, tg) = mask_window(w, t, mask, &mut rng, false);
            ids.extend(i);
            targets.push(tg);
            ts.push(t);
        }
        let logits = model.logits(&ids, draws);
        let len = w.len();
        let mut total = 0.0;
        for k in 0..draws {
            let mut s = 0.0;
            for (j, tg) in targets[k].iter().enumerate() {
                if let Some(y) = *tg {
                    s += nll(logits.row(k * len + j), y);
                }
            }
            total += s / (1.0 - ts[k]) / len as f64;
        }
        per_window.push(total / draws as f64);
    }
    let n = per_window.len() as f64;
    let mean = per_window.iter().sum::<f64>() / n;
    let var = per_window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

/// Sampling settings, tiny-llm-in-rust's [generate] defaults.
#[derive(Clone, Copy)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
}

/// A token drawn from one row of logits with temperature and top-k, and
/// its probability under that distribution (the confidence).
fn draw(row: &[f32], s: Sampling, rng: &mut Rng) -> (usize, f32) {
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    let k = if s.top_k == 0 {
        row.len()
    } else {
        s.top_k.min(row.len())
    };
    let top = &idx[..k];
    let max = row[top[0]];
    let w: Vec<f64> = top
        .iter()
        .map(|&i| (((row[i] - max) / s.temperature.max(1e-6)) as f64).exp())
        .collect();
    let total: f64 = w.iter().sum();
    let mut u = rng.unit() * total;
    for (j, &wi) in w.iter().enumerate() {
        if u < wi || j + 1 == k {
            return (top[j], (wi / total) as f32);
        }
        u -= wi;
    }
    unreachable!()
}

/// One generation: the tokens, and the step at which each position was
/// settled (None for positions given, never masked).
pub struct Generated {
    pub tokens: Vec<usize>,
    pub settled_at: Vec<Option<usize>>,
}

/// Generate by unmasking, one sequence of `len` positions: those given in
/// `fixed` (Some) are kept, the rest start as MASK; over `steps` steps,
/// each predicts every masked position, draws a token at each, and keeps
/// the most confident, as many as bring the settled share to (step + 1) /
/// steps of the masked positions; the rest are masked again.
pub fn generate(
    model: &TextModel<f32>,
    fixed: &[Option<usize>],
    steps: usize,
    s: Sampling,
    rng: &mut Rng,
) -> Generated {
    let mask = model.cfg.mask();
    let len = fixed.len();
    let mut tokens: Vec<usize> = fixed.iter().map(|f| f.unwrap_or(mask)).collect();
    let mut settled_at: Vec<Option<usize>> = vec![None; len];
    let open: Vec<usize> = (0..len).filter(|&i| fixed[i].is_none()).collect();
    let total = open.len();
    let mut done = 0;
    for step in 0..steps {
        let target = ((step + 1) * total).div_ceil(steps);
        let target = if step + 1 == steps {
            total
        } else {
            target.min(total)
        };
        let want = target.saturating_sub(done);
        if want == 0 {
            continue;
        }
        let logits = model.logits(&tokens, 1);
        let mut cands: Vec<(f32, usize, usize)> = open
            .iter()
            .filter(|&&i| tokens[i] == mask && settled_at[i].is_none())
            .map(|&i| {
                let (tok, p) = draw(logits.row(i), s, rng);
                (p, i, tok)
            })
            .collect();
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(_, i, tok) in cands.iter().take(want) {
            tokens[i] = tok;
            settled_at[i] = Some(step);
        }
        done += want.min(cands.len());
    }
    Generated { tokens, settled_at }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::text::Corpus;

    fn small() -> TextConfig {
        TextConfig {
            vocab: 11,
            width: 8,
            blocks: 2,
            heads: 2,
            context: 6,
            ffn: 12,
        }
    }

    /// tiny-llm-in-rust's development model's size, plus MASK's row
    /// (PLAN.md, stage 6), and the count against its formula.
    #[test]
    fn parameter_count() {
        let c = TextConfig {
            vocab: 4096,
            width: 128,
            blocks: 4,
            heads: 4,
            context: 256,
            ffn: 512,
        };
        assert_eq!(c.parameters(), 1_350_528);
        let (v, d, l, t, f) = (4097, 128, 4, 256, 512);
        let formula = v * d
            + t * d
            + l * (2 * d + 3 * d * d + 3 * d + d * d + d + 2 * d + 2 * d * f + f + d)
            + 2 * d;
        assert_eq!(formula, c.parameters());
        let m = TextModel::<f32>::new(small(), 1);
        let total: usize = m.params.iter().map(|p| p.data.len()).sum();
        assert_eq!(total, small().parameters());
        let decayed: Vec<String> = small()
            .layout()
            .iter()
            .zip(small().decayed())
            .filter(|(_, d)| *d)
            .map(|((n, _, _), _)| n.clone())
            .collect();
        assert!(
            decayed.contains(&"tok".to_string()) && decayed.contains(&"block0.qkv.w".to_string())
        );
        assert!(!decayed
            .iter()
            .any(|n| n.ends_with(".g") || n.ends_with(".b")));
    }

    #[test]
    fn gradients_match_finite_differences() {
        let mut m = TextModel::<f64>::new(small(), 2);
        let mut rng = Rng::new(3);
        for p in m.params.iter_mut() {
            for x in p.data.iter_mut() {
                *x += 0.2 * rng.normal();
            }
        }
        let ids = [3, 11, 5, 11, 0, 7, 11, 2, 9, 1, 11, 4]; // two sequences of 6; 11 is MASK
        let targets: Vec<Option<usize>> = ids
            .iter()
            .enumerate()
            .map(|(i, &t)| if t == 11 { Some(i % 11) } else { None })
            .collect();
        let loss_of = |m: &TextModel<f64>| {
            let mut t = Tape::new();
            let (l, _) = m.forward(&mut t, &ids, 2);
            let loss = t.cross_entropy(l, &targets);
            t.value(loss).data[0]
        };
        let mut t = Tape::new();
        let (l, vars) = m.forward(&mut t, &ids, 2);
        let loss = t.cross_entropy(l, &targets);
        t.backward(loss);
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

    /// Bidirectional: changing a later token changes an earlier position's
    /// prediction, which the causal mask would forbid.
    #[test]
    fn attention_sees_both_ways() {
        let m = TextModel::<f32>::new(small(), 4);
        let a = m.logits(&[1, 2, 3, 4, 5, 6], 1);
        let b = m.logits(&[1, 2, 3, 4, 5, 9], 1);
        assert_ne!(a.row(0), b.row(0));
    }

    #[test]
    fn masking() {
        let w: Vec<u32> = (0..10000).map(|i| (i % 7) as u32).collect();
        let mut rng = Rng::new(5);
        let (ids, tg) = mask_window(&w, 0.3, 99, &mut rng, false);
        let share = ids.iter().filter(|&&i| i == 99).count() as f64 / w.len() as f64;
        assert!((share - 0.7).abs() < 0.02, "{share}");
        for ((&i, t), &orig) in ids.iter().zip(&tg).zip(&w) {
            assert_eq!(i == 99, t.is_some());
            if let Some(y) = t {
                assert_eq!(*y, orig as usize);
            }
        }
        let (_, tg) = mask_window(&[1, 2, 3], 1.0, 99, &mut rng, true);
        assert_eq!(tg.iter().filter(|t| t.is_some()).count(), 1);
        let (_, tg) = mask_window(&[1, 2, 3], 1.0, 99, &mut rng, false);
        assert!(tg.iter().all(Option::is_none));
    }

    /// A model whose real token rows are all zero gives every token the
    /// same score, probability 1/V: its diffusion bound is log V (PLAN.md).
    #[test]
    fn bound_of_a_uniform_model_is_log_v() {
        let c = TextConfig {
            vocab: 50,
            width: 8,
            blocks: 1,
            heads: 2,
            context: 32,
            ffn: 8,
        };
        let mut m = TextModel::<f32>::new(c, 6);
        for x in m.params[0].data[..50 * 8].iter_mut() {
            *x = 0.0;
        }
        let data: Vec<u32> = (0..32 * 40).map(|i| (i * 7 % 50) as u32).collect();
        let windows: Vec<&[u32]> = data.chunks_exact(32).collect();
        let (b, se) = diffusion_bound(&m, &windows, 16, 1);
        let want = (50f64).ln();
        assert!((b - want).abs() < 0.03 * want, "{b} ± {se}, want {want}");
    }

    #[test]
    fn generation_settles_every_position_once() {
        let m = TextModel::<f32>::new(small(), 7);
        let fixed = [None, Some(4), None, None, Some(2), None];
        for steps in [1, 2, 4, 10] {
            let g = generate(
                &m,
                &fixed,
                steps,
                Sampling {
                    temperature: 0.8,
                    top_k: 5,
                },
                &mut Rng::new(8),
            );
            assert!(g.tokens.iter().all(|&t| t < 11), "nothing left masked");
            assert_eq!((g.tokens[1], g.tokens[4]), (4, 2), "given tokens kept");
            for (i, s) in g.settled_at.iter().enumerate() {
                assert_eq!(s.is_some(), fixed[i].is_none());
                assert!(s.is_none_or(|k| k < steps));
            }
        }
    }

    /// A golden test, as for the images.
    #[test]
    fn golden_text_losses() {
        // The first is about log V = log 317 = 5.76: the untrained model is
        // nearly uniform.
        const GOLDEN: [f32; 3] = [5.76344, 5.116874, 4.715004];
        let c = Corpus::decode(include_bytes!("../tests/data/tiny.ids")).unwrap();
        let cfg = TextConfig {
            vocab: c.vocab_size,
            width: 16,
            blocks: 2,
            heads: 2,
            context: 32,
            ffn: 32,
        };
        let mut m = TextModel::new(cfg, 1);
        let tc = TextTraining {
            steps: 40,
            batch: 4,
            learning_rate: 3e-3,
            warmup: 5,
            min_lr_fraction: 0.1,
            weight_decay: 0.1,
            grad_clip: 1.0,
            beta2: 0.95,
            seed: 1,
        };
        let l = train(&mut m, &c.ids, &tc, |_, _| {});
        let got = [l[0], l[19], l[39]];
        println!("text golden: {got:?}");
        for (g, w) in got.iter().zip(GOLDEN) {
            assert!((g - w).abs() < 2e-3, "{got:?}, golden {GOLDEN:?}");
        }
    }

    #[test]
    fn files_round_trip() {
        let m = TextModel::<f32>::new(small(), 9);
        let back = TextModel::from_file(&m.to_file(BTreeMap::new())).unwrap();
        assert_eq!((back.cfg, &back.params), (m.cfg, &m.params));
    }
}
