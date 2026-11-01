//! Training the DiT on images by flow matching (PLAN.md, stage 3): the same
//! loop as stage 2's, with images in place of points and patches in place
//! of coordinates.

use crate::data2d::noise;
use crate::dit::Dit;
use crate::flow::{interpolate, target};
use crate::image::{patchify, read_idx, shapes, Images, Shape};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tape::Tape;
use crate::tensor::Mat;
use std::path::Path;

/// Where the images come from.
pub enum Source {
    /// A downloaded set (MNIST, Fashion-MNIST), all in memory.
    Fixed(Images),
    /// Synthetic shapes, drawn fresh for every batch: unlimited data at any
    /// size.
    Shapes { size: usize, channels: usize },
}

impl Source {
    /// The data set by name: mnist or fashion (from `data/<name>/`, the
    /// Makefile's download), or shapes.
    pub fn open(
        name: &str,
        data_dir: &Path,
        size: usize,
        channels: usize,
    ) -> Result<Source, String> {
        match name {
            "mnist" | "fashion" => Ok(Source::Fixed(labelled(
                name, data_dir, size, channels, false,
            )?)),
            "shapes" => Ok(Source::Shapes { size, channels }),
            _ => Err(format!(
                "unknown data set {name:?}: mnist, fashion or shapes"
            )),
        }
    }

    pub fn shape(&self) -> Shape {
        match self {
            Source::Fixed(s) => s.shape,
            Source::Shapes { size, channels } => Shape {
                height: *size,
                width: *size,
                channels: *channels,
            },
        }
    }

    /// `n` images, from a random stream fixed by the seed and `stream`.
    pub fn batch(&self, n: usize, seed: u64, stream: u64) -> Mat<f32> {
        self.labelled_batch(n, seed, stream).0
    }

    /// The same images, with their labels.
    pub fn labelled_batch(&self, n: usize, seed: u64, stream: u64) -> (Mat<f32>, Vec<usize>) {
        let mut rng = Rng::stream(seed, stream);
        match self {
            Source::Fixed(s) => {
                let rows: Vec<usize> = (0..n).map(|_| rng.below(s.pixels.rows)).collect();
                let labels = rows.iter().map(|&r| s.labels[r] as usize).collect();
                (s.batch(&rows), labels)
            }
            Source::Shapes { size, channels } => {
                let set = shapes(n, *size, *channels, rng.next_u64());
                (set.pixels, set.labels.iter().map(|&l| l as usize).collect())
            }
        }
    }

    /// How many classes the labels run over: 10 for the digits and the
    /// clothing, 3 for the shapes.
    pub fn classes(&self) -> usize {
        match self {
            Source::Fixed(s) => 1 + *s.labels.iter().max().unwrap_or(&0) as usize,
            Source::Shapes { .. } => 3,
        }
    }
}

/// A labelled set by name, its training images or (`test`) its separate
/// test images: mnist and fashion from `data/<name>/` (the Makefile's
/// downloads), or shapes, drawn here (20,000 to train, 2,000 to test, from
/// different seeds).
pub fn labelled(
    name: &str,
    data_dir: &Path,
    size: usize,
    channels: usize,
    test: bool,
) -> Result<Images, String> {
    match name {
        "mnist" | "fashion" => {
            let d = data_dir.join(name);
            let (prefix, target) = if test {
                ("t10k", format!("data/{name}/t10k-images-idx3-ubyte"))
            } else {
                ("train", format!("data/{name}"))
            };
            read_idx(
                &d.join(format!("{prefix}-images-idx3-ubyte")),
                &d.join(format!("{prefix}-labels-idx1-ubyte")),
            )
            .map_err(|e| format!("{e} (download it with: make {target})"))
        }
        "shapes" => Ok(if test {
            shapes(2000, size, channels, 2)
        } else {
            shapes(20000, size, channels, 1)
        }),
        _ => Err(format!(
            "unknown data set {name:?}: mnist, fashion or shapes"
        )),
    }
}

pub struct ImgTrainConfig {
    pub steps: u64,
    pub batch: usize,
    pub learning_rate: f32,
    pub warmup: u64,
    pub seed: u64,
    /// For a model with labels: the probability that an image's label is
    /// replaced by "no label" (label dropout, stage 4).
    pub drop: f32,
}

/// Replace each label by `none` with probability `drop`, from a random
/// stream of its own (so a model without labels trains exactly as before).
pub fn drop_labels(labels: &mut [usize], none: usize, drop: f32, seed: u64, stream: u64) {
    let mut rng = Rng::stream(seed ^ 0x1ABE1, stream);
    for y in labels.iter_mut() {
        if (rng.unit() as f32) < drop {
            *y = none;
        }
    }
}

impl ImgTrainConfig {
    /// Warm-up, then cosine decay to a tenth, as in stage 2.
    pub fn lr(&self, step: u64) -> f32 {
        let peak = self.learning_rate;
        if step <= self.warmup {
            return peak * step as f32 / self.warmup.max(1) as f32;
        }
        let p = (step - self.warmup) as f32 / (self.steps - self.warmup).max(1) as f32;
        peak * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos()))
    }
}

/// Train `model` on `source`; `report(step, loss)` after every step, and
/// `checkpoint(step, model)` every `save_every` steps (0: never), so a long
/// run can be sampled from while it continues.
pub fn train(
    model: &mut Dit<f32>,
    source: &Source,
    c: &ImgTrainConfig,
    save_every: u64,
    mut report: impl FnMut(u64, f32),
    mut checkpoint: impl FnMut(u64, &Dit<f32>),
) -> Vec<f32> {
    let (shape, patch) = (model.cfg.image, model.cfg.patch);
    assert_eq!(
        shape,
        source.shape(),
        "the model's image shape must match the data's"
    );
    let mut adam = Adam::new(&model.params);
    let mut losses = Vec::with_capacity(c.steps as usize);
    let classes = model.cfg.labels;
    if classes > 0 {
        assert_eq!(
            classes,
            source.classes(),
            "the model's labels must match the data's"
        );
    }
    for step in 1..=c.steps {
        let (x1, mut labels) = source.labelled_batch(c.batch, c.seed, step);
        if classes == 0 {
            labels.clear();
        } else {
            drop_labels(&mut labels, classes, c.drop, c.seed, step);
        }
        let mut rng = Rng::stream(c.seed ^ 0x5EED, step);
        let x0 = noise(c.batch, shape.values(), &mut rng);
        let t: Vec<f32> = (0..c.batch).map(|_| rng.unit() as f32).collect();
        let xt = interpolate(&x0, &x1, &t);
        let tgt = patchify(&target(&x0, &x1), shape, patch);
        let mut tape = Tape::new();
        let (out, vars) = model.forward(&mut tape, patchify(&xt, shape, patch), &t, &labels);
        let loss = tape.mse(out, &tgt);
        let value = tape.value(loss).data[0];
        tape.backward(loss);
        let grads: Vec<Mat<f32>> = vars.iter().map(|&v| tape.take_grad(v)).collect();
        adam.update(&mut model.params, &grads, c.lr(step));
        losses.push(value);
        report(step, value);
        if save_every > 0 && step % save_every == 0 && step < c.steps {
            checkpoint(step, model);
        }
    }
    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dit::DitConfig;

    fn small() -> DitConfig {
        DitConfig {
            image: Shape {
                height: 8,
                width: 8,
                channels: 1,
            },
            patch: 4,
            width: 16,
            blocks: 2,
            heads: 2,
            freqs: 2,
            labels: 0,
        }
    }

    fn cfg(steps: u64) -> ImgTrainConfig {
        ImgTrainConfig {
            steps,
            batch: 16,
            learning_rate: 2e-3,
            warmup: 10,
            seed: 1,
            drop: 0.1,
        }
    }

    /// A golden test, as in stage 2: a fixed seed reproduces recorded
    /// losses. If a change to training is meant, rerun with
    /// `cargo test golden -- --nocapture` and update the values.
    #[test]
    fn golden_losses_on_shapes() {
        const GOLDEN: [f32; 3] = [2.113295, 1.629856, 1.425706];
        let source = Source::Shapes {
            size: 8,
            channels: 1,
        };
        let mut m = Dit::new(small(), 1);
        let l = train(&mut m, &source, &cfg(40), 0, |_, _| {}, |_, _| {});
        let got = [l[0], l[19], l[39]];
        println!("dit golden: {got:?}");
        for (g, w) in got.iter().zip(GOLDEN) {
            assert!((g - w).abs() < 2e-3, "{got:?}, golden {GOLDEN:?}");
        }
    }

    /// The same for a model with labels (stage 4), with label dropout.
    #[test]
    fn golden_losses_with_labels() {
        // The first equals the unlabelled model's: untrained, both predict 0.
        const GOLDEN: [f32; 3] = [2.113295, 1.625234, 1.41897];
        let source = Source::Shapes {
            size: 8,
            channels: 1,
        };
        let mut c = small();
        c.labels = 3;
        let mut m = Dit::new(c, 1);
        let l = train(&mut m, &source, &cfg(40), 0, |_, _| {}, |_, _| {});
        let got = [l[0], l[19], l[39]];
        println!("labelled dit golden: {got:?}");
        for (g, w) in got.iter().zip(GOLDEN) {
            assert!((g - w).abs() < 2e-3, "{got:?}, golden {GOLDEN:?}");
        }
    }

    /// Label dropout replaces about the requested share of labels, and only
    /// with "no label".
    #[test]
    fn label_dropout_rate() {
        let mut ys: Vec<usize> = (0..20000).map(|i| i % 10).collect();
        drop_labels(&mut ys, 10, 0.1, 3, 1);
        let dropped = ys.iter().filter(|&&y| y == 10).count() as f64 / 20000.0;
        assert!((dropped - 0.1).abs() < 0.01, "{dropped}");
        let mut none = vec![4usize; 1000];
        drop_labels(&mut none, 10, 0.0, 3, 1);
        assert!(none.iter().all(|&y| y == 4));
    }

    /// Training lowers the loss: a smoke check that the DiT learns at all.
    #[test]
    fn training_lowers_the_loss() {
        let source = Source::Shapes {
            size: 8,
            channels: 1,
        };
        let mut m = Dit::new(small(), 2);
        let l = train(&mut m, &source, &cfg(150), 0, |_, _| {}, |_, _| {});
        let first: f32 = l[..10].iter().sum::<f32>() / 10.0;
        let last: f32 = l[l.len() - 10..].iter().sum::<f32>() / 10.0;
        assert!(last < 0.8 * first, "{first} -> {last}");
    }

    #[test]
    fn sources_by_name() {
        assert!(Source::open("nonsense", Path::new("."), 8, 1).is_err());
        let e = Source::open("mnist", Path::new("/no/such/dir"), 8, 1)
            .err()
            .unwrap();
        assert!(e.contains("make data/mnist"), "{e}");
        let s = Source::open("shapes", Path::new("."), 12, 3).unwrap();
        assert_eq!(s.classes(), 3);
        let (imgs, ys) = s.labelled_batch(4, 1, 7);
        assert_eq!((imgs, ys.len()), (s.batch(4, 1, 7), 4));
        assert_eq!(s.shape().values(), 12 * 12 * 3);
        assert_eq!(s.batch(4, 1, 7), s.batch(4, 1, 7));
        assert_ne!(s.batch(4, 1, 7), s.batch(4, 1, 8));
    }
}
