//! Training the stage-2 MLP on points in a plane, by flow matching or by
//! DDPM (PLAN.md, The flows: Training).
//!
//! Every training step draws a fresh batch: data points, noise, and a time
//! per point, all from a random stream fixed by the seed and the step
//! number, so a run is reproducible and doesn't depend on how many steps
//! came before in this process.

use crate::data2d::{noise, sample};
use crate::ddpm::Schedule;
use crate::flow::{interpolate, target};
use crate::mlp::{input, Mlp};
use crate::optim::Adam;
use crate::rng::Rng;
use crate::tape::Tape;
use crate::tensor::Mat;

/// What the network learns to predict.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Objective {
    /// The velocity x₁ − x₀ at a point on the line (flow matching).
    Flow,
    /// The noise ε in a noised point (DDPM).
    Ddpm,
}

impl Objective {
    pub fn parse(s: &str) -> Result<Objective, String> {
        match s {
            "flow" => Ok(Objective::Flow),
            "ddpm" => Ok(Objective::Ddpm),
            _ => Err(format!("unknown objective {s:?}: flow or ddpm")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Objective::Flow => "flow",
            Objective::Ddpm => "ddpm",
        }
    }
}

pub struct TrainConfig {
    pub data: String,
    pub objective: Objective,
    pub steps: u64,
    pub batch: usize,
    pub learning_rate: f32,
    pub warmup: u64,
    pub seed: u64,
}

impl TrainConfig {
    /// The learning rate at a step: a linear warm-up from 0, then a cosine
    /// decay to a tenth of the peak, as in tiny-llm-in-rust.
    pub fn lr(&self, step: u64) -> f32 {
        let peak = self.learning_rate;
        if step <= self.warmup {
            return peak * step as f32 / self.warmup.max(1) as f32;
        }
        let span = (self.steps - self.warmup).max(1) as f32;
        let p = (step - self.warmup) as f32 / span;
        let cos = 0.5 * (1.0 + (std::f32::consts::PI * p.min(1.0)).cos());
        peak * (0.1 + 0.9 * cos)
    }
}

/// One batch: the network's input and its target.
pub fn batch(c: &TrainConfig, freqs: usize, sched: &Schedule, step: u64) -> (Mat<f32>, Mat<f32>) {
    let mut rng = Rng::stream(c.seed, step);
    let x1 = sample(&c.data, c.batch, &mut rng).expect("data set checked before training");
    let eps = noise(c.batch, x1.cols, &mut rng);
    match c.objective {
        Objective::Flow => {
            let t: Vec<f32> = (0..c.batch).map(|_| rng.unit() as f32).collect();
            let xt = interpolate(&eps, &x1, &t);
            (input(&xt, &t, freqs), target(&eps, &x1))
        }
        Objective::Ddpm => {
            let steps: Vec<usize> = (0..c.batch).map(|_| 1 + rng.below(sched.steps())).collect();
            let xi = sched.noisy(&x1, &eps, &steps);
            let t: Vec<f32> = steps.iter().map(|&i| sched.time(i)).collect();
            (input(&xi, &t, freqs), eps)
        }
    }
}

/// Train `model` for `c.steps` steps; `report(step, loss)` is called after
/// every step. Returns the loss of every step.
pub fn train(model: &mut Mlp<f32>, c: &TrainConfig, mut report: impl FnMut(u64, f32)) -> Vec<f32> {
    let sched = Schedule::linear(crate::ddpm::STEPS);
    let mut adam = Adam::new(&model.params);
    let mut losses = Vec::with_capacity(c.steps as usize);
    for step in 1..=c.steps {
        let (inp, tgt) = batch(c, model.cfg.freqs, &sched, step);
        let mut tape = Tape::new();
        let (out, vars) = model.forward(&mut tape, inp);
        let loss = tape.mse(out, &tgt);
        let value = tape.value(loss).data[0];
        tape.backward(loss);
        let grads: Vec<Mat<f32>> = vars.iter().map(|&v| tape.take_grad(v)).collect();
        adam.update(&mut model.params, &grads, c.lr(step));
        losses.push(value);
        report(step, value);
    }
    losses
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data2d::{GAUSSIAN_MEAN, GAUSSIAN_STD};
    use crate::flow::GaussianTruth;
    use crate::mlp::MlpConfig;

    fn small() -> MlpConfig {
        MlpConfig {
            dim: 2,
            width: 64,
            depth: 3,
            freqs: 4,
        }
    }

    fn config(data: &str, objective: Objective, steps: u64) -> TrainConfig {
        TrainConfig {
            data: data.into(),
            objective,
            steps,
            batch: 256,
            learning_rate: 3e-3,
            warmup: 50,
            seed: 1,
        }
    }

    #[test]
    fn schedule_warms_up_then_decays_to_a_tenth() {
        let c = config("moons", Objective::Flow, 1050);
        assert_eq!(c.lr(0), 0.0);
        assert!((c.lr(25) - 1.5e-3).abs() < 1e-9);
        assert!((c.lr(50) - 3e-3).abs() < 1e-9);
        assert!((c.lr(1050) - 3e-4).abs() < 1e-9);
    }

    /// A golden test: a fixed seed and model must reproduce recorded losses
    /// (PLAN.md, Design for test). If a change to training is meant, rerun
    /// with `cargo test golden -- --nocapture` and update the values.
    #[test]
    fn golden_losses() {
        const GOLDEN: [(&str, Objective, [f32; 3]); 2] = [
            ("moons", Objective::Flow, [2.387945, 2.127066, 1.860097]),
            ("moons", Objective::Ddpm, [0.941759, 0.655641, 0.338823]),
        ];
        for (data, obj, want) in GOLDEN {
            let mut m = Mlp::new(small(), 1);
            let l = train(&mut m, &config(data, obj, 60), |_, _| {});
            let got = [l[0], l[29], l[59]];
            println!("{data} {obj:?}: {got:?}");
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() < 2e-3,
                    "{data} {obj:?}: {got:?}, golden {want:?}"
                );
            }
        }
    }

    /// Trained on the Gaussian, the network must approach the exact velocity
    /// field: the closed form checks the learning, not just the sampler.
    #[test]
    fn flow_learns_the_exact_field_for_gaussian_data() {
        let mut m = Mlp::new(small(), 2);
        train(
            &mut m,
            &config("gaussian", Objective::Flow, 1500),
            |_, _| {},
        );
        let truth = GaussianTruth {
            mean: GAUSSIAN_MEAN.to_vec(),
            std: GAUSSIAN_STD.to_vec(),
        };
        let mut rng = Rng::new(9);
        let mut err = 0.0;
        let mut size = 0.0;
        let ts = [0.0, 0.25, 0.5, 0.75, 0.95];
        for &t in &ts {
            // Points where x_t actually lies at time t.
            let x1 = sample("gaussian", 500, &mut rng).unwrap();
            let x0 = noise(500, 2, &mut rng);
            let xt = interpolate(&x0, &x1, &[t; 500]);
            let (p, v) = (m.predict(&xt, t), truth.velocity(&xt, t));
            for (a, b) in p.data.iter().zip(&v.data) {
                err += ((a - b) as f64).powi(2);
                size += (*b as f64).powi(2);
            }
        }
        let rel = (err / size).sqrt();
        assert!(rel < 0.1, "relative error {rel}");
    }
}
