//! DDPM (Ho et al. 2020), for comparison with flow matching (PLAN.md,
//! Terms, and Order of work, stage 2), and DDIM's faster sampler for the
//! same trained model (Song et al. 2021).
//!
//! DDPM counts its own way: step i = 0 is data and i = N is (almost) pure
//! noise, the opposite of flow matching's t. Noise is added on a fixed
//! schedule β₁ … β_N (small variances, growing linearly), and ᾱ_i, the
//! product of (1 − β) up to step i, says how much of the data survives:
//!
//!   x_i = √ᾱ_i · x₀ + √(1 − ᾱ_i) · ε,   ε ~ N(0, 1)
//!
//! The model learns to predict ε from x_i and i. Generation starts from
//! noise at i = N and walks back to i = 0:
//!
//! - **ancestral** (DDPM's own): every one of the N steps removes a little of
//!   the predicted noise and adds a little fresh noise back;
//! - **DDIM**: a subsequence of the steps, say 50, each jumping to the next
//!   by estimating the clean point and re-noising it to the next level, with
//!   no fresh noise. The same trained model serves both.

use crate::tensor::Mat;

/// DDPM's usual number of steps and schedule (Ho et al. 2020).
pub const STEPS: usize = 1000;
const BETA_START: f64 = 1e-4;
const BETA_END: f64 = 0.02;

/// The noise schedule: β_i, α_i = 1 − β_i and ᾱ_i for i = 1 … N (index 0
/// holds i = 0: β = 0, ᾱ = 1, the clean data).
pub struct Schedule {
    pub beta: Vec<f64>,
    pub alpha_bar: Vec<f64>,
}

impl Schedule {
    pub fn linear(n: usize) -> Schedule {
        let mut beta = vec![0.0];
        let mut alpha_bar = vec![1.0];
        for i in 1..=n {
            let b = BETA_START + (BETA_END - BETA_START) * (i - 1) as f64 / (n - 1) as f64;
            beta.push(b);
            alpha_bar.push(alpha_bar[i - 1] * (1.0 - b));
        }
        Schedule { beta, alpha_bar }
    }

    pub fn steps(&self) -> usize {
        self.beta.len() - 1
    }

    /// The network's time input for step i: i/N, so 0 is data and 1 noise.
    pub fn time(&self, i: usize) -> f32 {
        i as f32 / self.steps() as f32
    }

    /// x_i from clean points x₀ and noise ε, each row at its own step.
    pub fn noisy(&self, x0: &Mat<f32>, eps: &Mat<f32>, steps: &[usize]) -> Mat<f32> {
        let mut out = Mat::zeros(x0.rows, x0.cols);
        for (r, &i) in steps.iter().enumerate() {
            let (a, b) = (
                self.alpha_bar[i].sqrt() as f32,
                (1.0 - self.alpha_bar[i]).sqrt() as f32,
            );
            for ((o, &x), &e) in out.row_mut(r).iter_mut().zip(x0.row(r)).zip(eps.row(r)) {
                *o = a * x + b * e;
            }
        }
        out
    }

    /// DDPM's ancestral sampler: all N steps, from noise x at i = N.
    /// `eps_model(x, i)` predicts the noise in points x at step i; `fresh`
    /// supplies the new noise added at each step.
    pub fn ancestral(
        &self,
        mut x: Mat<f32>,
        mut eps_model: impl FnMut(&Mat<f32>, usize) -> Mat<f32>,
        mut fresh: impl FnMut(usize, usize) -> Mat<f32>,
    ) -> Mat<f32> {
        for i in (1..=self.steps()).rev() {
            let e = eps_model(&x, i);
            let (b, a) = (self.beta[i], 1.0 - self.beta[i]);
            let coef = (b / (1.0 - self.alpha_bar[i]).sqrt()) as f32;
            let scale = (1.0 / a.sqrt()) as f32;
            x = x.zip_map(&e, |xv, ev| scale * (xv - coef * ev));
            if i > 1 {
                let sigma = b.sqrt() as f32;
                let z = fresh(x.rows, x.cols);
                x = x.zip_map(&z, |xv, zv| xv + sigma * zv);
            }
        }
        x
    }

    /// DDIM with `steps` evenly spaced steps (η = 0: no fresh noise).
    pub fn ddim(
        &self,
        mut x: Mat<f32>,
        steps: usize,
        mut eps_model: impl FnMut(&Mat<f32>, usize) -> Mat<f32>,
    ) -> Mat<f32> {
        let n = self.steps();
        // The steps visited, from N down; the last jump goes to i = 0.
        let mut seq: Vec<usize> = (0..steps).map(|k| n - k * n / steps).collect();
        seq.push(0);
        for w in seq.windows(2) {
            let (i, prev) = (w[0], w[1]);
            let e = eps_model(&x, i);
            let (ab, abp) = (self.alpha_bar[i], self.alpha_bar[prev]);
            let (sa, sb) = (ab.sqrt() as f32, (1.0 - ab).sqrt() as f32);
            let (spa, spb) = (abp.sqrt() as f32, (1.0 - abp).sqrt() as f32);
            x = x.zip_map(&e, |xv, ev| {
                let x0 = (xv - sb * ev) / sa; // the estimated clean point
                spa * x0 + spb * ev // re-noised to the next level
            });
        }
        x
    }
}

/// The ideal noise prediction when the data is Gaussian, each coordinate
/// N(μ, s²): x_i is Gaussian with mean √ᾱ·μ and variance ᾱs² + 1 − ᾱ, and
/// the best estimate of ε from it is
///
///   ε̂ = √(1 − ᾱ) · (x − √ᾱ·μ) / (ᾱs² + 1 − ᾱ)
pub fn gaussian_eps(
    sched: &Schedule,
    mean: &[f64],
    std: &[f64],
    x: &Mat<f32>,
    i: usize,
) -> Mat<f32> {
    let ab = sched.alpha_bar[i];
    let mut out = Mat::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        for c in 0..x.cols {
            let var = ab * std[c] * std[c] + 1.0 - ab;
            let v = (1.0 - ab).sqrt() * (x.row(r)[c] as f64 - ab.sqrt() * mean[c]) / var;
            out.row_mut(r)[c] = v as f32;
        }
    }
    out
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // coordinate loops read more plainly
mod tests {
    use super::*;
    use crate::data2d::{column_stats, noise, sample, GAUSSIAN_MEAN, GAUSSIAN_STD};
    use crate::rng::Rng;

    #[test]
    fn schedule_runs_from_data_to_noise() {
        let s = Schedule::linear(STEPS);
        assert_eq!(s.steps(), STEPS);
        assert_eq!(s.alpha_bar[0], 1.0);
        assert!((s.beta[1] - 1e-4).abs() < 1e-12 && (s.beta[STEPS] - 0.02).abs() < 1e-12);
        assert!(s.alpha_bar[STEPS] < 1e-4, "{}", s.alpha_bar[STEPS]);
        assert!(s.alpha_bar.windows(2).all(|w| w[1] < w[0]));
    }

    /// The forward process: noised Gaussian data has exactly the mean and
    /// spread the schedule implies, √ᾱ·μ and √(ᾱs² + 1 − ᾱ).
    #[test]
    fn noised_data_has_the_stated_statistics() {
        let s = Schedule::linear(STEPS);
        let mut rng = Rng::new(1);
        let n = 100_000;
        let x0 = sample("gaussian", n, &mut rng).unwrap();
        let e = noise(n, 2, &mut rng);
        for i in [1, 100, 500, 1000] {
            let st = column_stats(&s.noisy(&x0, &e, &vec![i; n]));
            let ab = s.alpha_bar[i];
            for c in 0..2 {
                let (m, sd) = (GAUSSIAN_MEAN[c], GAUSSIAN_STD[c]);
                assert!((st[c].0 - ab.sqrt() * m).abs() < 0.02, "{i} {st:?}");
                assert!(
                    (st[c].1 - (ab * sd * sd + 1.0 - ab).sqrt()).abs() < 0.02,
                    "{i} {st:?}"
                );
            }
        }
    }

    fn exact(s: &Schedule) -> impl FnMut(&Mat<f32>, usize) -> Mat<f32> + '_ {
        move |x, i| gaussian_eps(s, &GAUSSIAN_MEAN, &GAUSSIAN_STD, x, i)
    }

    /// Both samplers, given the exact noise prediction, must carry noise onto
    /// the Gaussian: tested before any network is trained, as for flow.
    #[test]
    fn exact_predictions_carry_noise_onto_the_gaussian() {
        let s = Schedule::linear(STEPS);
        let mut rng = Rng::new(2);
        let start = noise(10_000, 2, &mut rng);
        let a = s.ancestral(start.clone(), exact(&s), |r, c| noise(r, c, &mut rng));
        // 250 steps: with fewer, DDIM's samples come out measurably too
        // concentrated (the next test).
        let d = s.ddim(start, 250, exact(&s));
        for (name, out) in [("ancestral", a), ("ddim", d)] {
            let st = column_stats(&out);
            for c in 0..2 {
                assert!((st[c].0 - GAUSSIAN_MEAN[c]).abs() < 0.03, "{name} {st:?}");
                assert!((st[c].1 - GAUSSIAN_STD[c]).abs() < 0.03, "{name} {st:?}");
            }
        }
    }

    /// With few steps, DDIM's samples are too concentrated: each jump
    /// assumes the noise estimate holds all the way to the next level. The
    /// error shrinks as the steps increase.
    #[test]
    fn ddim_error_shrinks_with_more_steps() {
        let s = Schedule::linear(STEPS);
        let start = noise(5_000, 2, &mut Rng::new(3));
        let spread = |k| column_stats(&s.ddim(start.clone(), k, exact(&s)))[0].1;
        let (a, b, c) = (spread(10), spread(50), spread(250));
        assert!(a < b && b < c && c < GAUSSIAN_STD[0] + 0.02, "{a} {b} {c}");
        assert!(GAUSSIAN_STD[0] - a > 0.1, "10 steps: {a}");
    }

    #[test]
    fn ddim_visits_evenly_spaced_steps_down_to_zero() {
        let s = Schedule::linear(STEPS);
        let mut seen = Vec::new();
        s.ddim(Mat::zeros(1, 2), 4, |x, i| {
            seen.push(i);
            Mat::zeros(x.rows, x.cols)
        });
        assert_eq!(seen, [1000, 750, 500, 250]);
    }
}
