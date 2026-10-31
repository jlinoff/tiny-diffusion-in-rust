//! Flow matching (PLAN.md, The flows): the straight-line mixing of noise
//! and data that training learns from, the Euler sampler that generation
//! runs, and the one case where the ideal velocity field is known exactly.
//!
//! Convention: t = 0 is noise, t = 1 is data.

use crate::tensor::Mat;

/// x_t = (1 − t)·x₀ + t·x₁, row by row, each row with its own t: a point
/// on the straight line from noise x₀ to data x₁.
pub fn interpolate(x0: &Mat<f32>, x1: &Mat<f32>, t: &[f32]) -> Mat<f32> {
    assert_eq!((x0.rows, x0.cols), (x1.rows, x1.cols), "shapes");
    assert_eq!(t.len(), x0.rows, "one t per row");
    let mut out = Mat::zeros(x0.rows, x0.cols);
    for (r, &tr) in t.iter().enumerate() {
        for ((o, &a), &b) in out.row_mut(r).iter_mut().zip(x0.row(r)).zip(x1.row(r)) {
            *o = (1.0 - tr) * a + tr * b;
        }
    }
    out
}

/// The training target: x₁ − x₀, the direction (and length) of the line,
/// the same at every t along it.
pub fn target(x0: &Mat<f32>, x1: &Mat<f32>) -> Mat<f32> {
    x1.zip_map(x0, |b, a| b - a)
}

/// Generate by Euler steps: from x (noise, at t = 0) to t = 1 in `steps`
/// equal steps, x ← x + v(x, t)·Δt. `velocity` is given the current points
/// and t. With `record`, also returns the points after every step (the
/// paths, for plotting); the first entry is the starting noise.
pub fn euler(
    mut x: Mat<f32>,
    steps: usize,
    mut velocity: impl FnMut(&Mat<f32>, f32) -> Mat<f32>,
    record: bool,
) -> (Mat<f32>, Vec<Mat<f32>>) {
    assert!(steps > 0, "at least one step");
    let dt = 1.0 / steps as f32;
    let mut path = if record { vec![x.clone()] } else { Vec::new() };
    for k in 0..steps {
        let t = k as f32 * dt;
        let v = velocity(&x, t);
        x = x.zip_map(&v, |a, b| a + b * dt);
        if record {
            path.push(x.clone());
        }
    }
    (x, path)
}

/// The ideal velocity field when the data is Gaussian, each coordinate
/// independently N(μ, s²), and the noise N(0, 1).
///
/// For one coordinate, x_t = (1 − t)·x₀ + t·x₁ is Gaussian with mean tμ and
/// variance (1 − t)² + t²s², and the target x₁ − x₀ has covariance
/// t·s² − (1 − t) with it. The best prediction of the target from x_t, the
/// average over every line through that point, is therefore linear:
///
///   v(x, t) = μ + c(t)·(x − tμ),   c(t) = (t·s² − (1 − t)) / ((1 − t)² + t²s²)
///
/// Following it from t = 0 carries N(0, 1) exactly onto N(μ, s²) at t = 1.
pub struct GaussianTruth {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl GaussianTruth {
    pub fn velocity(&self, x: &Mat<f32>, t: f32) -> Mat<f32> {
        assert_eq!(x.cols, self.mean.len(), "dimensions");
        let t = t as f64;
        let mut v = Mat::zeros(x.rows, x.cols);
        for r in 0..x.rows {
            for c in 0..x.cols {
                let (mu, s2) = (self.mean[c], self.std[c] * self.std[c]);
                let k = (t * s2 - (1.0 - t)) / ((1.0 - t).powi(2) + t * t * s2);
                v.row_mut(r)[c] = (mu + k * (x.row(r)[c] as f64 - t * mu)) as f32;
            }
        }
        v
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // coordinate loops read more plainly
mod tests {
    use super::*;
    use crate::data2d::{column_stats, noise, sample, GAUSSIAN_MEAN, GAUSSIAN_STD};
    use crate::rng::Rng;

    fn truth() -> GaussianTruth {
        GaussianTruth {
            mean: GAUSSIAN_MEAN.to_vec(),
            std: GAUSSIAN_STD.to_vec(),
        }
    }

    #[test]
    fn the_line_runs_from_noise_to_data() {
        let mut rng = Rng::new(1);
        let (x0, x1) = (noise(5, 2, &mut rng), sample("moons", 5, &mut rng).unwrap());
        assert_eq!(interpolate(&x0, &x1, &[0.0; 5]), x0);
        assert_eq!(interpolate(&x0, &x1, &[1.0; 5]), x1);
        // From any point on the line, the remaining (1 − t) of the target
        // reaches the data: the target is the same all along the line.
        let t = [0.1, 0.3, 0.5, 0.7, 0.9];
        let (xt, v) = (interpolate(&x0, &x1, &t), target(&x0, &x1));
        for r in 0..5 {
            for c in 0..2 {
                let end = xt.row(r)[c] + (1.0 - t[r]) * v.row(r)[c];
                assert!((end - x1.row(r)[c]).abs() < 1e-5);
            }
        }
    }

    /// The test that needs no network: the exact field must carry noise
    /// onto the Gaussian, so the sampler is checked before anything is
    /// trained.
    #[test]
    fn the_exact_field_carries_noise_onto_the_gaussian() {
        let x0 = noise(20_000, 2, &mut Rng::new(2));
        let (x1, _) = euler(x0, 200, |x, t| truth().velocity(x, t), false);
        let s = column_stats(&x1);
        for c in 0..2 {
            assert!((s[c].0 - GAUSSIAN_MEAN[c]).abs() < 0.02, "{s:?}");
            assert!((s[c].1 - GAUSSIAN_STD[c]).abs() < 0.02, "{s:?}");
        }
    }

    /// At t = 0 the point is pure noise and says nothing about which data
    /// point its line leads to, so the ideal velocity is the data's mean
    /// minus the point, and one Euler step lands every sample exactly on the
    /// mean. (The same holds for any data: stage 2's prediction for
    /// one-step generation.)
    #[test]
    fn one_step_lands_every_sample_on_the_mean() {
        let x0 = noise(100, 2, &mut Rng::new(3));
        let (x1, _) = euler(x0, 1, |x, t| truth().velocity(x, t), false);
        for r in 0..x1.rows {
            for c in 0..2 {
                assert!((x1.row(r)[c] as f64 - GAUSSIAN_MEAN[c]).abs() < 1e-5);
            }
        }
    }

    /// The paths of the exact field curve (c(t) changes with t), so a
    /// straight Euler step overshoots inwards: with few steps the samples
    /// are too concentrated, and the error shrinks as the steps increase.
    #[test]
    fn euler_error_shrinks_with_more_steps() {
        let start = noise(5_000, 2, &mut Rng::new(5));
        let spread = |k| {
            let (x, _) = euler(start.clone(), k, |x, t| truth().velocity(x, t), false);
            column_stats(&x)[0].1
        };
        let (a, b, c) = (spread(2), spread(10), spread(100));
        assert!(a < b && b < c, "{a} {b} {c}");
        assert!((c - GAUSSIAN_STD[0]).abs() < 0.02, "{c}");
    }

    #[test]
    fn euler_records_the_path() {
        let x0 = noise(3, 2, &mut Rng::new(4));
        let (end, path) = euler(x0.clone(), 4, |x, _| x.map(|_| 1.0), true);
        assert_eq!(path.len(), 5);
        assert_eq!(path[0], x0);
        assert_eq!(path[4], end);
        // A constant velocity of 1 for a total time of 1 moves every point by 1.
        for (a, b) in end.data.iter().zip(&x0.data) {
            assert!((a - b - 1.0).abs() < 1e-6);
        }
    }
}
