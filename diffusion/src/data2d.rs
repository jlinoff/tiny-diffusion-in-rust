//! Toy data in a plane: each example is two numbers, a point (PLAN.md,
//! Order of work, stage 2). Every distribution can be drawn from without
//! limit, so training never sees the same example twice, and every one is
//! small enough to plot.
//!
//! - `moons`: two interleaved half-circles, the classic test of whether a
//!   model can produce two separate, curved pieces;
//! - `spiral`: one long curve that winds around itself;
//! - `rings`: eight small Gaussian blobs on a circle, a test of whether every
//!   blob gets its share or some are missed ("mode collapse");
//! - `gaussian`: a single Gaussian with known mean and spread, the case where
//!   the ideal velocity field has a formula (see `flow::GaussianTruth`).
//!
//! All are scaled to lie roughly within [−2.5, 2.5] on both axes, the same
//! scale as the noise they start from.

use crate::rng::Rng;
use crate::tensor::Mat;
use std::f64::consts::PI;

/// The data sets, by name.
pub const NAMES: [&str; 4] = ["moons", "spiral", "rings", "gaussian"];

/// The mean and standard deviation of `gaussian`, per coordinate.
pub const GAUSSIAN_MEAN: [f64; 2] = [1.5, -1.0];
pub const GAUSSIAN_STD: [f64; 2] = [0.5, 0.8];

/// `n` points from the named data set, as an n × 2 matrix.
pub fn sample(name: &str, n: usize, rng: &mut Rng) -> Result<Mat<f32>, String> {
    let point: fn(&mut Rng) -> [f64; 2] = match name {
        "moons" => moons,
        "spiral" => spiral,
        "rings" => rings,
        "gaussian" => gaussian,
        _ => {
            return Err(format!(
                "unknown data set {name:?}; the choices are {}",
                NAMES.join(", ")
            ))
        }
    };
    let mut data = Vec::with_capacity(2 * n);
    for _ in 0..n {
        let [x, y] = point(rng);
        data.push(x as f32);
        data.push(y as f32);
    }
    Ok(Mat::from_vec(n, 2, data))
}

/// Standard normal noise, n × cols: the starting point of generation.
pub fn noise(n: usize, cols: usize, rng: &mut Rng) -> Mat<f32> {
    Mat::from_vec(
        n,
        cols,
        (0..n * cols).map(|_| rng.normal() as f32).collect(),
    )
}

fn moons(rng: &mut Rng) -> [f64; 2] {
    let theta = PI * rng.unit();
    let (x, y) = if rng.below(2) == 0 {
        (theta.cos(), theta.sin())
    } else {
        (1.0 - theta.cos(), 0.5 - theta.sin())
    };
    // Centered, scaled to about the noise's spread, with a little jitter.
    [
        1.6 * (x - 0.5) + 0.08 * rng.normal(),
        1.6 * (y - 0.25) + 0.08 * rng.normal(),
    ]
}

fn spiral(rng: &mut Rng) -> [f64; 2] {
    // Uniform along the arc length (which grows with θ), out to 1.5 turns.
    let theta = 3.0 * PI * rng.unit().sqrt();
    let r = 2.2 * theta / (3.0 * PI);
    [
        r * theta.cos() + 0.06 * rng.normal(),
        r * theta.sin() + 0.06 * rng.normal(),
    ]
}

fn rings(rng: &mut Rng) -> [f64; 2] {
    let k = rng.below(8) as f64;
    let angle = 2.0 * PI * k / 8.0;
    [
        2.0 * angle.cos() + 0.12 * rng.normal(),
        2.0 * angle.sin() + 0.12 * rng.normal(),
    ]
}

fn gaussian(rng: &mut Rng) -> [f64; 2] {
    [
        GAUSSIAN_MEAN[0] + GAUSSIAN_STD[0] * rng.normal(),
        GAUSSIAN_MEAN[1] + GAUSSIAN_STD[1] * rng.normal(),
    ]
}

/// Each column's mean and standard deviation, in f64.
pub fn column_stats(m: &Mat<f32>) -> Vec<(f64, f64)> {
    (0..m.cols)
        .map(|c| {
            let n = m.rows as f64;
            let mean = (0..m.rows).map(|r| m.row(r)[c] as f64).sum::<f64>() / n;
            let var = (0..m.rows)
                .map(|r| (m.row(r)[c] as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            (mean, var.sqrt())
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)] // coordinate loops read more plainly
mod tests {
    use super::*;

    #[test]
    fn every_data_set_is_reproducible_and_in_range() {
        for name in NAMES {
            let a = sample(name, 2000, &mut Rng::new(1)).unwrap();
            let b = sample(name, 2000, &mut Rng::new(1)).unwrap();
            assert_eq!(a, b, "{name}");
            assert_eq!((a.rows, a.cols), (2000, 2));
            assert!(a.data.iter().all(|x| x.abs() < 4.0), "{name}");
        }
        assert!(sample("nonsense", 1, &mut Rng::new(1)).is_err());
    }

    #[test]
    fn gaussian_has_its_stated_mean_and_spread() {
        let s = column_stats(&sample("gaussian", 100_000, &mut Rng::new(2)).unwrap());
        for c in 0..2 {
            assert!((s[c].0 - GAUSSIAN_MEAN[c]).abs() < 0.01, "{s:?}");
            assert!((s[c].1 - GAUSSIAN_STD[c]).abs() < 0.01, "{s:?}");
        }
    }

    #[test]
    fn rings_fill_all_eight_blobs() {
        let m = sample("rings", 8000, &mut Rng::new(3)).unwrap();
        let mut counts = [0usize; 8];
        for r in 0..m.rows {
            let [x, y] = [m.row(r)[0] as f64, m.row(r)[1] as f64];
            let k = ((y.atan2(x) / (2.0 * PI) * 8.0).round() as i64).rem_euclid(8);
            counts[k as usize] += 1;
        }
        assert!(
            counts.iter().all(|&c| (850..1150).contains(&c)),
            "{counts:?}"
        );
    }

    #[test]
    fn noise_is_standard_normal() {
        let s = column_stats(&noise(100_000, 2, &mut Rng::new(4)));
        for (mean, std) in s {
            assert!(
                mean.abs() < 0.01 && (std - 1.0).abs() < 0.01,
                "{mean} {std}"
            );
        }
    }
}
