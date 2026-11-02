//! Looking inside the trained models (PLAN.md, stage 5): the model's guess
//! of the finished image along a generation, how straight the paths are,
//! the class means guidance pushes between, and how far attention reaches.

use crate::tensor::Mat;

/// The model's guess of the finished image at time t: one Euler step of
/// the whole remaining way, x̂₁ = x + (1 − t)·v.
pub fn guess(x: &Mat<f32>, v: &Mat<f32>, t: f32) -> Mat<f32> {
    assert_eq!((x.rows, x.cols), (v.rows, v.cols));
    let data = x
        .data
        .iter()
        .zip(&v.data)
        .map(|(a, b)| a + (1.0 - t) * b)
        .collect();
    Mat::from_vec(x.rows, x.cols, data)
}

fn norm(a: &[f32]) -> f64 {
    a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
}

/// The Euclidean distance between two images.
pub fn dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// The average length of each row of `m` (the square root of the sum of
/// its squared values).
pub fn mean_norm(m: &Mat<f32>) -> f64 {
    (0..m.rows).map(|r| norm(m.row(r))).sum::<f64>() / m.rows.max(1) as f64
}

/// The distance from `a` to `target`, relative to the target's length.
pub fn relative_error(a: &[f32], target: &[f32]) -> f64 {
    dist(a, target) / norm(target).max(1e-12)
}

/// For each path (row r of every step's matrix, `path[0]` the noise, the
/// last the finished image): the straight-line distance from start to end
/// divided by the length of the path taken. 1 for a straight path.
pub fn straightness(path: &[Mat<f32>]) -> Vec<f64> {
    let rows = path[0].rows;
    (0..rows)
        .map(|r| {
            let chord = dist(path[0].row(r), path[path.len() - 1].row(r));
            let length: f64 = path
                .windows(2)
                .map(|w| dist(w[0].row(r), w[1].row(r)))
                .sum();
            if length == 0.0 {
                1.0
            } else {
                chord / length
            }
        })
        .collect()
}

/// Where along the path the bending is: the path's excess length over its
/// chord, for the first half of the steps and for the second, as the
/// first half's share of the total (0.5 if bent evenly; 0.5 if straight).
pub fn first_half_bend(path: &[Mat<f32>], r: usize) -> f64 {
    let mid = (path.len() - 1) / 2;
    let excess = |a: usize, b: usize| {
        let chord = dist(path[a].row(r), path[b].row(r));
        let length: f64 = (a..b)
            .map(|k| dist(path[k].row(r), path[k + 1].row(r)))
            .sum();
        length - chord
    };
    let (first, second) = (excess(0, mid), excess(mid, path.len() - 1));
    if first + second <= 1e-12 {
        0.5
    } else {
        first / (first + second)
    }
}

/// The mean image of each class (labels 0 to `classes` − 1), and of all.
pub fn class_means(pixels: &Mat<f32>, labels: &[u8], classes: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let cols = pixels.cols;
    let mut sums = vec![vec![0.0f64; cols]; classes];
    let mut counts = vec![0usize; classes];
    for (r, &l) in labels.iter().enumerate() {
        counts[l as usize] += 1;
        for (s, &x) in sums[l as usize].iter_mut().zip(pixels.row(r)) {
            *s += x as f64;
        }
    }
    let n = labels.len().max(1) as f64;
    let mut all = vec![0.0f64; cols];
    for s in &sums {
        for (a, &x) in all.iter_mut().zip(s) {
            *a += x;
        }
    }
    let means = sums
        .iter()
        .zip(&counts)
        .map(|(s, &k)| s.iter().map(|&x| (x / k.max(1) as f64) as f32).collect())
        .collect();
    (means, all.iter().map(|&x| (x / n) as f32).collect())
}

/// The commitment curve: for each recorded moment, the fraction of
/// generations whose guess the classifier already calls their final digit.
pub fn commitment(per_moment: &[Vec<usize>], last: &[usize]) -> Vec<f64> {
    per_moment
        .iter()
        .map(|g| {
            g.iter().zip(last).filter(|(a, b)| a == b).count() as f64 / last.len().max(1) as f64
        })
        .collect()
}

/// The attention distance of one head's weights over a side × side grid of
/// patches: the average, over patches, of the attention-weighted distance
/// (in patches) to the patches attended to.
pub fn attention_distance(w: &Mat<f32>, side: usize) -> f64 {
    let n = side * side;
    assert_eq!((w.rows, w.cols), (n, n));
    let pos = |i: usize| ((i / side) as f64, (i % side) as f64);
    let mut total = 0.0;
    for i in 0..n {
        let (yi, xi) = pos(i);
        for j in 0..n {
            let (yj, xj) = pos(j);
            total += w.data[i * n + j] as f64 * ((yi - yj).powi(2) + (xi - xj).powi(2)).sqrt();
        }
    }
    total / n as f64
}

/// Velocities shown as an image: 0 grey, positive towards white, negative
/// towards black, `scale` per unit of velocity (values clipped to −1..1).
pub fn velocity_image(v: &Mat<f32>, scale: f32) -> Mat<f32> {
    Mat::from_vec(
        v.rows,
        v.cols,
        v.data
            .iter()
            .map(|&x| (x * scale).clamp(-1.0, 1.0))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::GaussianTruth;

    /// With stage 2's exact field for Gaussian data, the guess is the data's
    /// mean at t = 0, and at every t the conditional mean E[x₁ | x_t].
    #[test]
    fn guess_against_the_gaussian_closed_form() {
        let truth = GaussianTruth {
            mean: vec![1.5, -1.0],
            std: vec![0.5, 0.8],
        };
        let x = Mat::from_vec(3, 2, vec![0.3, -1.2, 2.0, 0.5, -0.7, 0.0]);
        let g = guess(&x, &truth.velocity(&x, 0.0), 0.0);
        for r in 0..3 {
            assert!((g.row(r)[0] - 1.5).abs() < 1e-6 && (g.row(r)[1] + 1.0).abs() < 1e-6);
        }
        for t in [0.2f32, 0.5, 0.9] {
            let g = guess(&x, &truth.velocity(&x, t), t);
            let td = t as f64;
            for r in 0..3 {
                for c in 0..2 {
                    let (mu, s2) = (truth.mean[c], truth.std[c].powi(2));
                    let xt = x.row(r)[c] as f64;
                    let want = mu + td * s2 / ((1.0 - td).powi(2) + td * td * s2) * (xt - td * mu);
                    assert!((g.row(r)[c] as f64 - want).abs() < 1e-5, "t {t}");
                }
            }
        }
    }

    #[test]
    fn straight_and_bent_paths() {
        let line: Vec<Mat<f32>> = (0..5)
            .map(|k| Mat::from_vec(1, 2, vec![k as f32, 2.0 * k as f32]))
            .collect();
        assert!((straightness(&line)[0] - 1.0).abs() < 1e-12);
        assert!((first_half_bend(&line, 0) - 0.5).abs() < 1e-12);
        // One step along x, then three up along y: chord √10, length 4; the
        // corner is at step 1 of 4, so all the excess is in the first half.
        let corner: Vec<Mat<f32>> = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
            .iter()
            .map(|&(a, b)| Mat::from_vec(1, 2, vec![a, b]))
            .collect();
        let want = (1.0f64 + 9.0).sqrt() / 4.0;
        assert!((straightness(&corner)[0] - want).abs() < 1e-9);
        assert!(first_half_bend(&corner, 0) > 0.99);
    }

    #[test]
    fn class_means_by_hand() {
        let px = Mat::from_vec(4, 2, vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let (m, all) = class_means(&px, &[0, 1, 0, 1], 2);
        assert_eq!(m, vec![vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(all, vec![4.0, 5.0]);
    }

    #[test]
    fn commitment_by_hand() {
        let c = commitment(
            &[vec![3, 3, 3, 3], vec![1, 2, 3, 3], vec![1, 2, 5, 4]],
            &[1, 2, 5, 4],
        );
        assert_eq!(c, vec![0.0, 0.5, 1.0]);
        assert_eq!(
            commitment(&[vec![7, 8], vec![7, 8]], &[7, 8]),
            vec![1.0, 1.0]
        );
    }

    /// Even attention gives the closed-form mean distance on the grid
    /// (3.609 for 7 × 7, 2.008 for 4 × 4, PLAN.md); attention to itself, 0.
    #[test]
    fn attention_distance_baselines() {
        for (side, want) in [(7usize, 3.609), (4, 2.008)] {
            let n = side * side;
            let even = Mat::from_vec(n, n, vec![1.0 / n as f32; n * n]);
            assert!(
                (attention_distance(&even, side) - want).abs() < 1e-3,
                "{side}"
            );
            let mut own = Mat::zeros(n, n);
            for i in 0..n {
                own.data[i * n + i] = 1.0;
            }
            assert_eq!(attention_distance(&own, side), 0.0);
        }
    }

    #[test]
    fn helpers() {
        assert!((relative_error(&[3.0, 4.0], &[3.0, 4.0])).abs() < 1e-12);
        assert!((relative_error(&[0.0, 0.0], &[3.0, 4.0]) - 1.0).abs() < 1e-12);
        let m = Mat::from_vec(2, 2, vec![3.0, 4.0, 0.0, 0.0]);
        assert!((mean_norm(&m) - 2.5).abs() < 1e-12);
        let v = velocity_image(&Mat::from_vec(1, 3, vec![-3.0, 0.0, 0.2]), 0.5);
        assert_eq!(v.data, vec![-1.0, 0.0, 0.1]);
    }
}
