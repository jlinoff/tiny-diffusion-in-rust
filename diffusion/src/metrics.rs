//! How close generated points are to the data: a single number, so step
//! counts, samplers and objectives can be compared.
//!
//! The **energy distance** (Székely and Rizzo 2013) between two sets of
//! points X and Y is
//!
//!   2·mean|x − y| − mean|x − x′| − mean|y − y′|
//!
//! over all pairs (x and x′ different points of X, and so on). It's zero
//! only when the two distributions are the same, and grows as they differ,
//! in location, in spread or in shape; unlike comparing means and standard
//! deviations, it notices a missing blob or a filled-in gap.

use crate::tensor::Mat;

fn dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn mean_between(x: &Mat<f32>, y: &Mat<f32>) -> f64 {
    let mut s = 0.0;
    for i in 0..x.rows {
        for j in 0..y.rows {
            s += dist(x.row(i), y.row(j));
        }
    }
    s / (x.rows * y.rows) as f64
}

fn mean_within(x: &Mat<f32>) -> f64 {
    let mut s = 0.0;
    for i in 0..x.rows {
        for j in i + 1..x.rows {
            s += dist(x.row(i), x.row(j));
        }
    }
    s / (x.rows * (x.rows - 1) / 2) as f64
}

/// The energy distance between the point sets `x` and `y` (rows are points).
pub fn energy_distance(x: &Mat<f32>, y: &Mat<f32>) -> f64 {
    2.0 * mean_between(x, y) - mean_within(x) - mean_within(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data2d::sample;
    use crate::rng::Rng;

    #[test]
    fn small_for_the_same_distribution_and_grows_with_difference() {
        let mut rng = Rng::new(1);
        let a = sample("moons", 1000, &mut rng).unwrap();
        let b = sample("moons", 1000, &mut rng).unwrap();
        let same = energy_distance(&a, &b);
        let shifted = energy_distance(&a, &b.map(|v| v + 0.3));
        let far = energy_distance(&a, &b.map(|v| v + 1.0));
        assert!(same.abs() < 0.01, "{same}");
        assert!(
            shifted > 10.0 * same.abs() && far > shifted,
            "{same} {shifted} {far}"
        );
    }

    #[test]
    fn notices_a_missing_blob_that_means_and_spreads_could_miss() {
        let mut rng = Rng::new(2);
        let full = sample("rings", 1600, &mut rng).unwrap();
        // The same ring with one blob (around angle 0) moved to the centre.
        let mut partial = sample("rings", 1600, &mut rng).unwrap();
        for r in 0..partial.rows {
            let row = partial.row_mut(r);
            if row[0] > 1.5 && row[1].abs() < 0.6 {
                row[0] -= 2.0;
            }
        }
        let fresh = sample("rings", 1600, &mut rng).unwrap();
        assert!(energy_distance(&full, &partial) > 3.0 * energy_distance(&full, &fresh));
    }

    #[test]
    fn by_hand() {
        // One point each, at distance 2: 2·2 − 0 − 0, with the within-set
        // means taken over two points per set here.
        let x = Mat::from_vec(2, 1, vec![0.0, 0.0]);
        let y = Mat::from_vec(2, 1, vec![2.0, 2.0]);
        assert!((energy_distance(&x, &y) - 4.0).abs() < 1e-12);
    }
}
