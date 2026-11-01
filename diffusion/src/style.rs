//! How a digit is written, apart from which digit it is (PLAN.md, Stage 4,
//! prediction 6): its slant, its position and how much ink it uses,
//! measured from the image's moments, the way a physicist finds a body's
//! centre of mass and its tilt.
//!
//! Each pixel's ink is its brightness on a 0-to-1 scale ((value + 1) / 2).
//! The centre is the ink-weighted mean position; the slant is how far the
//! strokes move right per row downwards, cov(x, y) / var(y), negated so a
//! digit leaning right (top to the right of the bottom) has positive slant.

use crate::image::Shape;
use crate::tensor::Mat;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    /// The mean ink per pixel, 0 to 1: a thick digit uses more.
    pub ink: f64,
    /// The centre of the ink, in pixels from the left and from the top.
    pub cx: f64,
    pub cy: f64,
    /// Pixels moved right per pixel upwards: positive leans right.
    pub slant: f64,
}

/// The style of one grayscale image (its first channel, if more).
pub fn style(img: &[f32], shape: Shape) -> Style {
    let (h, w, c) = (shape.height, shape.width, shape.channels);
    let (mut m, mut sx, mut sy) = (0.0, 0.0, 0.0);
    let ink = |y: usize, x: usize| ((img[(y * w + x) * c] as f64 + 1.0) / 2.0).clamp(0.0, 1.0);
    for y in 0..h {
        for x in 0..w {
            let v = ink(y, x);
            m += v;
            sx += v * x as f64;
            sy += v * y as f64;
        }
    }
    if m == 0.0 {
        return Style {
            ink: 0.0,
            cx: w as f64 / 2.0,
            cy: h as f64 / 2.0,
            slant: 0.0,
        };
    }
    let (cx, cy) = (sx / m, sy / m);
    let (mut vyy, mut vxy) = (0.0, 0.0);
    for y in 0..h {
        for x in 0..w {
            let v = ink(y, x);
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            vyy += v * dy * dy;
            vxy += v * dx * dy;
        }
    }
    Style {
        ink: m / (h * w) as f64,
        cx,
        cy,
        // y grows downwards, so a right lean has x falling as y grows.
        slant: if vyy > 0.0 { -vxy / vyy } else { 0.0 },
    }
}

pub fn styles(images: &Mat<f32>, shape: Shape) -> Vec<Style> {
    (0..images.rows)
        .map(|r| style(images.row(r), shape))
        .collect()
}

/// Pearson's correlation of two equally long series: 1 if they rise and
/// fall together, −1 if opposite, near 0 if unrelated.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa == 0.0 || sbb == 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

/// For a grid of `classes` rows × `k` columns (row-major: class c, noise
/// j at c·k + j) of one property, the mean correlation between two
/// classes' rows when the columns share their noise, and when one row is
/// shifted by a column so they don't.
pub fn same_and_different(values: &[f64], classes: usize, k: usize) -> (f64, f64) {
    assert_eq!(values.len(), classes * k);
    let row = |c: usize| &values[c * k..(c + 1) * k];
    let (mut same, mut diff, mut pairs) = (0.0, 0.0, 0);
    for a in 0..classes {
        for b in a + 1..classes {
            same += correlation(row(a), row(b));
            let shifted: Vec<f64> = (0..k).map(|j| row(b)[(j + 1) % k]).collect();
            diff += correlation(row(a), &shifted);
            pairs += 1;
        }
    }
    (same / pairs as f64, diff / pairs as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Shape = Shape {
        height: 9,
        width: 9,
        channels: 1,
    };

    /// A one-pixel line through the middle with `lean` columns of shift
    /// from bottom to top.
    fn line(lean: i32) -> Vec<f32> {
        let mut img = vec![-1.0f32; 81];
        for y in 0..9i32 {
            let x = 4 + (lean * (4 - y)) / 4;
            img[(y * 9 + x) as usize] = 1.0;
        }
        img
    }

    #[test]
    fn upright_and_leaning_lines() {
        let s = style(&line(0), S);
        assert!((s.cx - 4.0).abs() < 1e-9 && (s.cy - 4.0).abs() < 1e-9);
        assert!(s.slant.abs() < 1e-9);
        assert!((s.ink - 9.0 / 81.0).abs() < 1e-9);
        assert!(style(&line(4), S).slant > 0.9, "leaning right");
        assert!(style(&line(-4), S).slant < -0.9, "leaning left");
        let blank = style(&[-1.0; 81], S);
        assert_eq!((blank.ink, blank.slant), (0.0, 0.0));
    }

    #[test]
    fn correlations() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((correlation(&a, &a) - 1.0).abs() < 1e-12);
        assert!((correlation(&a, &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
        assert_eq!(correlation(&a, &[5.0; 4]), 0.0);
        // Two classes whose rows are equal: same 1; shifted, lower.
        let v = [1.0, 5.0, 2.0, 8.0, 1.0, 5.0, 2.0, 8.0];
        let (same, diff) = same_and_different(&v, 2, 4);
        assert!((same - 1.0).abs() < 1e-12 && diff < 0.5, "{same} {diff}");
    }
}
