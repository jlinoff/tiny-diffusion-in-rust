//! The convolutional pieces of stage 7's decoder (PLAN.md, stage 7):
//! convolution by im2col, group normalization, nearest-neighbour
//! upsampling and SiLU, on feature maps of channels × height × width.
//!
//! Forward passes only: nothing in stage 7 trains, so these work on plain
//! f32 arrays, not on the tape.

use crate::tensor::Mat;

/// A feature map: `c` channels of `h` × `w` values, channel by channel,
/// each channel row by row (PyTorch's layout for one image).
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Feature {
    pub fn new(c: usize, h: usize, w: usize, data: Vec<f32>) -> Feature {
        assert_eq!(data.len(), c * h * w, "feature map size");
        Feature { c, h, w, data }
    }

    pub fn zeros(c: usize, h: usize, w: usize) -> Feature {
        Feature::new(c, h, w, vec![0.0; c * h * w])
    }

    pub fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[(c * self.h + y) * self.w + x]
    }

    /// Positions as rows, channels as columns: (h·w) × c, the layout an
    /// attention layer works on.
    pub fn to_rows(&self) -> Mat<f32> {
        let n = self.h * self.w;
        let mut m = Mat::zeros(n, self.c);
        for ch in 0..self.c {
            for p in 0..n {
                m.data[p * self.c + ch] = self.data[ch * n + p];
            }
        }
        m
    }

    /// The reverse of `to_rows`.
    pub fn from_rows(m: &Mat<f32>, h: usize, w: usize) -> Feature {
        assert_eq!(m.rows, h * w, "one row per position");
        let n = h * w;
        let mut data = vec![0.0; m.cols * n];
        for p in 0..n {
            for ch in 0..m.cols {
                data[ch * n + p] = m.data[p * m.cols + ch];
            }
        }
        Feature::new(m.cols, h, w, data)
    }

    pub fn add_assign(&mut self, other: &Feature) {
        assert_eq!((self.c, self.h, self.w), (other.c, other.h, other.w));
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }
}

/// A convolution's weights: `out` filters of `inp` channels × k × k,
/// PyTorch's order, and a bias per filter.
pub struct Conv {
    pub out: usize,
    pub inp: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
    /// (inp·k·k) × out: the filters as columns, ready for the product.
    pub cols: Mat<f32>,
    pub bias: Vec<f32>,
}

impl Conv {
    /// From PyTorch's weight [out, inp, k, k] and bias [out].
    pub fn new(
        weight: &[f32],
        bias: &[f32],
        out: usize,
        inp: usize,
        k: usize,
        stride: usize,
        pad: usize,
    ) -> Conv {
        assert_eq!(weight.len(), out * inp * k * k, "convolution weight size");
        assert_eq!(bias.len(), out, "one bias per filter");
        let rows = Mat::from_vec(out, inp * k * k, weight.to_vec());
        Conv {
            out,
            inp,
            k,
            stride,
            pad,
            cols: rows.transpose(),
            bias: bias.to_vec(),
        }
    }

    fn out_size(&self, n: usize) -> usize {
        (n + 2 * self.pad - self.k) / self.stride + 1
    }

    /// The convolution by im2col: each output position's k × k
    /// neighbourhood, across all input channels, laid out as one row
    /// (zeros where it reaches past the edge), then one matrix product
    /// with the filters.
    pub fn forward(&self, x: &Feature) -> Feature {
        assert_eq!(x.c, self.inp, "input channels");
        let (ho, wo) = (self.out_size(x.h), self.out_size(x.w));
        let (k, s, pad) = (self.k, self.stride, self.pad as isize);
        let width = self.inp * k * k;
        let mut patches = Mat::zeros(ho * wo, width);
        for oy in 0..ho {
            for ox in 0..wo {
                let row = patches.row_mut(oy * wo + ox);
                for c in 0..self.inp {
                    for ky in 0..k {
                        let y = (oy * s + ky) as isize - pad;
                        if y < 0 || y >= x.h as isize {
                            continue;
                        }
                        for kx in 0..k {
                            let xx = (ox * s + kx) as isize - pad;
                            if xx < 0 || xx >= x.w as isize {
                                continue;
                            }
                            row[(c * k + ky) * k + kx] = x.at(c, y as usize, xx as usize);
                        }
                    }
                }
            }
        }
        let mut y = patches.matmul(&self.cols); // (ho·wo) × out
        for p in 0..ho * wo {
            for (o, b) in y.row_mut(p).iter_mut().zip(&self.bias) {
                *o += b;
            }
        }
        Feature::from_rows(&y, ho, wo)
    }
}

/// Group normalization: the channels split into `groups` equal groups,
/// each group (all its channels, all positions) normalized to mean 0 and
/// variance 1, then each channel scaled and shifted.
pub fn group_norm(x: &Feature, groups: usize, gamma: &[f32], beta: &[f32], eps: f32) -> Feature {
    assert!(x.c.is_multiple_of(groups), "channels divide into groups");
    assert_eq!(
        (gamma.len(), beta.len()),
        (x.c, x.c),
        "a scale and shift per channel"
    );
    let per = x.c / groups;
    let n = x.h * x.w;
    let mut out = x.clone();
    for g in 0..groups {
        let span = g * per * n..(g + 1) * per * n;
        let vals = &x.data[span.clone()];
        let count = vals.len() as f64;
        let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / count;
        let var = vals.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / count;
        let inv = 1.0 / (var + eps as f64).sqrt();
        for ch in g * per..(g + 1) * per {
            for p in 0..n {
                let i = ch * n + p;
                out.data[i] = ((x.data[i] as f64 - mean) * inv) as f32 * gamma[ch] + beta[ch];
            }
        }
    }
    out
}

/// Each value copied to a 2 × 2 square: twice the width and height.
pub fn upsample2(x: &Feature) -> Feature {
    let (h, w) = (2 * x.h, 2 * x.w);
    let mut out = Feature::zeros(x.c, h, w);
    for c in 0..x.c {
        for y in 0..h {
            for xx in 0..w {
                out.data[(c * h + y) * w + xx] = x.at(c, y / 2, xx / 2);
            }
        }
    }
    out
}

/// SiLU, x·sigmoid(x), in place.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn rand(n: usize, seed: u64) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| r.normal() as f32).collect()
    }

    /// The convolution written out as its definition, loop by loop.
    fn direct(
        x: &Feature,
        w: &[f32],
        b: &[f32],
        out: usize,
        k: usize,
        s: usize,
        pad: usize,
    ) -> Feature {
        let ho = (x.h + 2 * pad - k) / s + 1;
        let wo = (x.w + 2 * pad - k) / s + 1;
        let mut y = Feature::zeros(out, ho, wo);
        for o in 0..out {
            for oy in 0..ho {
                for ox in 0..wo {
                    let mut sum = b[o] as f64;
                    for c in 0..x.c {
                        for ky in 0..k {
                            for kx in 0..k {
                                let yy = (oy * s + ky) as isize - pad as isize;
                                let xx = (ox * s + kx) as isize - pad as isize;
                                if yy >= 0 && xx >= 0 && (yy as usize) < x.h && (xx as usize) < x.w
                                {
                                    sum += (w[((o * x.c + c) * k + ky) * k + kx]
                                        * x.at(c, yy as usize, xx as usize))
                                        as f64;
                                }
                            }
                        }
                    }
                    y.data[(o * ho + oy) * wo + ox] = sum as f32;
                }
            }
        }
        y
    }

    #[test]
    fn im2col_matches_the_definition() {
        for (inp, out, h, w, k, s, pad) in [
            (3, 5, 7, 6, 3, 1, 1),
            (2, 4, 8, 8, 3, 2, 1),
            (4, 3, 5, 5, 1, 1, 0),
            (2, 2, 6, 4, 2, 2, 0),
        ] {
            let x = Feature::new(inp, h, w, rand(inp * h * w, 1));
            let wt = rand(out * inp * k * k, 2);
            let b = rand(out, 3);
            let got = Conv::new(&wt, &b, out, inp, k, s, pad).forward(&x);
            let want = direct(&x, &wt, &b, out, k, s, pad);
            assert_eq!((got.c, got.h, got.w), (want.c, want.h, want.w));
            for (a, b) in got.data.iter().zip(&want.data) {
                assert!((a - b).abs() < 1e-4, "{a} {b}");
            }
        }
    }

    /// Before its scale and shift, every group has mean 0 and variance 1;
    /// and a case worked by hand.
    #[test]
    fn group_norm_normalizes_each_group() {
        let x = Feature::new(4, 3, 3, rand(36, 4).iter().map(|v| 3.0 * v + 1.0).collect());
        let y = group_norm(&x, 2, &[1.0; 4], &[0.0; 4], 1e-6);
        for g in 0..2 {
            let vals = &y.data[g * 18..(g + 1) * 18];
            let mean: f32 = vals.iter().sum::<f32>() / 18.0;
            let var: f32 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 18.0;
            assert!(
                mean.abs() < 1e-5 && (var - 1.0).abs() < 1e-4,
                "{mean} {var}"
            );
        }
        // One group of two channels, values 1, 2 and 3, 4: mean 2.5,
        // variance 1.25; channel 1 scaled by 2 and shifted by 1.
        let x = Feature::new(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let y = group_norm(&x, 1, &[1.0, 2.0], &[0.0, 1.0], 0.0);
        let z = |v: f32| (v - 2.5) / 1.25f32.sqrt();
        let want = [z(1.0), z(2.0), 2.0 * z(3.0) + 1.0, 2.0 * z(4.0) + 1.0];
        for (a, b) in y.data.iter().zip(want) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn upsampling_copies_each_value_to_a_square() {
        let x = Feature::new(2, 1, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let y = upsample2(&x);
        assert_eq!((y.c, y.h, y.w), (2, 2, 4));
        assert_eq!(
            y.data,
            vec![1., 1., 2., 2., 1., 1., 2., 2., 3., 3., 4., 4., 3., 3., 4., 4.]
        );
    }

    #[test]
    fn rows_round_trip() {
        let x = Feature::new(3, 2, 4, rand(24, 5));
        let m = x.to_rows();
        assert_eq!((m.rows, m.cols), (8, 3));
        assert_eq!(m.data[5 * 3 + 2], x.at(2, 1, 1));
        assert_eq!(Feature::from_rows(&m, 2, 4), x);
    }
}
