//! Adam, the optimizer (Kingma and Ba 2015), as in tiny-llm-in-rust: each
//! weight moves by its gradient's running average divided by the running
//! average of its square's root, so every weight takes steps of about the
//! learning rate whatever its gradient's scale. No weight decay here: the
//! models of stage 2 are small and trained on unlimited fresh data, so
//! there's nothing to overfit.

use crate::tensor::Mat;

pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<Mat<f32>>,
    v: Vec<Mat<f32>>,
    step: i32,
}

impl Adam {
    /// Running averages for parameters of these shapes, starting at zero.
    pub fn new(params: &[Mat<f32>]) -> Adam {
        let zeros = || params.iter().map(|p| Mat::zeros(p.rows, p.cols)).collect();
        Adam {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: zeros(),
            v: zeros(),
            step: 0,
        }
    }

    /// One update of every parameter, at learning rate `lr`. The running
    /// averages start at zero, so early on they underestimate; dividing by
    /// 1 − β^step corrects that ("bias correction").
    pub fn update(&mut self, params: &mut [Mat<f32>], grads: &[Mat<f32>], lr: f32) {
        assert_eq!(params.len(), grads.len());
        self.step += 1;
        let (b1, b2) = (self.beta1, self.beta2);
        let c1 = 1.0 - b1.powi(self.step);
        let c2 = 1.0 - b2.powi(self.step);
        for ((p, g), (m, v)) in params
            .iter_mut()
            .zip(grads)
            .zip(self.m.iter_mut().zip(self.v.iter_mut()))
        {
            for i in 0..p.data.len() {
                let gi = g.data[i];
                m.data[i] = b1 * m.data[i] + (1.0 - b1) * gi;
                v.data[i] = b2 * v.data[i] + (1.0 - b2) * gi * gi;
                let mh = m.data[i] / c1;
                let vh = v.data[i] / c2;
                p.data[i] -= lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

impl Adam {
    /// AdamW, as tiny-llm-in-rust trains: the Adam step, then each weight
    /// with `decay[i]` set shrunk by lr × weight_decay (decoupled weight
    /// decay, applied to matrices and embeddings, not to biases or layer
    /// norms).
    pub fn update_decayed(
        &mut self,
        params: &mut [Mat<f32>],
        grads: &[Mat<f32>],
        lr: f32,
        decay: &[bool],
        weight_decay: f32,
    ) {
        assert_eq!(params.len(), decay.len());
        for (p, &d) in params.iter_mut().zip(decay) {
            if d {
                for x in p.data.iter_mut() {
                    *x -= lr * weight_decay * *x;
                }
            }
        }
        self.update(params, grads, lr);
    }
}

/// Scale the gradients down so their combined length (the square root of
/// the sum of every squared value) is at most `max`; returns the length
/// before clipping. tiny-llm-in-rust clips at 1.0.
pub fn clip_gradients(grads: &mut [Mat<f32>], max: f32) -> f32 {
    let norm = grads
        .iter()
        .flat_map(|g| g.data.iter())
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > max {
        let s = max / norm;
        for g in grads.iter_mut() {
            for x in g.data.iter_mut() {
                *x *= s;
            }
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first step moves every weight by almost exactly the learning
    /// rate, against its gradient's sign, whatever the gradient's size.
    #[test]
    fn first_step_is_the_learning_rate() {
        let mut p = vec![Mat::from_vec(1, 3, vec![1.0, 1.0, 1.0])];
        let g = vec![Mat::from_vec(1, 3, vec![1000.0, -0.001, 3.0])];
        Adam::new(&p).update(&mut p, &g, 0.1);
        let want = [0.9, 1.1, 0.9];
        for (a, b) in p[0].data.iter().zip(want) {
            assert!((a - b).abs() < 1e-4, "{:?}", p[0].data);
        }
    }

    #[test]
    fn clipping_and_decay() {
        let mut g = vec![Mat::from_vec(1, 2, vec![3.0, 4.0])];
        assert_eq!(clip_gradients(&mut g, 1.0), 5.0);
        assert!((g[0].data[0] - 0.6).abs() < 1e-6 && (g[0].data[1] - 0.8).abs() < 1e-6);
        let mut small = vec![Mat::from_vec(1, 1, vec![0.5])];
        clip_gradients(&mut small, 1.0);
        assert_eq!(small[0].data[0], 0.5);
        // With a zero gradient only the decay moves a weight: by lr × wd × w.
        let mut p = vec![
            Mat::from_vec(1, 1, vec![2.0]),
            Mat::from_vec(1, 1, vec![2.0]),
        ];
        let z = vec![Mat::zeros(1, 1), Mat::zeros(1, 1)];
        Adam::new(&p).update_decayed(&mut p, &z, 0.1, &[true, false], 0.5);
        assert!((p[0].data[0] - 1.9).abs() < 1e-6 && p[1].data[0] == 2.0);
    }

    /// It minimizes a quadratic: (x − 3)², from x = 0.
    #[test]
    fn finds_a_minimum() {
        let mut p = vec![Mat::from_vec(1, 1, vec![0.0])];
        let mut adam = Adam::new(&p);
        for _ in 0..2000 {
            let g = vec![Mat::from_vec(1, 1, vec![2.0 * (p[0].data[0] - 3.0)])];
            adam.update(&mut p, &g, 0.01);
        }
        assert!((p[0].data[0] - 3.0).abs() < 1e-2, "{}", p[0].data[0]);
    }
}
