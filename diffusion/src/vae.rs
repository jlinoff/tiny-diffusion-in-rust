//! The decoder of Stable Diffusion's autoencoder (sd-vae-ft-ema), which
//! turns DiT-XL/2's 32 × 32 × 4 latents into 256 × 256 photographs
//! (PLAN.md, stage 7). Forward passes only, under diffusers' names.
//!
//! ```text
//! z → post_quant_conv (1 × 1) → conv_in (3 × 3)
//!   → middle: residual block, attention, residual block
//!   → up blocks, widest first: (layers + 1) residual blocks, then (all but the last) upsample and a 3 × 3 convolution
//!   → group norm, SiLU, conv_out (3 × 3) → the image, about −1 to 1
//! residual block: x + conv₂(SiLU(GN(conv₁(SiLU(GN(x)))))), with a 1 × 1 convolution on x when the width changes
//! ```

use crate::conv::{group_norm, silu, upsample2, Conv, Feature};
use crate::ditxl::{Linear, Weights};
use crate::safetensors::File;
use crate::tensor::Mat;

/// The scale the latents were normalized by in training: a latent from
/// DiT-XL/2 is divided by it before decoding.
pub const SCALING: f32 = 0.18215;

/// Group normalization's ε throughout the decoder.
const EPS: f32 = 1e-6;

#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    /// Widths of the encoder's blocks; the decoder uses them in reverse.
    pub widths: Vec<usize>,
    pub layers: usize,
    pub groups: usize,
    pub latent: usize,
}

impl VaeConfig {
    /// From diffusers' configuration JSON, found by name.
    pub fn from_json(json: &str) -> Result<VaeConfig, String> {
        let after = |key: &str| -> Result<&str, String> {
            let at = json
                .find(&format!("\"{key}\""))
                .ok_or(format!("no {key:?} in the configuration"))?;
            let rest = json[at + key.len() + 2..].trim_start();
            Ok(rest
                .strip_prefix(':')
                .ok_or(format!("{key}: no value"))?
                .trim_start())
        };
        let num = |key: &str| -> Result<usize, String> {
            let d: String = after(key)?
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            d.parse().map_err(|_| format!("{key} isn't a whole number"))
        };
        let list = after("block_out_channels")?;
        let end = list.find(']').ok_or("block_out_channels: no closing ]")?;
        let widths = list[1..end]
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .map_err(|_| "block_out_channels: not numbers".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VaeConfig {
            widths,
            layers: num("layers_per_block")?,
            groups: num("norm_num_groups")?,
            latent: num("latent_channels")?,
        })
    }
}

struct Norm {
    g: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    fn load(w: &Weights, name: &str) -> Result<Norm, String> {
        Ok(Norm {
            g: w.get(&format!("{name}.weight"))?.data.clone(),
            b: w.get(&format!("{name}.bias"))?.data.clone(),
        })
    }
}

fn conv(w: &Weights, name: &str, pad: usize) -> Result<Conv, String> {
    let t = w.get(&format!("{name}.weight"))?;
    if t.shape.len() != 4 || t.shape[2] != t.shape[3] {
        return Err(format!(
            "{name}.weight: expected square filters, got {:?}",
            t.shape
        ));
    }
    let b = w.get(&format!("{name}.bias"))?;
    Ok(Conv::new(
        &t.data, &b.data, t.shape[0], t.shape[1], t.shape[2], 1, pad,
    ))
}

struct Resnet {
    norm1: Norm,
    conv1: Conv,
    norm2: Norm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl Resnet {
    fn load(w: &Weights, name: &str) -> Result<Resnet, String> {
        let n = |s: &str| format!("{name}.{s}");
        Ok(Resnet {
            norm1: Norm::load(w, &n("norm1"))?,
            conv1: conv(w, &n("conv1"), 1)?,
            norm2: Norm::load(w, &n("norm2"))?,
            conv2: conv(w, &n("conv2"), 1)?,
            shortcut: match w.find(&n("conv_shortcut.weight")) {
                Some(_) => Some(conv(w, &n("conv_shortcut"), 0)?),
                None => None,
            },
        })
    }

    fn forward(&self, x: &Feature, groups: usize) -> Feature {
        let mut h = group_norm(x, groups, &self.norm1.g, &self.norm1.b, EPS);
        silu(&mut h.data);
        let h = self.conv1.forward(&h);
        let mut h = group_norm(&h, groups, &self.norm2.g, &self.norm2.b, EPS);
        silu(&mut h.data);
        let mut h = self.conv2.forward(&h);
        match &self.shortcut {
            Some(s) => h.add_assign(&s.forward(x)),
            None => h.add_assign(x),
        }
        h
    }
}

/// The middle's attention: one head over all positions, with the input
/// added back.
struct Attn {
    norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl Attn {
    fn forward(&self, x: &Feature, groups: usize) -> Feature {
        let h = group_norm(x, groups, &self.norm.g, &self.norm.b, EPS).to_rows();
        let (q, k, v) = (self.q.forward(&h), self.k.forward(&h), self.v.forward(&h));
        let scale = 1.0 / (x.c as f32).sqrt();
        let mut w: Mat<f32> = q.matmul(&k.transpose());
        for i in 0..w.rows {
            let row = w.row_mut(i);
            let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum = 0.0;
            for e in row.iter_mut() {
                *e = ((*e - max) * scale).exp();
                sum += *e;
            }
            for e in row.iter_mut() {
                *e /= sum;
            }
        }
        let a = self.out.forward(&w.matmul(&v));
        let mut y = Feature::from_rows(&a, x.h, x.w);
        y.add_assign(x);
        y
    }
}

struct Up {
    resnets: Vec<Resnet>,
    upsample: Option<Conv>,
}

pub struct Decoder {
    pub cfg: VaeConfig,
    post_quant: Conv,
    conv_in: Conv,
    mid1: Resnet,
    attn: Attn,
    mid2: Resnet,
    ups: Vec<Up>,
    norm_out: Norm,
    conv_out: Conv,
}

impl Decoder {
    pub fn load(file: &File, cfg: VaeConfig) -> Result<Decoder, String> {
        let w = Weights::new(file);
        let a = "decoder.mid_block.attentions.0";
        let mut ups = Vec::new();
        for i in 0..cfg.widths.len() {
            let resnets = (0..=cfg.layers)
                .map(|j| Resnet::load(&w, &format!("decoder.up_blocks.{i}.resnets.{j}")))
                .collect::<Result<Vec<_>, _>>()?;
            let upsample = if i + 1 < cfg.widths.len() {
                Some(conv(
                    &w,
                    &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                    1,
                )?)
            } else {
                None
            };
            ups.push(Up { resnets, upsample });
        }
        Ok(Decoder {
            post_quant: conv(&w, "post_quant_conv", 0)?,
            conv_in: conv(&w, "decoder.conv_in", 1)?,
            mid1: Resnet::load(&w, "decoder.mid_block.resnets.0")?,
            attn: Attn {
                norm: Norm::load(&w, &format!("{a}.group_norm"))?,
                q: Linear::load_named(&w, &format!("{a}.to_q"))?,
                k: Linear::load_named(&w, &format!("{a}.to_k"))?,
                v: Linear::load_named(&w, &format!("{a}.to_v"))?,
                out: Linear::load_named(&w, &format!("{a}.to_out.0"))?,
            },
            mid2: Resnet::load(&w, "decoder.mid_block.resnets.1")?,
            ups,
            norm_out: Norm::load(&w, "decoder.conv_norm_out")?,
            conv_out: conv(&w, "decoder.conv_out", 1)?,
            cfg,
        })
    }

    /// A latent (latent channels × h × w, already divided by `SCALING`)
    /// to an image of 3 × 2ᵏh × 2ᵏw, values about −1 to 1.
    pub fn decode(&self, z: &Feature) -> Feature {
        let g = self.cfg.groups;
        let x = self.post_quant.forward(z);
        let x = self.conv_in.forward(&x);
        let x = self.mid1.forward(&x, g);
        let x = self.attn.forward(&x, g);
        let mut x = self.mid2.forward(&x, g);
        for up in &self.ups {
            for r in &up.resnets {
                x = r.forward(&x, g);
            }
            if let Some(c) = &up.upsample {
                x = c.forward(&upsample2(&x));
            }
        }
        let mut x = group_norm(&x, g, &self.norm_out.g, &self.norm_out.b, EPS);
        silu(&mut x.data);
        self.conv_out.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors;

    /// The tiny decoder built by tools/stage7/fixtures.py in diffusers,
    /// with random weights: the same image.
    #[test]
    fn matches_diffusers_on_the_tiny_fixture() {
        let f = safetensors::decode(include_bytes!("../tests/data/vae-tiny.safetensors")).unwrap();
        let cfg = VaeConfig::from_json(&f.metadata["config"]).unwrap();
        assert_eq!(
            (cfg.widths.clone(), cfg.layers, cfg.groups, cfg.latent),
            (vec![8, 16], 2, 4, 4)
        );
        let d = Decoder::load(&f, cfg).unwrap();
        let w = Weights::new(&f);
        let z = Feature::new(4, 4, 4, w.get("input.latent").unwrap().data.clone());
        let img = d.decode(&z);
        let want = &w.get("output").unwrap().data;
        assert_eq!((img.c, img.h, img.w), (3, 8, 8));
        let scale = want.iter().fold(0f32, |a, b| a.max(b.abs()));
        let worst = img
            .data
            .iter()
            .zip(want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()));
        assert!(worst <= 1e-4 * scale, "worst {worst}, largest {scale}");
    }
}
