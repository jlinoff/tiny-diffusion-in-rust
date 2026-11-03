//! `diffusion`: the command line. Each stage of PLAN.md adds its commands.
//!
//! Stage 2, points in a plane:
//! - `train2d`: train the MLP by flow matching or DDPM, write a model file
//! - `sample2d`: generate points, measure them against the data, plot them
//! - `steps2d`: how the result changes with the number of sampling steps
//! - `exact`: the same question with the exact Gaussian fields, no network
//!
//! Stage 3, images: `train-img`, `sample-img`.
//!
//! Stage 4, conditioning: `classifier` trains the measuring instrument,
//! `measure` uses it on a model's samples, and `style` asks whether the
//! noise decides how a digit is written.
//!
//! Stage 5, looking inside: `inside` runs the probes of PLAN.md's stage 5.
//!
//! Stage 6, text: `train-text` trains the masked diffusion language model on
//! tiny-llm-in-rust's corpus and reports its diffusion bound; `sample-text`
//! generates, infills and times it.

use clap::{Parser, Subcommand};
use mini_diffusion::classify::{
    measure, Classifier, ClassifierConfig, ClassifierTraining, CONFIDENT,
};
use mini_diffusion::data2d::{self, noise, sample};
use mini_diffusion::ddpm::{self, gaussian_eps, Schedule};
use mini_diffusion::dit::{Dit, DitConfig};
use mini_diffusion::flow::{euler, GaussianTruth};
use mini_diffusion::image::write_grid;
use mini_diffusion::metrics::energy_distance;
use mini_diffusion::mlp::{Mlp, MlpConfig};
use mini_diffusion::rng::Rng;
use mini_diffusion::svg::{figure, grid, Panel};
use mini_diffusion::tensor::Mat;
use mini_diffusion::train2d::{train, Objective, TrainConfig};
use mini_diffusion::trainimg::{labelled, ImgTrainConfig, Source};
use mini_diffusion::{blas, safetensors};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "diffusion",
    version,
    about = "A small diffusion model, built by hand (see PLAN.md)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show what this build contains.
    Info,
    /// Train the stage-2 MLP on points in a plane.
    Train2d {
        /// The data: moons, spiral, rings or gaussian.
        #[arg(long, default_value = "moons")]
        data: String,
        /// What to predict: flow (the velocity) or ddpm (the noise).
        #[arg(long, default_value = "flow")]
        objective: String,
        #[arg(long, default_value_t = 5000)]
        steps: u64,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 2e-3)]
        lr: f32,
        #[arg(long, default_value_t = 100)]
        warmup: u64,
        /// Neurons per hidden layer.
        #[arg(long, default_value_t = 128)]
        width: usize,
        /// Hidden layers.
        #[arg(long, default_value_t = 3)]
        depth: usize,
        /// Sine-cosine pairs in the time embedding.
        #[arg(long, default_value_t = 8)]
        freqs: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Print the mean loss every this many steps.
        #[arg(long, default_value_t = 500)]
        log_every: u64,
        /// The model file to write.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Generate points with a trained model, measure and plot them.
    Sample2d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 2000)]
        n: usize,
        /// Sampling steps (Euler for flow, DDIM for ddpm; ancestral DDPM
        /// always takes all 1,000).
        #[arg(long, default_value_t = 50)]
        steps: usize,
        /// For ddpm models: ddim or ancestral.
        #[arg(long, default_value = "ddim")]
        sampler: String,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        /// The SVG figure to write.
        #[arg(long)]
        svg: Option<PathBuf>,
    },
    /// Generate with several step counts: the distance to the data for each.
    Steps2d {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16,32,64,128")]
        steps: Vec<usize>,
        #[arg(long, default_value_t = 2000)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        #[arg(long)]
        svg: Option<PathBuf>,
    },
    /// Step counts with the exact fields for Gaussian data: no network.
    Exact {
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "1,2,5,10,25,50,100,250,1000"
        )]
        steps: Vec<usize>,
        #[arg(long, default_value_t = 20000)]
        n: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
    },
    /// Train the DiT on images (stage 3).
    TrainImg {
        /// The images: mnist, fashion (downloaded into --data-dir by make)
        /// or shapes (synthetic, any size).
        #[arg(long, default_value = "mnist")]
        data: String,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// For shapes: the image side, in pixels.
        #[arg(long, default_value_t = 28)]
        size: usize,
        /// For shapes: 1 (grayscale) or 3 (color).
        #[arg(long, default_value_t = 1)]
        channels: usize,
        /// Patch side in pixels; it must divide the image's height and width.
        #[arg(long, default_value_t = 4)]
        patch: usize,
        /// d, the length of each token's vector.
        #[arg(long, default_value_t = 128)]
        width: usize,
        #[arg(long, default_value_t = 6)]
        blocks: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 8)]
        freqs: usize,
        #[arg(long, default_value_t = 20000)]
        steps: u64,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(long, default_value_t = 500)]
        warmup: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 100)]
        log_every: u64,
        /// Write a checkpoint (the model so far) every this many steps.
        #[arg(long, default_value_t = 1000)]
        save_every: u64,
        /// Train with labels (stage 4): the model can then be asked for a
        /// class.
        #[arg(long)]
        conditional: bool,
        /// With --conditional: the probability of dropping an image's label.
        #[arg(long, default_value_t = 0.1)]
        drop: f32,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Generate images with a trained DiT, as PNG grids.
    SampleImg {
        #[arg(long)]
        model: PathBuf,
        /// Images to generate (a grid of 8 across).
        #[arg(long, default_value_t = 64)]
        n: usize,
        /// Euler steps.
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        /// The grid of generated images.
        #[arg(long)]
        png: PathBuf,
        /// Also: 8 generations shown at 9 moments from noise to image.
        #[arg(long)]
        progress: Option<PathBuf>,
        /// Also: a grid of real images from the data set, for comparison.
        #[arg(long)]
        real: Option<PathBuf>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Pixels per image pixel in the grids.
        #[arg(long, default_value_t = 3)]
        zoom: usize,
        /// For a model with labels: draw this class in every image.
        #[arg(long)]
        label: Option<usize>,
        /// For a model with labels: one row per class, with the same noise
        /// down each column (n is then the number of columns).
        #[arg(long)]
        classes: bool,
        /// For a model with labels: the guidance scale w (1: plain
        /// conditional; 0: unconditional).
        #[arg(long, default_value_t = 1.0)]
        guidance: f32,
    },
    /// Train the digit classifier, stage 4's measuring instrument.
    Classifier {
        /// mnist, fashion or shapes.
        #[arg(long, default_value = "mnist")]
        data: String,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// For shapes: the image side, in pixels.
        #[arg(long, default_value_t = 28)]
        size: usize,
        /// For shapes: 1 (grayscale) or 3 (color).
        #[arg(long, default_value_t = 1)]
        channels: usize,
        /// Neurons per hidden layer.
        #[arg(long, default_value_t = 256)]
        hidden: usize,
        /// Hidden layers.
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 5000)]
        steps: u64,
        #[arg(long, default_value_t = 128)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f32,
        #[arg(long, default_value_t = 100)]
        warmup: u64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 500)]
        log_every: u64,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Generate images with a DiT and measure them with the classifier.
    Measure {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        classifier: PathBuf,
        /// Images to generate and measure.
        #[arg(long, default_value_t = 1000)]
        n: usize,
        /// Euler steps.
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        /// For a model with labels: guidance scales to measure, each with an
        /// equal share of every class requested (w = 0: unconditional).
        #[arg(long, value_delimiter = ',', default_value = "0,1,1.5,2,3,5,8")]
        guidance: Vec<f32>,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
    },
    /// Look inside a trained DiT (stage 5): its guess of the finished image
    /// along the way, how far guidance pushes, how straight the paths are,
    /// and how far attention reaches.
    Inside {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        classifier: PathBuf,
        /// Generations to probe.
        #[arg(long, default_value_t = 200)]
        n: usize,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        /// Also: the preview grid, 8 generations at 9 moments, each as the
        /// model's guess of the finished image above its velocity.
        #[arg(long)]
        preview: Option<PathBuf>,
        /// Also: attention maps for the centre patch of one generation, one
        /// row per block, one column per moment.
        #[arg(long)]
        attention: Option<PathBuf>,
    },
    /// Train the masked diffusion language model (stage 6) on a token-ID file
    /// written by tiny-llm-in-rust's tokenizer, and measure its diffusion
    /// bound on the held-out tokens.
    TrainText {
        /// The token IDs (tiny-llm-in-rust's out/corpus.ids).
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, default_value_t = 128)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        blocks: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 256)]
        context: usize,
        #[arg(long, default_value_t = 512)]
        ffn: usize,
        #[arg(long, default_value_t = 2000)]
        steps: u64,
        #[arg(long, default_value_t = 8)]
        batch: usize,
        #[arg(long, default_value_t = 2e-3)]
        lr: f32,
        #[arg(long, default_value_t = 100)]
        warmup: u64,
        #[arg(long, default_value_t = 0.1)]
        min_lr_fraction: f32,
        #[arg(long, default_value_t = 0.1)]
        weight_decay: f32,
        #[arg(long, default_value_t = 1.0)]
        grad_clip: f32,
        #[arg(long, default_value_t = 0.95)]
        beta2: f32,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Lighter masking: t drawn from (t-min, 1), so at most 1 - t-min of
        /// each training window is masked (0: anywhere from none to all).
        #[arg(long, default_value_t = 0.0)]
        t_min: f64,
        /// The control: train the same network causally, predicting each
        /// next token, as tiny-llm-in-rust trains its own (no masking; its
        /// validation loss is reported in place of the diffusion bound).
        #[arg(long)]
        next_token: bool,
        /// Rotary positions (RoPE) in place of the learned position table:
        /// each query and key rotated by its position, so attention sees
        /// the distance between two tokens from the first step.
        #[arg(long)]
        rope: bool,
        /// The share of the tokens held out, at the end, as tiny-llm-in-rust.
        #[arg(long, default_value_t = 0.05)]
        valid_fraction: f64,
        /// Held-out windows the bound is measured on, and values of t each.
        #[arg(long, default_value_t = 64)]
        valid_windows: usize,
        #[arg(long, default_value_t = 8)]
        draws: usize,
        #[arg(long, default_value_t = 100)]
        log_every: u64,
        /// tiny-llm-in-rust's trained model, to print its validation loss
        /// beside the diffusion bound (out/model.safetensors there).
        #[arg(long)]
        compare: Option<PathBuf>,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Generate stories with the masked diffusion model at several step
    /// counts, infill a held-out story, and time it (stage 6).
    SampleText {
        #[arg(long)]
        model: PathBuf,
        /// tiny-llm-in-rust's out/tok.bpe, for decoding.
        #[arg(long)]
        tokenizer: PathBuf,
        /// The same token IDs the model was trained on.
        #[arg(long)]
        ids: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "16,64,256")]
        steps: Vec<usize>,
        /// Stories at each step count.
        #[arg(long, default_value_t = 8)]
        n: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Which drawn tokens each step keeps: the most confident, or
        /// positions chosen at random (MDLM's sampler).
        #[arg(long, default_value = "confidence", value_parser = ["confidence", "random"])]
        order: String,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, default_value_t = 0.05)]
        valid_fraction: f64,
        /// Files written: <prefix>-s16.txt and so on, <prefix>-real.txt,
        /// <prefix>-infill.txt.
        #[arg(long, default_value = "out/text")]
        out_prefix: String,
    },
    /// For a model with labels: does the starting noise decide the style?
    /// Draws every class from the same noises and correlates slant,
    /// position and ink between classes (PLAN.md, Stage 4, prediction 6).
    Style {
        #[arg(long)]
        model: PathBuf,
        /// Noises: each is drawn once as every class.
        #[arg(long, default_value_t = 100)]
        n: usize,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 2)]
        seed: u64,
        #[arg(long, default_value_t = 1.0)]
        guidance: f32,
    },
    /// Stage 7: photographs from DiT-XL/2 and its decoder, downloaded and
    /// converted by make dit-download (non-commercial use only: PLAN.md,
    /// Stage 7). A row per class, a column per starting noise.
    DitSample {
        #[arg(long, default_value = "models/dit-xl-2-256.safetensors")]
        dit: PathBuf,
        #[arg(long, default_value = "models/sd-vae-ft-ema-decoder.safetensors")]
        vae: PathBuf,
        /// Classes, by number or by name from --names, comma-separated.
        #[arg(long, default_value = "207,279,360,387,974,88,979,417")]
        classes: String,
        /// The 1,000 ImageNet class names, one per line (written by make
        /// dit-download).
        #[arg(long, default_value = "models/imagenet_classes.txt")]
        names: PathBuf,
        /// Starting noises: the columns.
        #[arg(long, default_value_t = 8)]
        noises: usize,
        #[arg(long, default_value_t = 4.0)]
        guidance: f32,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Whose arithmetic: original (the DiT repository's, the model's
        /// training) or diffusers.
        #[arg(long, default_value = "original", value_parser = ["original", "diffusers"])]
        reference: String,
        /// Guide only the first n of ε's channels (the DiT repository's own
        /// sampler guides 3 of 4); 0 means all.
        #[arg(long, default_value_t = 0)]
        guide_channels: usize,
        /// A grid made earlier: if its legend shows the same settings and
        /// rows, copy it instead of generating again.
        #[arg(long)]
        same_as: Option<PathBuf>,
        #[arg(short, long, default_value = "out/dit.png")]
        output: PathBuf,
    },
    /// Stage 7: the model's guess of the finished photograph during one
    /// generation, decoded at several steps (stage 5's preview at full
    /// scale). A row per noise, a column per moment, the finished
    /// photograph last.
    DitInside {
        #[arg(long, default_value = "models/dit-xl-2-256.safetensors")]
        dit: PathBuf,
        #[arg(long, default_value = "models/sd-vae-ft-ema-decoder.safetensors")]
        vae: PathBuf,
        #[arg(long, default_value = "207")]
        class: String,
        #[arg(long, default_value = "models/imagenet_classes.txt")]
        names: PathBuf,
        #[arg(long, default_value_t = 4)]
        noises: usize,
        /// Moments shown, evenly spaced through the steps.
        #[arg(long, default_value_t = 8)]
        moments: usize,
        #[arg(long, default_value_t = 4.0)]
        guidance: f32,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(short, long, default_value = "out/dit-inside.png")]
        output: PathBuf,
    },
    /// Stage 7: DiT-XL/2 and its decoder against the official code's
    /// outputs (tools/stage7/reference.py), number for number, with
    /// diffusers' arithmetic; and the time per pass (PLAN.md, Stage 7,
    /// predictions 2 and 3).
    DitCheck {
        #[arg(long, default_value = "models/dit-xl-2-256.safetensors")]
        dit: PathBuf,
        #[arg(long, default_value = "models/sd-vae-ft-ema-decoder.safetensors")]
        vae: PathBuf,
        #[arg(long, default_value = "models/reference-dit.safetensors")]
        dit_reference: PathBuf,
        #[arg(long, default_value = "models/reference-vae.safetensors")]
        vae_reference: PathBuf,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("diffusion: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Info => {
            println!("diffusion {}", env!("CARGO_PKG_VERSION"));
            println!("  large matrix products: {}", blas::backend());
            println!(
                "  stages built: foundation; points in a plane; images; stage 4 under way (PLAN.md, Order of work)"
            );
            Ok(())
        }
        Command::Train2d {
            data,
            objective,
            steps,
            batch,
            lr,
            warmup,
            width,
            depth,
            freqs,
            seed,
            log_every,
            output,
        } => {
            let objective = Objective::parse(&objective)?;
            sample(&data, 1, &mut Rng::new(0))?; // the name, before any work
            check_writable(&output)?;
            let cfg = MlpConfig {
                dim: 2,
                width,
                depth,
                freqs,
            };
            let tc = TrainConfig {
                data: data.clone(),
                objective,
                steps,
                batch,
                learning_rate: lr,
                warmup,
                seed,
            };
            println!(
                "Training an MLP ({} parameters: width {width}, {depth} hidden layers, {freqs} time frequencies) on {data} by {}; {steps} steps of {batch}",
                cfg.parameters(),
                objective.name()
            );
            println!("   step    loss        lr   steps/s");
            let mut model = Mlp::new(cfg, seed);
            let (mut sum, mut count, mut tick) = (0.0f64, 0u64, Instant::now());
            let start = Instant::now();
            let losses = train(&mut model, &tc, |step, loss| {
                sum += loss as f64;
                count += 1;
                if step % log_every == 0 || step == steps {
                    let rate = count as f64 / tick.elapsed().as_secs_f64();
                    println!(
                        "  {step:>5}   {:.4}  {:.2e}  {rate:8.0}",
                        sum / count as f64,
                        tc.lr(step)
                    );
                    (sum, count, tick) = (0.0, 0, Instant::now());
                }
            });
            let tail = &losses[losses.len().saturating_sub(500)..];
            let final_loss = tail.iter().map(|&l| l as f64).sum::<f64>() / tail.len() as f64;
            let mut meta = BTreeMap::new();
            meta.insert("objective".into(), objective.name().into());
            meta.insert("data".into(), data);
            meta.insert("steps".into(), steps.to_string());
            meta.insert("seed".into(), seed.to_string());
            meta.insert("final_loss".into(), format!("{final_loss:.5}"));
            let f = model.to_file(meta);
            safetensors::write(&output, &f.tensors, &f.metadata)?;
            println!(
                "Wrote {} in {:.1} s; mean loss over the last 500 steps {final_loss:.4}",
                output.display(),
                start.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Command::Sample2d {
            model,
            n,
            steps,
            sampler,
            seed,
            svg,
        } => {
            let m = load(&model)?;
            let mut rng = Rng::new(seed);
            let x0 = noise(n, 2, &mut rng);
            let (out, paths) = generate(&m, x0, steps, &sampler, true, &mut rng)?;
            let data = sample(&m.data, n, &mut Rng::new(seed ^ 0xDA7A))?;
            let ed = energy_distance(&cap(&out), &cap(&data));
            let floor = energy_distance(&cap(&data), &cap(&sample(&m.data, n, &mut rng)?));
            println!(
                "{n} points from {} ({} model, {}): energy distance to the data {ed:.4} (two data samples: {floor:.4})",
                model.display(),
                m.objective.name(),
                sampler_name(&m, steps, &sampler)
            );
            if let Some(path) = svg {
                let mut panels = vec![
                    panel("data", &data, "#1f77b4"),
                    panel(
                        &format!("generated ({})", sampler_name(&m, steps, &sampler)),
                        &out,
                        "#d62728",
                    ),
                ];
                if !paths.is_empty() {
                    let mut p = Panel::new("paths from noise (grey) to points (red)");
                    p.points(&paths[0], "#999", 1.2, 0.5);
                    p.paths(&paths, 150, "#ff7f0e");
                    p.points(paths.last().unwrap(), "#d62728", 1.2, 0.7);
                    panels.push(p);
                }
                if m.objective == Objective::Flow {
                    for t in [0.0, 0.5, 0.9] {
                        let g = grid(15);
                        let mut p = Panel::new(&format!("velocity field at t = {t}"));
                        p.points(&data, "#1f77b4", 1.0, 0.15);
                        p.arrows(&g, &m.mlp.predict(&g, t), 0.12, "#333");
                        panels.push(p);
                    }
                }
                let caption = format!(
                    "{}: {} model, {}",
                    m.data,
                    m.objective.name(),
                    sampler_name(&m, steps, &sampler)
                );
                write_file(&path, &figure(&caption, &panels, 3))?;
            }
            Ok(())
        }
        Command::Steps2d {
            model,
            steps,
            n,
            seed,
            svg,
        } => {
            let m = load(&model)?;
            let mut rng = Rng::new(seed);
            let x0 = noise(n, 2, &mut rng);
            let data = sample(&m.data, n, &mut Rng::new(seed ^ 0xDA7A))?;
            let floor = energy_distance(&cap(&data), &cap(&sample(&m.data, n, &mut rng)?));
            println!(
                "{} ({} model): energy distance to the data, by sampling steps",
                model.display(),
                m.objective.name()
            );
            println!("  two data samples: {floor:.4}");
            println!("   steps   energy distance");
            let mut panels = Vec::new();
            let samplers: Vec<(String, usize)> = match m.objective {
                Objective::Flow => steps.iter().map(|&k| ("euler".to_string(), k)).collect(),
                Objective::Ddpm => {
                    let mut v: Vec<(String, usize)> =
                        steps.iter().map(|&k| ("ddim".to_string(), k)).collect();
                    v.push(("ancestral".into(), ddpm::STEPS));
                    v
                }
            };
            for (s, k) in samplers {
                let (out, _) = generate(&m, x0.clone(), k, &s, false, &mut Rng::new(seed + 1))?;
                let ed = energy_distance(&cap(&out), &cap(&data));
                let label = sampler_name(&m, k, &s);
                println!("  {k:>6}   {ed:.4}   {label}");
                panels.push(panel(&format!("{label}: {ed:.3}"), &out, "#d62728"));
            }
            if let Some(path) = svg {
                let caption = format!(
                    "{}: {} model, by sampling steps (energy distance to the data)",
                    m.data,
                    m.objective.name()
                );
                write_file(&path, &figure(&caption, &panels, 4))?;
            }
            Ok(())
        }
        Command::Exact { steps, n, seed } => {
            let (mean, std) = (data2d::GAUSSIAN_MEAN, data2d::GAUSSIAN_STD);
            let truth = GaussianTruth {
                mean: mean.to_vec(),
                std: std.to_vec(),
            };
            let sched = Schedule::linear(ddpm::STEPS);
            let x0 = noise(n, 2, &mut Rng::new(seed));
            println!("Exact fields for Gaussian data (std {:?}), {n} samples: the samples' standard deviations", std);
            println!("   steps   flow (Euler)        DDPM (DDIM)");
            for k in steps {
                let (f, _) = euler(x0.clone(), k, |x, t| truth.velocity(x, t), false);
                let d = sched.ddim(x0.clone(), k.min(ddpm::STEPS), |x, i| {
                    gaussian_eps(&sched, &mean, &std, x, i)
                });
                let (sf, sd) = (data2d::column_stats(&f), data2d::column_stats(&d));
                println!(
                    "  {k:>6}   {:.3}  {:.3}        {:.3}  {:.3}",
                    sf[0].1, sf[1].1, sd[0].1, sd[1].1
                );
            }
            Ok(())
        }
        Command::TrainImg {
            data,
            data_dir,
            size,
            channels,
            patch,
            width,
            blocks,
            heads,
            freqs,
            steps,
            batch,
            lr,
            warmup,
            seed,
            log_every,
            save_every,
            conditional,
            drop,
            output,
        } => {
            check_writable(&output)?;
            let source = Source::open(&data, &data_dir, size, channels)?;
            let labels = if conditional { source.classes() } else { 0 };
            let cfg = DitConfig {
                image: source.shape(),
                patch,
                width,
                blocks,
                heads,
                freqs,
                labels,
            };
            cfg.check()?;
            let tc = ImgTrainConfig {
                steps,
                batch,
                learning_rate: lr,
                warmup,
                seed,
                drop,
            };
            let sh = cfg.image;
            println!(
                "Training a DiT ({} parameters: width {width}, {blocks} blocks, {heads} heads, {patch}×{patch} patches, {} tokens) on {data} ({}×{}×{}); {steps} steps of {batch}",
                cfg.parameters(),
                cfg.tokens(),
                sh.height,
                sh.width,
                sh.channels
            );
            if labels > 0 {
                println!(
                    "  with labels: {labels} classes, each image's dropped with probability {drop}"
                );
            }
            println!("   step    loss        lr   images/s   remaining");
            let mut model = Dit::new(cfg, seed);
            let meta = |done: u64, loss: f64| {
                let mut m = BTreeMap::new();
                m.insert("objective".to_string(), "flow".to_string());
                m.insert("data".to_string(), data.clone());
                m.insert("steps".to_string(), done.to_string());
                m.insert("total_steps".to_string(), steps.to_string());
                m.insert("seed".to_string(), seed.to_string());
                m.insert("final_loss".to_string(), format!("{loss:.5}"));
                m
            };
            let (mut sum, mut count, mut tick) = (0.0f64, 0u64, Instant::now());
            let start = Instant::now();
            let last = std::cell::Cell::new(0.0f64);
            let mut save_error = None;
            let losses = mini_diffusion::trainimg::train(
                &mut model,
                &source,
                &tc,
                save_every,
                |step, loss| {
                    sum += loss as f64;
                    count += 1;
                    if step % log_every == 0 || step == steps {
                        let secs = tick.elapsed().as_secs_f64();
                        let left = secs / count as f64 * (steps - step) as f64;
                        last.set(sum / count as f64);
                        println!(
                            "  {step:>5}   {:.4}  {:.2e}  {:9.0}   {:>9}",
                            last.get(),
                            tc.lr(step),
                            (count as f64 * batch as f64) / secs,
                            duration(left)
                        );
                        (sum, count, tick) = (0.0, 0, Instant::now());
                    }
                },
                |step, m| {
                    // The model so far, to the same file: sample from it
                    // while training continues.
                    let f = m.to_file(meta(step, last.get()));
                    match safetensors::write(&output, &f.tensors, &f.metadata) {
                        Ok(()) => {
                            println!("  checkpoint: step {step} written to {}", output.display())
                        }
                        Err(e) => save_error = Some(e),
                    }
                },
            );
            if let Some(e) = save_error {
                return Err(e);
            }
            let tail = &losses[losses.len().saturating_sub(500)..];
            let final_loss = tail.iter().map(|&l| l as f64).sum::<f64>() / tail.len() as f64;
            let f = model.to_file(meta(steps, final_loss));
            safetensors::write(&output, &f.tensors, &f.metadata)?;
            println!(
                "Wrote {} in {}; mean loss over the last 500 steps {final_loss:.4}",
                output.display(),
                duration(start.elapsed().as_secs_f64())
            );
            Ok(())
        }
        Command::SampleImg {
            model,
            n,
            steps,
            seed,
            png,
            progress,
            real,
            data_dir,
            zoom,
            label,
            classes,
            guidance,
        } => {
            let f = safetensors::read(&model)?;
            let dit = Dit::from_file(&f).map_err(|e| format!("{}: {e}", model.display()))?;
            println!("{}", describe_checkpoint(&model, &f.metadata));
            let shape = dit.cfg.image;
            let l = dit.cfg.labels;
            if l == 0 && (label.is_some() || classes || guidance != 1.0) {
                return Err(format!(
                    "{} has no labels: --label, --classes and --guidance need a model trained with --conditional",
                    model.display()
                ));
            }
            if label.is_some_and(|y| y >= l) {
                return Err(format!("--label runs from 0 to {}", l - 1));
            }
            // With --classes: one row per class, the same n noises in each.
            let (x0, ys, cols) = if classes {
                let base = noise(n, shape.values(), &mut Rng::new(seed));
                let mut data = Vec::with_capacity(l * base.data.len());
                for _ in 0..l {
                    data.extend_from_slice(&base.data);
                }
                let ys: Vec<usize> = (0..l).flat_map(|y| std::iter::repeat_n(y, n)).collect();
                (Mat::from_vec(l * n, shape.values(), data), ys, n)
            } else {
                let x0 = noise(n, shape.values(), &mut Rng::new(seed));
                let ys = match label {
                    Some(y) => vec![y; n],
                    None if l > 0 => (0..n).map(|i| i % l).collect(),
                    None => Vec::new(),
                };
                (x0, ys, 8)
            };
            let count = x0.rows;
            let start = Instant::now();
            let (out, path) = euler(
                x0,
                steps,
                |x, t| {
                    if l == 0 {
                        dit.predict(x, t)
                    } else {
                        dit.predict_guided(x, t, &ys, guidance)
                    }
                },
                progress.is_some(),
            );
            let what = if l > 0 {
                format!(", guidance w = {guidance}")
            } else {
                String::new()
            };
            println!(
                "{count} images from {} with {steps} Euler steps{what} in {:.1} s",
                model.display(),
                start.elapsed().as_secs_f64()
            );
            check_writable(&png)?;
            write_grid(&png, &out, shape, cols, zoom)?;
            println!("Wrote {}", png.display());
            if let Some(p) = progress {
                // 8 generations (rows) at 9 moments (columns), t = 0 to 1.
                let k = 8.min(count);
                let mut frames = Vec::new();
                for r in 0..k {
                    for j in 0..=8 {
                        let idx = j * steps / 8;
                        frames.extend_from_slice(path[idx].row(r));
                    }
                }
                let m = Mat::from_vec(k * 9, shape.values(), frames);
                check_writable(&p)?;
                write_grid(&p, &m, shape, 9, zoom)?;
                println!(
                    "Wrote {} (each row: one image from noise, t = 0, 1/8, …, 1)",
                    p.display()
                );
            }
            if let Some(p) = real {
                let name = f.metadata.get("data").cloned().unwrap_or_default();
                let src = Source::open(&name, &data_dir, shape.height, shape.channels)?;
                check_writable(&p)?;
                write_grid(&p, &src.batch(n, seed, 0), shape, 8, zoom)?;
                println!("Wrote {} (real {name} images)", p.display());
            }
            Ok(())
        }
        Command::Classifier {
            data,
            data_dir,
            size,
            channels,
            hidden,
            layers,
            steps,
            batch,
            lr,
            warmup,
            seed,
            log_every,
            output,
        } => {
            check_writable(&output)?;
            let train = labelled(&data, &data_dir, size, channels, false)?;
            let test = labelled(&data, &data_dir, size, channels, true)?;
            let classes = 1 + *train.labels.iter().max().unwrap_or(&0) as usize;
            let cfg = ClassifierConfig {
                inputs: train.shape.values(),
                hidden,
                layers,
                classes,
            };
            let tc = ClassifierTraining {
                steps,
                batch,
                learning_rate: lr,
                warmup,
                seed,
            };
            println!(
                "Training a classifier ({} parameters: {} inputs, {layers} hidden layers of {hidden}, {classes} classes) on {data}: {} training images, {} test images; {steps} steps of {batch}",
                cfg.parameters(),
                cfg.inputs,
                train.labels.len(),
                test.labels.len()
            );
            println!("   step    loss        lr");
            let mut clf = Classifier::new(cfg, seed);
            let (mut sum, mut count) = (0.0f64, 0u64);
            let start = Instant::now();
            clf.train(&train, &tc, |step, loss| {
                sum += loss as f64;
                count += 1;
                if step % log_every == 0 || step == steps {
                    println!(
                        "  {step:>5}   {:.4}  {:.2e}",
                        sum / count as f64,
                        tc.lr(step)
                    );
                    (sum, count) = (0.0, 0);
                }
            });
            // Training accuracy on (up to) the first 10,000 training images,
            // as many as the test set: a gap between the two means memorizing.
            let k = train.labels.len().min(10_000);
            let subset = mini_diffusion::image::Images {
                shape: train.shape,
                pixels: Mat::from_vec(k, cfg.inputs, train.pixels.data[..k * cfg.inputs].to_vec()),
                labels: train.labels[..k].to_vec(),
            };
            let (tr, te) = (clf.accuracy(&subset), clf.accuracy(&test));
            println!(
                "Accuracy: {:.2}% on the test images, {:.2}% on {k} training images ({})",
                100.0 * te,
                100.0 * tr,
                duration(start.elapsed().as_secs_f64())
            );
            let mut meta = BTreeMap::new();
            meta.insert("data".to_string(), data);
            meta.insert("steps".to_string(), steps.to_string());
            meta.insert("test_accuracy".to_string(), format!("{te:.4}"));
            meta.insert("train_accuracy".to_string(), format!("{tr:.4}"));
            let f = clf.to_file(meta);
            safetensors::write(&output, &f.tensors, &f.metadata)?;
            println!("Wrote {}", output.display());
            Ok(())
        }
        Command::Measure {
            model,
            classifier,
            n,
            steps,
            seed,
            guidance,
            data_dir,
        } => {
            let f = safetensors::read(&model)?;
            let dit = Dit::from_file(&f).map_err(|e| format!("{}: {e}", model.display()))?;
            println!("{}", describe_checkpoint(&model, &f.metadata));
            let cf = safetensors::read(&classifier)?;
            let clf =
                Classifier::from_file(&cf).map_err(|e| format!("{}: {e}", classifier.display()))?;
            let shape = dit.cfg.image;
            if clf.cfg.inputs != shape.values() {
                return Err(format!(
                    "the classifier takes {} values per image, the model makes {}",
                    clf.cfg.inputs,
                    shape.values()
                ));
            }
            let acc = cf
                .metadata
                .get("test_accuracy")
                .cloned()
                .unwrap_or("?".into());
            println!(
                "Classifier {}: trained on {}, test accuracy {acc}",
                classifier.display(),
                cf.metadata.get("data").map(String::as_str).unwrap_or("?")
            );
            if dit.cfg.labels > 0 {
                let data = f.metadata.get("data").cloned().unwrap_or_default();
                let real = labelled(&data, &data_dir, shape.height, shape.channels, true)?;
                return measure_guided(&dit, &clf, &real, n, steps, seed, &guidance);
            }
            let x0 = noise(n, shape.values(), &mut Rng::new(seed));
            let start = Instant::now();
            let (out, _) = euler(x0, steps, |x, t| dit.predict(x, t), false);
            println!(
                "{n} images with {steps} Euler steps in {}",
                duration(start.elapsed().as_secs_f64())
            );
            let ms = measure(&clf, &out, None);
            println!(
                "  confident (probability above {CONFIDENT}): {:.1}%",
                100.0 * ms.confident
            );
            println!("  overshoot (pixel values beyond -1 to 1 by more than a margin):");
            for (margin, f) in &ms.overshoot {
                println!("    margin {margin:<4}  {:6.2}%", 100.0 * f);
            }
            println!("  share of images in each class:");
            for (c, &k) in ms.counts.iter().enumerate() {
                println!("    {c}: {:5.1}%", 100.0 * k as f64 / n.max(1) as f64);
            }
            Ok(())
        }
        Command::Inside {
            model,
            classifier,
            n,
            steps,
            seed,
            data_dir,
            preview,
            attention,
        } => {
            let f = safetensors::read(&model)?;
            let dit = Dit::from_file(&f).map_err(|e| format!("{}: {e}", model.display()))?;
            println!("{}", describe_checkpoint(&model, &f.metadata));
            let cf = safetensors::read(&classifier)?;
            let clf =
                Classifier::from_file(&cf).map_err(|e| format!("{}: {e}", classifier.display()))?;
            let data = f.metadata.get("data").cloned().unwrap_or_default();
            let shape = dit.cfg.image;
            let train = labelled(&data, &data_dir, shape.height, shape.channels, false)?;
            inside_probes(&dit, &clf, &train, n, steps, seed, preview, attention)
        }
        Command::TrainText {
            ids,
            width,
            blocks,
            heads,
            context,
            ffn,
            steps,
            batch,
            lr,
            warmup,
            min_lr_fraction,
            weight_decay,
            grad_clip,
            beta2,
            seed,
            t_min,
            next_token,
            rope,
            valid_fraction,
            valid_windows,
            draws,
            log_every,
            compare,
            output,
        } => {
            use mini_diffusion::text::{split, valid_windows as vw, Corpus};
            use mini_diffusion::textmodel::{
                diffusion_bound, train, TextConfig, TextModel, TextTraining,
            };
            check_writable(&output)?;
            let corpus = Corpus::load(&ids)?;
            let (tr, va) = split(&corpus.ids, valid_fraction);
            if tr.len() <= context {
                return Err(format!(
                    "{} training tokens: too few for windows of {context}",
                    tr.len()
                ));
            }
            let cfg = TextConfig {
                vocab: corpus.vocab_size,
                width,
                blocks,
                heads,
                context,
                ffn,
                rope,
            };
            let tc = TextTraining {
                steps,
                batch,
                learning_rate: lr,
                warmup,
                min_lr_fraction,
                weight_decay,
                grad_clip,
                beta2,
                seed,
                t_min,
                next_token,
            };
            let kind = if next_token {
                "the control, a causal next-token model"
            } else {
                "a masked diffusion language model"
            };
            println!(
                "Training {kind} ({} parameters: width {width}, {blocks} blocks, {heads} heads, context {context}, V = {} + MASK) on {}: {} training tokens, {} held out; {steps} steps of {batch} windows",
                cfg.parameters(),
                cfg.vocab,
                ids.display(),
                tr.len(),
                va.len()
            );
            if next_token {
                println!("  causal attention, each position predicting the next token, no masking");
            }
            if rope {
                println!("  rotary positions (RoPE), no position table");
            }
            if t_min > 0.0 {
                println!(
                    "  lighter masking: at most {:.0}% of each window masked",
                    100.0 * (1.0 - t_min)
                );
            }
            println!("   step    loss        lr   tokens/s   remaining");
            let mut model = TextModel::new(cfg, seed);
            let (mut sum, mut count, mut tick) = (0.0f64, 0u64, Instant::now());
            let start = Instant::now();
            let losses = train(&mut model, tr, &tc, |step, loss| {
                sum += loss as f64;
                count += 1;
                if step % log_every == 0 || step == steps {
                    let secs = tick.elapsed().as_secs_f64();
                    println!(
                        "  {step:>5}   {:.4}  {:.2e}  {:9.0}   {:>9}",
                        sum / count as f64,
                        tc.lr(step),
                        (count as f64 * (batch * context) as f64) / secs,
                        duration(secs / count as f64 * (steps - step) as f64)
                    );
                    (sum, count, tick) = (0.0, 0, Instant::now());
                }
            });
            println!("Trained in {}", duration(start.elapsed().as_secs_f64()));
            let mut control = f64::NAN;
            if next_token {
                let windows = vw(va, context + 1, valid_windows);
                let v = mini_diffusion::textmodel::next_token_loss(&model, &windows);
                control = v;
                println!(
                    "Next-token validation loss on {} held-out windows: {v:.4} nats per token (the control: the same network trained as tiny-llm-in-rust's)",
                    windows.len()
                );
            }
            let windows = vw(va, context, valid_windows);
            let (bound, se) = if next_token {
                (f64::NAN, f64::NAN)
            } else if windows.is_empty() {
                println!("(no held-out window of {context} tokens: no bound measured)");
                (f64::NAN, f64::NAN)
            } else {
                let t0 = Instant::now();
                let (b, se) = diffusion_bound(&model, &windows, draws, seed);
                println!(
                    "Diffusion bound on {} held-out windows, {draws} values of t each: {b:.4} ± {se:.4} nats per token ({})",
                    windows.len(),
                    duration(t0.elapsed().as_secs_f64())
                );
                (b, se)
            };
            if let Some(p) = compare {
                let f = safetensors::read(&p)?;
                match f.metadata.get("valid_loss") {
                    Some(v) => {
                        let v: f64 = v.parse().map_err(|_| "bad valid_loss".to_string())?;
                        let step = f.metadata.get("step").map(String::as_str).unwrap_or("?");
                        println!(
                            "tiny-llm-in-rust's validation loss ({}, step {step}): {v:.4} nats per token; {} is {:+.1}% of it",
                            p.display(),
                            if next_token { "the control's" } else { "the diffusion bound" },
                            100.0 * (if next_token { control } else { bound } / v - 1.0)
                        );
                    }
                    None => println!("{}: no valid_loss in its metadata", p.display()),
                }
            }
            let tail = &losses[losses.len().saturating_sub(100)..];
            let final_loss = tail.iter().map(|&l| l as f64).sum::<f64>() / tail.len() as f64;
            let mut meta = BTreeMap::new();
            meta.insert("data".to_string(), ids.display().to_string());
            meta.insert("steps".to_string(), steps.to_string());
            meta.insert("seed".to_string(), seed.to_string());
            meta.insert("t_min".to_string(), t_min.to_string());
            meta.insert("final_loss".to_string(), format!("{final_loss:.5}"));
            meta.insert("bound".to_string(), format!("{bound:.5}"));
            meta.insert("bound_se".to_string(), format!("{se:.5}"));
            let f = model.to_file(meta);
            safetensors::write(&output, &f.tensors, &f.metadata)?;
            println!("Wrote {}", output.display());
            Ok(())
        }
        Command::DitSample {
            dit,
            vae,
            classes,
            names,
            noises,
            guidance,
            steps,
            seed,
            reference,
            guide_channels,
            same_as,
            output,
        } => dit_sample(
            &dit,
            &vae,
            &classes,
            &names,
            (noises, guidance, guide_channels, steps, seed),
            &reference,
            same_as.as_deref(),
            &output,
        ),
        Command::DitInside {
            dit,
            vae,
            class,
            names,
            noises,
            moments,
            guidance,
            steps,
            seed,
            output,
        } => dit_inside(
            &dit, &vae, &class, &names, noises, moments, guidance, steps, seed, &output,
        ),
        Command::DitCheck {
            dit,
            vae,
            dit_reference,
            vae_reference,
        } => dit_check(&dit, &vae, &dit_reference, &vae_reference),
        Command::SampleText {
            model,
            tokenizer,
            ids,
            steps,
            n,
            temperature,
            top_k,
            order,
            seed,
            valid_fraction,
            out_prefix,
        } => sample_text(
            &model,
            &tokenizer,
            &ids,
            &steps,
            n,
            mini_diffusion::textmodel::Sampling {
                temperature,
                top_k,
                random_order: order == "random",
            },
            seed,
            valid_fraction,
            &out_prefix,
        ),
        Command::Style {
            model,
            n,
            steps,
            seed,
            guidance,
        } => {
            let f = safetensors::read(&model)?;
            let dit = Dit::from_file(&f).map_err(|e| format!("{}: {e}", model.display()))?;
            println!("{}", describe_checkpoint(&model, &f.metadata));
            style_table(&dit, n, steps, seed, guidance)
        }
    }
}

/// Stage 6's sampling: stories at each step count (timed one at a time,
/// as tiny-llm-in-rust times its generation), the order of settling, a
/// held-out story's middle infilled, and text files for tools/text_stats.py.
#[allow(clippy::too_many_arguments)]
fn sample_text(
    model: &Path,
    tokenizer: &Path,
    ids: &Path,
    steps: &[usize],
    n: usize,
    s: mini_diffusion::textmodel::Sampling,
    seed: u64,
    valid_fraction: f64,
    prefix: &str,
) -> Result<(), String> {
    use mini_diffusion::text::{split, Corpus, Vocab};
    use mini_diffusion::textmodel::{generate, TextModel};
    let f = safetensors::read(model)?;
    let m = TextModel::from_file(&f).map_err(|e| format!("{}: {e}", model.display()))?;
    let vocab = Vocab::load(tokenizer)?;
    let corpus = Corpus::load(ids)?;
    if vocab.size() != m.cfg.vocab || corpus.vocab_size != m.cfg.vocab {
        return Err(format!(
            "the model has V = {}, the tokenizer {} and the IDs {}",
            m.cfg.vocab,
            vocab.size(),
            corpus.vocab_size
        ));
    }
    let len = m.cfg.context;
    println!(
        "{}: trained for {} steps, diffusion bound {} nats per token",
        model.display(),
        f.metadata.get("steps").map(String::as_str).unwrap_or("?"),
        f.metadata.get("bound").map(String::as_str).unwrap_or("?")
    );
    // Each token's frequency rank in the training tokens (0 = commonest).
    let (train, valid) = split(&corpus.ids, valid_fraction);
    let mut counts = vec![0usize; m.cfg.vocab];
    for &t in train {
        counts[t as usize] += 1;
    }
    let mut order: Vec<usize> = (0..m.cfg.vocab).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
    let mut rank = vec![0usize; m.cfg.vocab];
    for (r, &t) in order.iter().enumerate() {
        rank[t] = r;
    }
    let top20: usize = order.iter().take(20).map(|&t| counts[t]).sum();
    println!(
        "The 20 commonest tokens are {:.1}% of the training tokens",
        100.0 * top20 as f64 / train.len().max(1) as f64
    );
    println!(
        "Each step keeps {} (temperature {}, top-k {})",
        if s.random_order {
            "positions chosen at random"
        } else {
            "the most confident tokens"
        },
        s.temperature,
        s.top_k
    );
    let median = |v: &mut Vec<usize>| {
        v.sort();
        if v.is_empty() {
            f64::NAN
        } else {
            v[v.len() / 2] as f64
        }
    };
    println!(
        "\n   steps   tokens/s   median rank of the tokens settled in the first tenth, last tenth"
    );
    let mut rng = Rng::new(seed);
    for &st in steps {
        let mut text = String::new();
        let (mut early, mut late) = (Vec::new(), Vec::new());
        let t0 = Instant::now();
        for _ in 0..n {
            let g = generate(&m, &vec![None; len], st, s, &mut rng);
            for (i, at) in g.settled_at.iter().enumerate() {
                if let Some(k) = at {
                    if *k < st.div_ceil(10) {
                        early.push(rank[g.tokens[i]]);
                    } else if *k >= st - st.div_ceil(10) {
                        late.push(rank[g.tokens[i]]);
                    }
                }
            }
            text.push_str(&vocab.decode(&g.tokens));
            text.push_str("\n\n");
        }
        let secs = t0.elapsed().as_secs_f64();
        let path = PathBuf::from(format!("{prefix}-s{st}.txt"));
        check_writable(&path)?;
        std::fs::write(&path, &text).map_err(|e| format!("can't write {}: {e}", path.display()))?;
        println!(
            "  {st:>6}   {:8.0}   {:>8.0}, {:>8.0}   ({})",
            (n * len) as f64 / secs,
            median(&mut early),
            median(&mut late),
            path.display()
        );
    }
    // Real held-out text, for the measures.
    let real: String = valid
        .chunks_exact(len)
        .take(64)
        .map(|w| vocab.decode(&w.iter().map(|&t| t as usize).collect::<Vec<_>>()) + "\n\n")
        .collect();
    let path = PathBuf::from(format!("{prefix}-real.txt"));
    check_writable(&path)?;
    std::fs::write(&path, &real).map_err(|e| format!("can't write {}: {e}", path.display()))?;
    println!("\nWrote {} (held-out text, for comparison)", path.display());
    // Infilling: the middle third of a held-out window masked.
    if let Some(w) = valid.chunks_exact(len).next() {
        let st = *steps.iter().max().unwrap_or(&64);
        let fixed: Vec<Option<usize>> = w
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if i >= len / 3 && i < 2 * len / 3 {
                    None
                } else {
                    Some(t as usize)
                }
            })
            .collect();
        let g = generate(&m, &fixed, st, s, &mut rng);
        let show = |toks: &[usize]| vocab.decode(toks);
        let orig: Vec<usize> = w.iter().map(|&t| t as usize).collect();
        let (a, b) = (len / 3, 2 * len / 3);
        let out = format!(
            "BEFORE THE GAP:\n{}\n\nTHE ORIGINAL MIDDLE:\n{}\n\nTHE MODEL'S MIDDLE ({st} steps):\n{}\n\nAFTER THE GAP:\n{}\n",
            show(&orig[..a]),
            show(&orig[a..b]),
            show(&g.tokens[a..b]),
            show(&orig[b..])
        );
        let path = PathBuf::from(format!("{prefix}-infill.txt"));
        check_writable(&path)?;
        std::fs::write(&path, &out).map_err(|e| format!("can't write {}: {e}", path.display()))?;
        println!(
            "Wrote {} (a held-out story's middle third, filled in)",
            path.display()
        );
    }
    Ok(())
}

/// The probes of stage 5 (PLAN.md), on n generations of `steps` Euler steps.
#[allow(clippy::too_many_arguments)]
fn inside_probes(
    dit: &Dit<f32>,
    clf: &Classifier<f32>,
    train: &mini_diffusion::image::Images,
    n: usize,
    steps: usize,
    seed: u64,
    preview: Option<PathBuf>,
    attention: Option<PathBuf>,
) -> Result<(), String> {
    use mini_diffusion::classify::best;
    use mini_diffusion::inside::*;
    let shape = dit.cfg.image;
    let l = dit.cfg.labels;
    let classes = 1 + *train.labels.iter().max().unwrap_or(&0) as usize;
    let (means, overall) = class_means(&train.pixels, &train.labels, classes);
    let ys: Vec<usize> = if l > 0 {
        (0..n).map(|i| i % l).collect()
    } else {
        Vec::new()
    };
    let what = if l > 0 {
        "each asked for a digit in turn, guidance w = 1"
    } else {
        "no labels"
    };
    println!("{n} generations of {steps} Euler steps ({what})");
    // Generate, keeping every step's velocity.
    let start = Instant::now();
    let mut vs: Vec<Mat<f32>> = Vec::with_capacity(steps);
    let x0 = noise(n, shape.values(), &mut Rng::new(seed));
    let (last, mut path) = euler(
        x0.clone(),
        steps,
        |x, t| {
            let v = if l > 0 {
                dit.predict_labels(x, t, &ys)
            } else {
                dit.predict(x, t)
            };
            vs.push(v.clone());
            v
        },
        true,
    );
    if path.is_empty() {
        path.push(x0);
    }
    println!("  generated in {}", duration(start.elapsed().as_secs_f64()));
    let tk = |k: usize| k as f32 / steps as f32;
    let guesses: Vec<Mat<f32>> = (0..steps).map(|k| guess(&path[k], &vs[k], tk(k))).collect();

    // Probe 1: the guess at t = 0 against the mean image.
    let g0 = &guesses[0];
    let err: f64 = (0..n)
        .map(|r| {
            let target = if l > 0 { &means[ys[r]] } else { &overall };
            relative_error(g0.row(r), target)
        })
        .sum::<f64>()
        / n as f64;
    let against = if l > 0 {
        "the mean image of the digit asked for"
    } else {
        "the data's mean image"
    };
    println!("\n1. The guess of the finished image at t = 0, against {against}:");
    println!(
        "   relative distance {:.1}% (averaged over the {n} generations)",
        100.0 * err
    );

    // Probe 2: the commitment curve.
    let digit = |m: &Mat<f32>| {
        best(&clf.probabilities(m))
            .iter()
            .map(|b| b.0)
            .collect::<Vec<_>>()
    };
    let finals = digit(&last);
    let marks: Vec<usize> = (0..=10).map(|i| (i * steps / 10).min(steps - 1)).collect();
    let per: Vec<Vec<usize>> = marks.iter().map(|&k| digit(&guesses[k])).collect();
    let curve = commitment(&per, &finals);
    println!("\n2. Generations whose guess is already their final digit, by t:");
    print!("   t       ");
    for &k in &marks {
        print!("{:>6.2}", tk(k));
    }
    print!("\n   share   ");
    for c in &curve {
        print!("{:>5.0}%", 100.0 * c);
    }
    println!();

    // Probe 3: guidance's push, for a model with labels.
    if l > 0 {
        let closed: f64 = ys.iter().map(|&c| dist(&means[c], &overall)).sum::<f64>() / n as f64;
        println!("\n3. How far guidance pushes: the length of v_cond - v_uncond, by t");
        println!("   at t = 0 it should equal the distance between the class mean and the overall mean: {closed:.2}");
        println!("       t   length   share of the t = 0 value   the two guesses apart");
        for &k in &marks {
            let vu = dit.predict(&path[k], tk(k));
            let diff = vs[k].zip_map(&vu, |a, b| a - b);
            let len = mean_norm(&diff);
            // The same difference in the guesses of the finished image:
            // (1 - t) times the velocities' difference.
            let apart = (1.0 - tk(k)) as f64 * len;
            println!(
                "    {:4.2}   {len:6.2}   {:22.1}%   {apart:6.2}",
                tk(k),
                100.0 * len / closed.max(1e-12)
            );
        }
    } else {
        println!("\n3. (guidance's push needs a model with labels)");
    }

    // Probe 4: straightness.
    let s = straightness(&path);
    let mean_s = s.iter().sum::<f64>() / n as f64;
    let min_s = s.iter().copied().fold(f64::INFINITY, f64::min);
    let bend = (0..n).map(|r| first_half_bend(&path, r)).sum::<f64>() / n as f64;
    println!("\n4. Straightness (straight-line distance over path length, 1 = straight):");
    println!(
        "   mean {mean_s:.3}, least {min_s:.3}; share of the bending in the first half {:.0}%",
        100.0 * bend
    );

    // Probe 5: attention distance, by t.
    let (gh, gw) = (shape.height / dit.cfg.patch, shape.width / dit.cfg.patch);
    let k_img = n.min(32);
    let few = |m: &Mat<f32>| Mat::from_vec(k_img, m.cols, m.data[..k_img * m.cols].to_vec());
    let few_ys: Vec<usize> = ys.iter().take(k_img).copied().collect();
    let att_marks = [0.1f32, 0.3, 0.5, 0.7, 0.9];
    let at_step = |t: f32| ((t * steps as f32).round() as usize).min(steps - 1);
    let mut maps = Vec::new();
    if gh == gw {
        let n_t = gh * gw;
        let even = {
            let e = Mat::from_vec(n_t, n_t, vec![1.0 / n_t as f32; n_t * n_t]);
            attention_distance(&e, gh)
        };
        println!(
            "\n5. Attention distance, in patches (even spread {even:.2}, own patch only 0), by t:"
        );
        print!("   block   ");
        for t in att_marks {
            print!("{t:>7.1}");
        }
        println!();
        let mut by_t = vec![Vec::new(); att_marks.len()];
        for (ti, &t) in att_marks.iter().enumerate() {
            let k = at_step(t);
            let (w, _) = dit.attention_weights(&few(&path[k]), tk(k), &few_ys);
            let mut per_block = Vec::new();
            for blk in &w {
                let mut s = 0.0;
                let mut cnt = 0;
                for img in blk {
                    for head in img {
                        s += attention_distance(head, gh);
                        cnt += 1;
                    }
                }
                per_block.push(s / cnt as f64);
            }
            by_t[ti] = per_block;
            maps.push(w);
        }
        for b in 0..dit.cfg.blocks {
            print!("   {:>5}   ", b + 1);
            for row in &by_t {
                print!("{:>7.2}", row[b]);
            }
            println!();
        }
        print!("   mean    ");
        for row in &by_t {
            print!("{:>7.2}", row.iter().sum::<f64>() / row.len() as f64);
        }
        println!();
    } else {
        println!("\n5. (attention distance needs a square grid of patches)");
    }

    // The figures.
    if let Some(p) = preview {
        let rows = 8.min(n);
        let mut data = Vec::new();
        for r in 0..rows {
            for pass in 0..2 {
                for j in 0..=8 {
                    let k = (j * steps / 8).min(steps - 1);
                    let img = if pass == 0 {
                        guesses[k].row(r).to_vec()
                    } else {
                        velocity_image(
                            &Mat::from_vec(1, shape.values(), vs[k].row(r).to_vec()),
                            0.5,
                        )
                        .data
                    };
                    data.extend(img);
                }
            }
        }
        check_writable(&p)?;
        write_grid(
            &p,
            &Mat::from_vec(rows * 18, shape.values(), data),
            shape,
            9,
            3,
        )?;
        println!(
            "\nWrote {} (for each generation: its guess of the finished image, then its velocity, grey 0, at t = 0, 1/8, ..., 1)",
            p.display()
        );
    }
    if let (Some(p), false) = (attention, maps.is_empty()) {
        let centre = (gh / 2) * gw + gw / 2;
        let ps = dit.cfg.patch;
        let mut data = Vec::new();
        for b in 0..dit.cfg.blocks {
            for w in &maps {
                let heads = &w[b][0];
                let mut avg = vec![0.0f32; gh * gw];
                for h in heads {
                    for (a, &x) in avg.iter_mut().zip(h.row(centre)) {
                        *a += x / heads.len() as f32;
                    }
                }
                let max = avg.iter().copied().fold(0.0f32, f32::max).max(1e-12);
                let mut img = vec![-1.0f32; shape.values()];
                for y in 0..shape.height {
                    for x in 0..shape.width {
                        let patch = (y / ps) * gw + x / ps;
                        let v = 2.0 * avg[patch] / max - 1.0;
                        for c in 0..shape.channels {
                            img[(y * shape.width + x) * shape.channels + c] = v;
                        }
                    }
                }
                data.extend(img);
            }
        }
        check_writable(&p)?;
        write_grid(
            &p,
            &Mat::from_vec(dit.cfg.blocks * maps.len(), shape.values(), data),
            shape,
            maps.len(),
            3,
        )?;
        println!(
            "Wrote {} (where the centre patch looks, averaged over heads, white = most: one row per block, t = 0.1, 0.3, 0.5, 0.7, 0.9)",
            p.display()
        );
    }
    Ok(())
}

/// Prediction 6: draw every class from the same `k` noises, and correlate
/// each style property between classes, with the noise shared and not.
fn style_table(dit: &Dit<f32>, k: usize, steps: usize, seed: u64, w: f32) -> Result<(), String> {
    use mini_diffusion::style::{same_and_different, styles};
    let l = dit.cfg.labels;
    if l < 2 {
        return Err("style needs a model trained with --conditional".into());
    }
    let shape = dit.cfg.image;
    let base = noise(k, shape.values(), &mut Rng::new(seed));
    let mut data = Vec::with_capacity(l * base.data.len());
    for _ in 0..l {
        data.extend_from_slice(&base.data);
    }
    let ys: Vec<usize> = (0..l).flat_map(|y| std::iter::repeat_n(y, k)).collect();
    let start = Instant::now();
    let (out, _) = euler(
        Mat::from_vec(l * k, shape.values(), data),
        steps,
        |x, t| dit.predict_guided(x, t, &ys, w),
        false,
    );
    println!(
        "{} images: {k} noises, each drawn as all {l} classes, guidance w = {w}, {steps} Euler steps, in {}",
        l * k,
        duration(start.elapsed().as_secs_f64())
    );
    let st = styles(&out, shape);
    println!("  correlation between two classes drawn from   the same noise   different noise");
    type Property = fn(&mini_diffusion::style::Style) -> f64;
    let props: [(&str, Property); 4] = [
        ("slant", |s| s.slant),
        ("centre, across", |s| s.cx),
        ("centre, down", |s| s.cy),
        ("ink (thickness)", |s| s.ink),
    ];
    for (name, f) in props {
        let v: Vec<f64> = st.iter().map(f).collect();
        let (same, diff) = same_and_different(&v, l, k);
        println!("    {name:<16}{:>42.2}{:>18.2}", same, diff);
    }
    Ok(())
}

/// The measurement table for a model with labels: for each guidance scale,
/// n images with every class requested equally often (the same noise at
/// every scale), and what the classifier says about them. Variety compares
/// each class's samples with as many real test images of that class.
fn measure_guided(
    dit: &Dit<f32>,
    clf: &Classifier<f32>,
    real: &mini_diffusion::image::Images,
    n: usize,
    steps: usize,
    seed: u64,
    guidance: &[f32],
) -> Result<(), String> {
    let l = dit.cfg.labels;
    let shape = dit.cfg.image;
    let ys: Vec<usize> = (0..n).map(|i| i % l).collect();
    let rows_of = |imgs: &Mat<f32>, pick: &dyn Fn(usize) -> bool| {
        let v = shape.values();
        let mut data = Vec::new();
        for i in (0..imgs.rows).filter(|&i| pick(i)) {
            data.extend_from_slice(imgs.row(i));
        }
        Mat::from_vec(data.len() / v, v, data)
    };
    let per = n / l;
    let real_by: Vec<Mat<f32>> = (0..l)
        .map(|c| {
            let idx: Vec<usize> = (0..real.labels.len())
                .filter(|&i| real.labels[i] as usize == c)
                .take(per.max(2))
                .collect();
            rows_of(&real.pixels, &|i| idx.contains(&i))
        })
        .collect();
    println!(
        "{n} images per guidance scale, {per} of each class, {steps} Euler steps; variety against {} real test images per class",
        per.max(2)
    );
    println!("      w   agreement   confident   variety   overshoot (margin 0.25)   time");
    for &w in guidance {
        let x0 = noise(n, shape.values(), &mut Rng::new(seed));
        let start = Instant::now();
        let (out, _) = euler(x0, steps, |x, t| dit.predict_guided(x, t, &ys, w), false);
        let secs = start.elapsed().as_secs_f64();
        let ms = measure(clf, &out, if w == 0.0 { None } else { Some(&ys) });
        let variety: f64 = (0..l)
            .map(|c| {
                let g = rows_of(&out, &|i| ys[i] == c);
                mini_diffusion::classify::variety(&g, &real_by[c])
            })
            .sum::<f64>()
            / l as f64;
        let over = ms
            .overshoot
            .iter()
            .find(|(m, _)| *m == 0.25)
            .map_or(0.0, |o| o.1);
        let agree = ms
            .agreement
            .map_or("   (none)".to_string(), |a| format!("{:8.1}%", 100.0 * a));
        let var = if w == 0.0 {
            "   (none)".to_string()
        } else {
            format!("{variety:9.2}")
        };
        println!(
            "  {w:5}   {agree}   {:8.1}%   {var}   {:22.2}%   {}",
            100.0 * ms.confident,
            100.0 * over,
            duration(secs)
        );
    }
    Ok(())
}

/// What a DiT checkpoint is: its data, the training step it was saved at
/// (and of how many), and the loss then.
fn describe_checkpoint(path: &Path, meta: &BTreeMap<String, String>) -> String {
    let get = |k: &str| meta.get(k).map(String::as_str).unwrap_or("?");
    let (done, total) = (get("steps"), get("total_steps"));
    let progress = if total == "?" {
        // Files written before total_steps was recorded.
        format!("saved at step {done}")
    } else if done == total {
        format!("trained for {done} steps (finished)")
    } else {
        format!("a checkpoint at step {done} of {total}")
    };
    format!(
        "Model {}: {} on {}, loss {}",
        path.display(),
        progress,
        get("data"),
        get("final_loss")
    )
}

/// Seconds as "1 h 05 min", "3 min 20 s" or "12 s".
fn duration(secs: f64) -> String {
    let s = secs.round() as u64;
    if s >= 3600 {
        format!("{} h {:02} min", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{} min {:02} s", s / 60, s % 60)
    } else {
        format!("{s} s")
    }
}

/// A trained model and what its file says about it.
struct Loaded {
    mlp: Mlp<f32>,
    objective: Objective,
    data: String,
}

fn load(path: &Path) -> Result<Loaded, String> {
    let f = safetensors::read(path)?;
    let get = |k: &str| {
        f.metadata
            .get(k)
            .cloned()
            .ok_or(format!("{}: no {k:?} in its metadata", path.display()))
    };
    Ok(Loaded {
        mlp: Mlp::from_file(&f).map_err(|e| format!("{}: {e}", path.display()))?,
        objective: Objective::parse(&get("objective")?)?,
        data: get("data")?,
    })
}

/// Generate from noise x0. Flow models take `steps` Euler steps; DDPM
/// models take DDIM's `steps`, or all of the ancestral sampler's.
fn generate(
    m: &Loaded,
    x0: Mat<f32>,
    steps: usize,
    sampler: &str,
    record: bool,
    rng: &mut Rng,
) -> Result<(Mat<f32>, Vec<Mat<f32>>), String> {
    match m.objective {
        Objective::Flow => Ok(euler(x0, steps, |x, t| m.mlp.predict(x, t), record)),
        Objective::Ddpm => {
            let s = Schedule::linear(ddpm::STEPS);
            let eps = |x: &Mat<f32>, i: usize| m.mlp.predict(x, s.time(i));
            match sampler {
                "ddim" => Ok((s.ddim(x0, steps.min(ddpm::STEPS), eps), Vec::new())),
                "ancestral" => Ok((s.ancestral(x0, eps, |r, c| noise(r, c, rng)), Vec::new())),
                _ => Err(format!("unknown sampler {sampler:?}: ddim or ancestral")),
            }
        }
    }
}

fn sampler_name(m: &Loaded, steps: usize, sampler: &str) -> String {
    match (m.objective, sampler) {
        (Objective::Flow, _) => format!("{steps} Euler steps"),
        (Objective::Ddpm, "ancestral") => format!("ancestral, {} steps", ddpm::STEPS),
        (Objective::Ddpm, _) => format!("DDIM, {} steps", steps.min(ddpm::STEPS)),
    }
}

/// The energy distance costs n² distances: 2,000 points at most.
fn cap(m: &Mat<f32>) -> Mat<f32> {
    let n = m.rows.min(2000);
    Mat::from_vec(n, m.cols, m.data[..n * m.cols].to_vec())
}

fn panel(title: &str, m: &Mat<f32>, color: &str) -> Panel {
    let mut p = Panel::new(title);
    p.points(m, color, 1.3, 0.5);
    p
}

fn check_writable(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err(format!(
            "can't write {}: no folder {}",
            path.display(),
            parent.display()
        ));
    }
    Ok(())
}

fn write_file(path: &Path, text: &str) -> Result<(), String> {
    check_writable(path)?;
    std::fs::write(path, text).map_err(|e| format!("can't write {}: {e}", path.display()))?;
    println!("Wrote {}", path.display());
    Ok(())
}

/// The credit printed with everything stage 7 makes (PLAN.md, Stage 7:
/// the terms of the models' use).
const DIT_CREDIT: &str = "DiT-XL/2: Peebles and Xie (2023), facebook/DiT-XL-2-256, CC BY-NC 4.0 (non-commercial use only); \
autoencoder: stabilityai/sd-vae-ft-ema, MIT";

fn missing_models(path: &Path) -> String {
    format!(
        "{}: not found. Stage 7's models are downloaded, under their own licences, by\n  \
         make dit-download DIT_LICENSE=accept\n(PLAN.md, Stage 7, sets out the terms)",
        path.display()
    )
}

/// DiT-XL/2 and its decoder, from the converted files.
fn load_dit(
    dit: &Path,
    vae: &Path,
) -> Result<(mini_diffusion::ditxl::DitXl, mini_diffusion::vae::Decoder), String> {
    use mini_diffusion::ditxl::{DitXl, DitXlConfig};
    use mini_diffusion::vae::{Decoder, VaeConfig};
    for p in [dit, vae] {
        if !p.exists() {
            return Err(missing_models(p));
        }
    }
    let t0 = Instant::now();
    let df = safetensors::read(dit)?;
    let cfg = DitXlConfig::from_json(
        df.metadata
            .get("config")
            .ok_or(format!("{}: no config in its metadata", dit.display()))?,
    )?;
    let model = DitXl::load(&df, cfg).map_err(|e| format!("{}: {e}", dit.display()))?;
    drop(df);
    let vf = safetensors::read(vae)?;
    let vcfg = VaeConfig::from_json(
        vf.metadata
            .get("config")
            .ok_or(format!("{}: no config in its metadata", vae.display()))?,
    )?;
    let dec = Decoder::load(&vf, vcfg).map_err(|e| format!("{}: {e}", vae.display()))?;
    println!(
        "Loaded the DiT ({} blocks, width {}, {} classes) and the decoder in {:.1} s",
        model.cfg.blocks,
        model.cfg.width,
        model.cfg.classes,
        t0.elapsed().as_secs_f64()
    );
    Ok((model, dec))
}

/// Classes by number or by name (a whole line of the names file, or the
/// first name on it, case ignored).
fn parse_classes(list: &str, names: &Path, classes: usize) -> Result<Vec<(usize, String)>, String> {
    let table: Vec<String> = std::fs::read_to_string(names)
        .map(|t| t.lines().map(str::to_string).collect())
        .unwrap_or_default();
    let name_of = |i: usize| {
        table
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("class {i}"))
    };
    list.split(',')
        .map(|item| {
            let item = item.trim();
            if let Ok(i) = item.parse::<usize>() {
                return if i < classes {
                    Ok((i, name_of(i)))
                } else {
                    Err(format!(
                        "class {i}: the model has {classes} classes (0 to {})",
                        classes - 1
                    ))
                };
            }
            let want = item.to_lowercase();
            table
                .iter()
                .position(|line| {
                    let l = line.to_lowercase();
                    l == want || l.split(',').next().map(str::trim) == Some(want.as_str())
                })
                .map(|i| (i, name_of(i)))
                .ok_or(format!("no class named {item:?} in {}", names.display()))
        })
        .collect()
}

/// Latents decoded to pixels, each image's rows of interleaved RGB.
fn decode_latents(
    dec: &mini_diffusion::vae::Decoder,
    latents: &[f32],
    c: usize,
    s: usize,
) -> Mat<f32> {
    use mini_diffusion::conv::Feature;
    let per = c * s * s;
    let n = latents.len() / per;
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for e in 0..n {
        let z: Vec<f32> = latents[e * per..(e + 1) * per]
            .iter()
            .map(|v| v / mini_diffusion::vae::SCALING)
            .collect();
        let img = dec.decode(&Feature::new(c, s, s, z));
        let hw = img.h * img.w;
        let mut px = vec![0.0; hw * 3];
        for p in 0..hw {
            for ch in 0..3 {
                px[p * 3 + ch] = img.data[ch * hw + p];
            }
        }
        rows.push(px);
    }
    let width = rows.first().map(Vec::len).unwrap_or(0);
    Mat::from_vec(n, width, rows.concat())
}

fn image_side(pixels: usize) -> usize {
    ((pixels / 3) as f64).sqrt().round() as usize
}

/// dit-sample's numbers: noises, guidance, channels guided (0: all),
/// steps, seed.
type SampleSettings = (usize, f32, usize, usize, u64);

/// A grid's legend: its file, the credit, the settings, and the rows
/// done so far, top to bottom.
fn dit_legend(
    output: &Path,
    settings: &str,
    wanted: &[(usize, String)],
    done: usize,
    noises: usize,
) -> String {
    let mut legend = format!(
        "{}\n{DIT_CREDIT}\n{settings}\n\
         rows, top to bottom (columns: {noises} starting noises, the same in every row):\n",
        output.display()
    );
    for (i, (cl, nm)) in wanted.iter().take(done).enumerate() {
        legend.push_str(&format!("  {}: {cl} {nm}\n", i + 1));
    }
    if done < wanted.len() {
        legend.push_str(&format!("  ({done} of {} rows done)\n", wanted.len()));
    }
    legend
}

#[allow(clippy::too_many_arguments)]
fn dit_sample(
    dit: &Path,
    vae: &Path,
    classes: &str,
    names: &Path,
    (noises, guidance, guide_channels, steps, seed): SampleSettings,
    reference: &str,
    same_as: Option<&Path>,
    output: &Path,
) -> Result<(), String> {
    use mini_diffusion::ditxl::{generate_guided, Reference};
    check_writable(output)?;
    let (model, dec) = load_dit(dit, vae)?;
    let wanted = parse_classes(classes, names, model.cfg.classes)?;
    let r = if reference == "diffusers" {
        Reference::Diffusers
    } else {
        Reference::Original
    };
    let (c, s) = (model.cfg.in_channels, model.cfg.size);
    let gc = if guide_channels == 0 {
        c
    } else {
        guide_channels.min(c)
    };
    let settings = format!(
        "guidance {guidance} on {gc} of {c} channels, {steps} DDIM steps, seed {seed}, {reference} arithmetic"
    );
    // A matching grid made earlier: the same settings and rows, all done.
    if let Some(prev) = same_as {
        let body = |t: &str| t.lines().skip(1).collect::<Vec<_>>().join("\n");
        let want = dit_legend(output, &settings, &wanted, wanted.len(), noises);
        if let Ok(old) = std::fs::read_to_string(prev.with_extension("txt")) {
            if body(&old) == body(&want) && prev.exists() {
                std::fs::copy(prev, output)
                    .map_err(|e| format!("can't copy {}: {e}", prev.display()))?;
                std::fs::write(output.with_extension("txt"), want)
                    .map_err(|e| format!("can't write the legend: {e}"))?;
                println!(
                    "{} has the same settings and rows: copied it to {}",
                    prev.display(),
                    output.display()
                );
                return Ok(());
            }
        }
    }
    let per = c * s * s;
    // The same noises for every class, so a column shows one noise asked
    // for each class.
    let mut rng = Rng::new(seed);
    let noise: Vec<f32> = (0..noises * per).map(|_| rng.normal() as f32).collect();
    println!("{DIT_CREDIT}");
    println!("{} classes × {noises} noises, {settings}", wanted.len());
    println!(
        "Grid: a row per class (the ImageNet label the model is asked for: number, then name and synonyms),\n\
         a column per starting noise (the same noise in every row); every cell a 256 × 256 photograph"
    );
    let mut images: Vec<Mat<f32>> = Vec::new();
    let rows = wanted.len();
    let start = Instant::now();
    let mut decode_secs = 0.0; // decoding time so far, for the estimate
    for (k, (class, name)) in wanted.iter().enumerate() {
        let t0 = Instant::now();
        let lat = generate_guided(
            &model,
            &noise,
            &vec![*class; noises],
            guidance,
            gc,
            steps,
            r,
            |step, _| {
                // Progress, updated in place: this row's step, the time a step
                // takes, and an estimate of what's left (the remaining steps
                // of every row, plus decoding at the rate seen so far).
                let done = step + 1;
                let per_step = t0.elapsed().as_secs_f64() / done as f64;
                let steps_left = (steps - done) + (rows - k - 1) * steps;
                let decode_left = if k > 0 {
                    decode_secs / k as f64 * (rows - k) as f64
                } else {
                    0.0
                };
                print!(
                "\r  row {} of {rows} ({name}): step {done} of {steps} ({} of {} in all, {:.0}%), {per_step:.1} s a step, about {} remaining   ",
                k + 1,
                k * steps + done,
                rows * steps,
                100.0 * (k * steps + done) as f64 / (rows * steps) as f64,
                duration(per_step * steps_left as f64 + decode_left)
            );
                let _ = std::io::Write::flush(&mut std::io::stdout());
            },
        );
        let gen_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        images.push(decode_latents(&dec, &lat, c, s));
        let dec_s = t1.elapsed().as_secs_f64();
        decode_secs += dec_s;
        // The grid so far and its legend, rewritten after every row, so a
        // long run can be looked at as it goes and a stopped one keeps
        // its finished rows.
        write_rows(output, &images, noises)?;
        let legend = dit_legend(output, &settings, &wanted, k + 1, noises);
        std::fs::write(output.with_extension("txt"), legend)
            .map_err(|e| format!("can't write the legend: {e}"))?;
        println!(
            "\r\x1b[2K  row {} of {rows}: {class} {name}: generated in {} ({:.2} s a step), decoded in {dec_s:.1} s",
            k + 1,
            duration(gen_s),
            gen_s / steps as f64,
        );
    }
    println!(
        "All {rows} rows in {}",
        duration(start.elapsed().as_secs_f64())
    );
    println!(
        "Wrote {} and its legend, {}",
        output.display(),
        output.with_extension("txt").display()
    );
    Ok(())
}

/// Rows of photographs (each a matrix of `noises` images) as one grid.
fn write_rows(output: &Path, images: &[Mat<f32>], noises: usize) -> Result<(), String> {
    let width = images[0].cols;
    let all = Mat::from_vec(
        images.len() * noises,
        width,
        images.iter().flat_map(|m| m.data.iter().copied()).collect(),
    );
    let side = image_side(width);
    let shape = mini_diffusion::image::Shape {
        height: side,
        width: side,
        channels: 3,
    };
    write_grid(output, &all, shape, noises, 1)
}

#[allow(clippy::too_many_arguments)]
fn dit_inside(
    dit: &Path,
    vae: &Path,
    class: &str,
    names: &Path,
    noises: usize,
    moments: usize,
    guidance: f32,
    steps: usize,
    seed: u64,
    output: &Path,
) -> Result<(), String> {
    use mini_diffusion::ditxl::{generate, Reference};
    check_writable(output)?;
    let (model, dec) = load_dit(dit, vae)?;
    let wanted = parse_classes(class, names, model.cfg.classes)?;
    let (cl, name) = wanted.first().ok_or("no class given")?.clone();
    let (c, s) = (model.cfg.in_channels, model.cfg.size);
    let per = c * s * s;
    let mut rng = Rng::new(seed);
    let noise: Vec<f32> = (0..noises * per).map(|_| rng.normal() as f32).collect();
    // The moments: steps evenly spread from the first to the last.
    let shown: Vec<usize> = (0..moments.max(1))
        .map(|m| {
            if moments <= 1 {
                steps - 1
            } else {
                m * (steps - 1) / (moments - 1)
            }
        })
        .collect();
    println!("{DIT_CREDIT}");
    println!("{cl} {name}: {noises} noises, guidance {guidance}, {steps} steps; the guess at steps {shown:?} (counting from 0)");
    let mut guesses: Vec<Vec<f32>> = Vec::new();
    let t0 = Instant::now();
    let lat = generate(
        &model,
        &noise,
        &vec![cl; noises],
        guidance,
        steps,
        Reference::Original,
        |k, x0| {
            if shown.contains(&k) {
                guesses.push(x0.to_vec());
            }
            let done = k + 1;
            let per_step = t0.elapsed().as_secs_f64() / done as f64;
            print!(
                "\r  step {done} of {steps}, {per_step:.1} s a step, about {} remaining   ",
                duration(per_step * (steps - done) as f64)
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        },
    );
    println!(
        "\r\x1b[2K  generated in {}; decoding {} photographs",
        duration(t0.elapsed().as_secs_f64()),
        noises * (shown.len() + 1),
    );
    // Rows are noises, columns moments then the finished photograph.
    let cols = guesses.len() + 1;
    let mut decoded: Vec<Mat<f32>> = guesses
        .iter()
        .map(|g| decode_latents(&dec, g, c, s))
        .collect();
    decoded.push(decode_latents(&dec, &lat, c, s));
    let width = decoded[0].cols;
    let mut data = Vec::new();
    for e in 0..noises {
        for d in &decoded {
            data.extend_from_slice(d.row(e));
        }
    }
    let side = image_side(width);
    let shape = mini_diffusion::image::Shape {
        height: side,
        width: side,
        channels: 3,
    };
    write_grid(
        output,
        &Mat::from_vec(noises * cols, width, data),
        shape,
        cols,
        1,
    )?;
    println!("Wrote {}", output.display());
    Ok(())
}

fn dit_check(dit: &Path, vae: &Path, dit_ref: &Path, vae_ref: &Path) -> Result<(), String> {
    use mini_diffusion::conv::Feature;
    use mini_diffusion::ditxl::{Reference, Weights};
    let (model, dec) = load_dit(dit, vae)?;
    for p in [dit_ref, vae_ref] {
        if !p.exists() {
            return Err(format!(
                "{}: not found: make dit-reference writes it",
                p.display()
            ));
        }
    }
    let worst = |got: &[f32], want: &[f32]| -> f32 {
        let scale = want.iter().fold(0f32, |a, b| a.max(b.abs())).max(1e-30);
        got.iter()
            .zip(want)
            .fold(0f32, |a, (g, w)| a.max((g - w).abs()))
            / scale
    };
    let rf = safetensors::read(dit_ref)?;
    let w = Weights::new(&rf);
    let latent = &w.get("input.latent")?.data;
    let times = w.get("input.times")?.data.clone();
    let labels: Vec<usize> = w
        .get("input.labels")?
        .data
        .iter()
        .map(|&v| v as usize)
        .collect();
    let r = Reference::Diffusers;
    println!("DiT-XL/2 against the reference, worst difference as a share of the largest value:");
    let t0 = Instant::now();
    let mut x = model.embed(latent, labels.len());
    println!(
        "  after the patch embedding  {:.1e}",
        worst(&x.data, &w.get("inter.patches")?.data)
    );
    let mut checked = 0;
    for b in 0..model.cfg.blocks {
        model.block(b, &mut x, &times, &labels, r);
        if let Some(t) = w.find(&format!("inter.block{b}")) {
            println!(
                "  after block {:<2}            {:.1e}",
                b + 1,
                worst(&x.data, &t.data)
            );
            checked += 1;
        }
    }
    let out = model.finish(&x, &times, &labels, r);
    let secs = t0.elapsed().as_secs_f64();
    let want = &w.get("output")?.data;
    let per = model.cfg.in_channels * model.cfg.size * model.cfg.size;
    let each = out.len() / labels.len();
    let eps_got: Vec<f32> = (0..labels.len())
        .flat_map(|e| out[e * each..e * each + per].to_vec())
        .collect();
    let eps_want: Vec<f32> = (0..labels.len())
        .flat_map(|e| want[e * each..e * each + per].to_vec())
        .collect();
    println!(
        "  ε                          {:.1e}",
        worst(&eps_got, &eps_want)
    );
    println!("  the whole output           {:.1e}", worst(&out, want));
    println!(
        "  one pass of {} examples: {secs:.2} s ({checked} blocks checked)",
        labels.len()
    );
    let vf = safetensors::read(vae_ref)?;
    let vw = Weights::new(&vf);
    let z = vw.get("input.latent")?;
    let side = image_side(z.data.len() * 3 / dec.cfg.latent);
    let t1 = Instant::now();
    let img = dec.decode(&Feature::new(dec.cfg.latent, side, side, z.data.clone()));
    let secs = t1.elapsed().as_secs_f64();
    println!(
        "The decoder: {} × {} image, worst difference {:.1e} of the largest value; {secs:.2} s",
        img.h,
        img.w,
        worst(&img.data, &vw.get("output")?.data)
    );
    Ok(())
}
