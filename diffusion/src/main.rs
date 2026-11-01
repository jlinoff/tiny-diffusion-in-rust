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
