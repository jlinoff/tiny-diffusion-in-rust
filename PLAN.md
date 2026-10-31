# Plan: building a diffusion model by hand

This project builds a small diffusion model piece by piece, in Rust, to
understand how image generation works: first on points in a plane, where
every step can be plotted, then on small images. It's the companion to
[tiny-llm-in-rust](https://github.com/jlinoff/tiny-llm-in-rust), which did
the same for a language model, and it starts from that project's
foundation: its matrices, automatic differentiation and file format,
copied here (see [Foundation](#foundation)).

The goal is understanding, not performance. Each stage is small enough to
read and runs on its own, and each arrives with its tests: this project
starts with `make test` and continuous integration in its first commit,
rather than adding them at the end (see [Design for test](#design-for-test)).

Technical terms are defined in [Terms](#terms), just below; tool and product
names are explained where they first appear.

**Contents** (generated from the headings by `make toc`; don't edit by hand)

<!-- toc -->
- [Terms](#terms)
  - [What a diffusion model does](#what-a-diffusion-model-does)
  - [Two ways to train one](#two-ways-to-train-one)
  - [Generating](#generating)
  - [The networks](#the-networks)
  - [Steering](#steering)
- [The flow](#the-flow)
- [Order of work](#order-of-work)
- [Foundation](#foundation)
- [Design for test](#design-for-test)
- [Open decisions](#open-decisions)
- [Where things live](#where-things-live)
- [Bibliography](#bibliography)
<!-- /toc -->

## Terms

Terms already defined in tiny-llm-in-rust's PLAN.md (matrix, weights,
gradient, loss, backpropagation, Adam, learning rate, batch, attention,
transformer block, residual connection, layer normalization, safetensors)
mean the same here and aren't repeated.

### What a diffusion model does

- **generative model**: a model that produces new examples that look like
  its training data: new images of digits, say, not a label for an
  existing one. A language model is one too; it generates by predicting
  one token at a time.
- **noise**: random numbers, here drawn from a **Gaussian** (normal)
  distribution: mean 0, variance 1, independently for every number. An
  image of pure noise is static.
- **diffusion model**: a generative model that starts from pure noise and
  turns it into an example by many small steps, each a little less noisy.
  It's trained on the reverse: real examples with noise added, which is
  easy to make and comes with its own answer key (the noise that was
  added). The name comes from physics: adding noise step by step is like
  ink diffusing in water, and the model learns to run it backwards
  (Sohl-Dickstein et al. 2015).
- **time**, **t**: how far along the path from noise to data an example is.
  This project uses the convention of flow matching: t = 0 is pure noise,
  t = 1 is data. (DDPM, below, counts the other way: its t = 0 is data.)
- **noise level**: how much noise an example carries; it's a function of t.

### Two ways to train one

- **flow matching**, **rectified flow**: draw a data example x₁, a noise
  sample x₀ and a time t between 0 and 1; mix them in a straight line,
  x_t = (1 − t)·x₀ + t·x₁; and train the model to predict the direction of
  that line, x₁ − x₀, from x_t and t. The loss is the mean squared error
  between prediction and target (Lipman et al. 2023; Liu et al. 2023).
- **velocity**, **velocity field**: that predicted direction. As a
  function of position and time it's a field, like a wind map: at every
  point and moment, which way to move and how fast.
- **mean squared error** (**MSE**): the average of the squared differences
  between prediction and target, the loss for predicting numbers rather
  than choosing a token.
- **DDPM** (denoising diffusion probabilistic models; Ho et al. 2020): the
  original practical recipe. Noise is added over many steps (typically
  1,000) on a fixed **noise schedule**; the model predicts the noise ε
  that was added, and generation removes a little of the predicted noise
  at each of those steps, adding a little fresh noise back each time. It
  works, but slowly; flow matching is the simpler descendant that current
  image models such as Stable Diffusion 3 use (Esser et al. 2024).

### Generating

- **ordinary differential equation** (**ODE**): an equation for how a
  quantity changes, here dx/dt = v(x, t): the position x moves along the
  velocity field. Generating with flow matching means solving it from
  t = 0 (noise) to t = 1 (a new example).
- **Euler step**: the simplest way to solve it: x ← x + v(x, t)·Δt, a short
  straight step along the current velocity. Twenty to fifty steps are
  typical; fewer give blurrier results, because the true path curves
  between steps.
- **sampling**: generating an example, in diffusion as in language models.
  Every starting noise gives a different example, the way every seed gave
  a different story.

### The networks

- **MLP** (multilayer perceptron): matrix products with a nonlinearity
  between them, nothing else. Big enough for points in a plane (stage 1).
- **time embedding**: t turned into a vector, by sines and cosines of t at
  several frequencies, so the network can tell nearby times apart. The
  same idea as tiny-llm-in-rust's position embeddings, applied to time.
- **patch**: a small square of an image, say 4 × 4 pixels. A 28 × 28 image
  is 49 patches; each becomes one token.
- **DiT** (diffusion transformer; Peebles and Xie 2023): a transformer whose
  tokens are an image's patches. Unlike a language model's, its attention
  isn't causal: every patch sees every other, since an image has no
  reading order. Its output is a velocity for every pixel.
- **adaptive layer normalization** (**adaLN**): how a DiT takes in the time
  (and later a class label): instead of learned constants, the scale and
  shift of each layer normalization are computed from the time embedding,
  so every block behaves a little differently at each noise level.

### Steering

- **conditioning**: giving the model extra information to generate from,
  such as which digit to draw.
- **classifier-free guidance** (Ho and Salimans 2022): train the model both
  with and without the condition (dropping it at random), then at
  generation compute both velocities and push further in the direction
  the condition adds: v = v_uncond + w·(v_cond − v_uncond), with a
  **guidance scale** w above 1. Larger w gives examples more clearly of
  the requested kind and less varied.

## The flow

Everything follows one pattern, the same at every stage:

```
training, repeated:                      generation:
  x₁ = a real example                      x = noise (x₀)
  x₀ = noise, t = random in [0, 1]         for t = 0, Δt, 2Δt, … < 1:
  x_t = (1 − t)·x₀ + t·x₁                      x = x + v(x, t)·Δt
  loss = MSE(v(x_t, t), x₁ − x₀)           x is a new example
  adjust the weights (Adam)
```

Training never runs the ODE: it only ever sees single points on straight
lines between noise and data, and learns the average direction through
each point. Those averages are a velocity field whose paths carry the
whole noise distribution onto the whole data distribution, which is the
fact flow matching rests on (Lipman et al. 2023). Generation then follows
those paths, one Euler step at a time. Compared with a language model:
generation is again repeated passes through the network, but each pass
revises the whole example at once instead of adding one token, and the
number of passes is chosen (the steps), not set by the length.

## Order of work

1. **Foundation** (done): the matrices, tape, BLAS support, random numbers
   and safetensors files, copied from tiny-llm-in-rust with their tests;
   `make test` and continuous integration from the first commit.
2. **Points in a plane.** Flow matching with a small MLP on 2D data (two
   interleaved half-moons, a spiral, a ring of Gaussians), each example two
   numbers. Training takes seconds; every step can be plotted as SVG (a
   text format for drawings that browsers display): the data, the learned
   velocity field as arrows, sample paths from noise to data, and how the
   result degrades with fewer Euler steps. Then DDPM on the same data, for
   comparison: what the 1,000 steps buy and cost.
3. **Small images.** A DiT on 28 × 28 grayscale digits (see
   [Open decisions](#open-decisions) for the data): patches, attention
   without the causal mask, adaLN conditioning on time, a velocity per
   pixel. Grids of generated digits as image files, and the same picture
   at successive steps of one generation, noise sharpening into a digit.
4. **Conditioning and guidance.** The digit's class as a condition, and
   classifier-free guidance: "draw a 7", and what the guidance scale
   trades.
5. **Looking inside** (optional): what the velocity field looks like at
   each t, which patches attend to which, and how early in generation the
   digit's identity is settled.
6. **Text, later** (optional): a masked diffusion language model on
   TinyStories, reusing tiny-llm-in-rust's tokenizer. Its noise is masking
   rather than Gaussian: the forward process hides tokens at random, and
   the model learns to fill them all in at once (Sahoo et al. 2024; Nie et
   al. 2025). The comparison with tiny-llm-in-rust's autoregressive model
   at the same size is the point.

Each stage ends the way tiny-llm-in-rust's did: predictions written down
first, then the measurements, including the ones that turn out wrong.

## Foundation

Copied from tiny-llm-in-rust's `transformer/src/` (each file says so at the
top), and checked by its own tests, which came along:

| File | What it holds |
|---|---|
| `tensor.rs` | row-major matrices, generic over f32 and f64; products split across threads, each element summed in a fixed order |
| `tape.rs` | reverse-mode automatic differentiation on whole matrices, every operation checked against finite differences |
| `blas.rs` | optional Apple Accelerate for the large f32 products (`--features blas`, the default on macOS) |
| `rng.rs` | SplitMix64, with independent streams and normal draws |
| `safetensors.rs` | reading and writing model files, checked against one Hugging Face wrote |

Copying rather than depending on tiny-llm-in-rust keeps each project
standalone and free to change: diffusion needs a few things the language
model didn't, and they'll be added here:

- attention without the causal mask (stage 3), checked by **permutation
  equivariance**: without position embeddings, shuffling the input tokens
  must shuffle the outputs the same way and change nothing else;
- the mean squared error as a fused loss operation;
- element-wise products and per-example scale and shift, for adaLN;
- SiLU (x·sigmoid(x)), the nonlinearity DiT uses alongside GELU.

The rest of tiny-llm-in-rust (tokenizer, causal model, generation with a KV
cache) has no counterpart here: a diffusion model has no tokens to predict
in order, so nothing to cache between passes.

## Design for test

tiny-llm-in-rust added its test harness after its first release; its
experiments had been checked by eye, and the checks became tests only
when that work stopped. Here every stage arrives with its tests, and
`make test` runs them all, as continuous integration does on every push.
The properties that make a project testable are built in from the start:
fixed seeds, results independent of the thread count, and exact resuming.

What each stage will test, beyond the gradient checks every new operation
gets:

- **flow matching against a closed form.** If the data is itself Gaussian,
  say one number from N(μ, s²) with noise N(0, 1), the ideal velocity has
  a formula: E[x₁ − x₀ | x_t] = μ + c(t)·(x_t − tμ), with c(t) =
  (t·s² − (1 − t)) / ((1 − t)² + t²·s²). Integrating it carries N(0, 1)
  exactly onto N(μ, s²). So the sampler can be tested with the true field
  before any network is trained (the samples' mean and spread must come
  out μ and s), and a trained network can be compared with the truth.
- **the interpolation and its target**: x_t at t = 0 is the noise and at
  t = 1 the data; the target is the same for every t on one line.
- **DDPM's forward process**: noised data has exactly the mean and variance
  the schedule says, measured on many draws.
- **patches**: cutting an image into patches and reassembling them gives
  back the same image, bit for bit.
- **non-causal attention**: permutation equivariance (above), and a
  gradient check.
- **a golden loss**, as in tiny-llm-in-rust: a fixed seed and model must
  reproduce recorded losses, so a change that quietly alters training
  fails a test.
- **generation is reproducible**: the same seed gives the same samples,
  whatever the number of threads.
- **files round-trip**: the image and plot writers produce what a reader
  gets back.

## Open decisions

1. **The image data for stage 3.** MNIST (LeCun et al. 1998: 70,000
   handwritten digits, 28 × 28, about 11 MB compressed) is the classic,
   and digits make failures easy to see. Fashion-MNIST (Xiao et al. 2017:
   the same format, clothing instead of digits) is MIT-licensed, which
   settles redistribution questions MNIST's terms leave vague. Either is
   downloaded by the Makefile, never committed.
2. **How to write images.** PNG is what every viewer opens. A minimal PNG
   writer needs no library: uncompressed ("stored") deflate blocks, with
   the CRC-32 and Adler-32 checksums the format requires, a few dozen
   lines. The simpler alternative is PGM, a plain grayscale format, which
   macOS Preview opens but GitHub doesn't display.
3. **The DiT's size.** A first guess, to be measured: width 128, 6 blocks,
   4 heads, 4 × 4 patches (49 tokens), about 1.2 million parameters. At
   roughly 0.4 GFLOP per example per training step, a batch of 128 costs
   about 50 GFLOP; at the tens of GFLOP per second tiny-llm-in-rust's code
   reached on the M4 with Accelerate, that's around a second a step, and
   20,000 steps a few hours. A prediction, recorded so the measurement can
   correct it.
4. **DDPM's sampler.** Its own 1,000-step ancestral sampler, or DDIM's
   deterministic shortcut (Song et al. 2021), which takes 50 steps with
   the same trained model. Both, probably, since the comparison is the
   lesson.

## Where things live

```
tiny-diffusion-in-rust/
├── README.md      # a short introduction, for GitHub
├── PLAN.md        # this file
├── LICENSE        # MIT
├── Makefile       # make test runs every check; one target per step as stages arrive
├── diffusion/     # the Rust crate: the library and the `diffusion` command
│   ├── src/       # tensor, tape, blas, rng, safetensors (the foundation), main.rs
│   └── tests/data # test fixtures
├── tools/         # toc.py: this file's table of contents and link check
├── .github/       # workflows/main.yaml: formatting, lint and tests, on Linux and macOS
├── data/          # downloaded datasets (git-ignored)
└── out/           # generated files: models, plots, images (git-ignored)
```

## Bibliography

- Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015).
  "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." ICML.
  [arXiv:1503.03585](https://arxiv.org/abs/1503.03585). The original idea,
  from statistical physics.
- Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic
  Models." NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).
  DDPM: the noise-prediction objective that made diffusion practical.
- Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit
  Models." ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502). DDIM:
  fewer, deterministic sampling steps from a DDPM-trained model.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).
  "Flow Matching for Generative Modeling." ICLR.
  [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
- Liu, X., Gong, C., & Liu, Q. (2023). "Flow Straight and Fast: Learning to
  Generate and Transfer Data with Rectified Flow." ICLR.
  [arXiv:2209.03003](https://arxiv.org/abs/2209.03003).
- Peebles, W., & Xie, S. (2023). "Scalable Diffusion Models with
  Transformers." ICCV. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748).
  DiT, with adaLN conditioning.
- Ho, J., & Salimans, T. (2022). "Classifier-Free Diffusion Guidance."
  [arXiv:2207.12598](https://arxiv.org/abs/2207.12598).
- Esser, P., et al. (2024). "Scaling Rectified Flow Transformers for
  High-Resolution Image Synthesis." ICML.
  [arXiv:2403.03206](https://arxiv.org/abs/2403.03206). Stable Diffusion 3:
  flow matching and a transformer, at scale.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-Based
  Learning Applied to Document Recognition." Proceedings of the IEEE. The
  MNIST digits.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). "Fashion-MNIST: a Novel Image
  Dataset for Benchmarking Machine Learning Algorithms."
  [arXiv:1708.07747](https://arxiv.org/abs/1708.07747).
- Sahoo, S. S., et al. (2024). "Simple and Effective Masked Diffusion
  Language Models." NeurIPS.
  [arXiv:2406.07524](https://arxiv.org/abs/2406.07524).
- Nie, S., et al. (2025). "Large Language Diffusion Models."
  [arXiv:2502.09992](https://arxiv.org/abs/2502.09992). LLaDA.
