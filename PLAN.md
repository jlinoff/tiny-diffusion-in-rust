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
- [The flows](#the-flows)
  - [Training](#training)
  - [Generation (inference)](#generation-inference)
- [Order of work](#order-of-work)
- [Foundation](#foundation)
- [Design for test](#design-for-test)
- [How to read the figures](#how-to-read-the-figures)
- [Stage 2: points in a plane](#stage-2-points-in-a-plane)
  - [What was built](#what-was-built)
  - [Predictions](#predictions)
  - [Exact fields, no network](#exact-fields-no-network)
  - [Results](#results)
- [Stage 3: images](#stage-3-images)
  - [What was built](#what-was-built-1)
  - [Choosing the patch size](#choosing-the-patch-size)
  - [Predictions](#predictions-1)
  - [Results so far](#results-so-far)
- [Text, later](#text-later)
  - [How text diffusion works](#how-text-diffusion-works)
  - [Compared with tiny-llm-in-rust's model](#compared-with-tiny-llm-in-rusts-model)
- [Decisions](#decisions)
- [Where things live](#where-things-live)
- [Bibliography](#bibliography)
<!-- /toc -->

## Terms

Terms already defined in tiny-llm-in-rust's PLAN.md mean the same here
and aren't repeated: weights, tensor, gradient, loss, cross-entropy, backpropagation, automatic differentiation (the tape),
Adam, learning rate, batch, attention, GELU (the Gaussian error linear
unit, a smooth nonlinearity), transformer block, residual connection,
layer normalization, KV cache (key–value cache), checkpoint,
safetensors, and f32 and f64 (32- and 64-bit floating-point numbers).
Two more, used here: **BLAS** (Basic Linear Algebra
Subprograms, the standard interface for fast matrix products, here
Apple's Accelerate library on macOS) and **GFLOP** (a billion
floating-point operations; GFLOP/s, a billion per second).

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
- **DDIM** (denoising diffusion implicit models; Song et al. 2021): a faster
  way to generate from a DDPM-trained model, with no retraining. It visits
  a subsequence of the steps (say 50 of the 1,000), each jump estimating
  the clean example and re-noising it to the next level, and adds no
  fresh noise.

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

## The flows

There are two flows, as in tiny-llm-in-rust: training, which adjusts the
weights, and generation (inference), which uses them frozen. Both are the
same at every stage, whatever the data (points, images, later text).
"Step" means something different in each:

- a **training step** is one weight update: a batch of examples, a loss,
  a backward pass, Adam. Training takes thousands of them.
- a **sampling step** is one Euler step of generation: one pass through
  the network, with the weights frozen, moving the example a little way
  from noise towards data. Generation takes tens of them per example.

### Training

```
repeat, one training step at a time:
  x₁ = a batch of real examples
  x₀ = noise of the same shape; t = random in [0, 1], one per example
  x_t = (1 − t)·x₀ + t·x₁                 a point on each straight line
  loss = MSE(v(x_t, t), x₁ − x₀)          predicted velocity against the line's direction
  backward pass, then Adam adjusts every weight
```

Training never runs the ODE, and never takes a sampling step. Each
example is a single point on a straight line between some noise and some
data, and the model learns the direction of the line through that point.
Many different lines pass near any point, so what it learns is their
average direction there. Those averages make a velocity field whose
paths carry the whole noise distribution onto the whole data
distribution: the fact flow matching rests on (Lipman et al. 2023). The
model learns what to do at every moment of the journey without ever
making the journey.

Compared with tiny-llm-in-rust's training: there, one step scored the
next token at every position of a window; here, one step scores a
velocity for every example at its own random moment t. Both are
self-supervised, the answers coming free with the data: the next token
there, the noise that was added here.

### Generation (inference)

```
x = noise (x₀), the shape of one example
for t = 0, Δt, 2Δt, … < 1:                one sampling step each
    x = x + v(x, t)·Δt                     a short straight step along the velocity
x is a new example
```

Only generation runs the whole path, one Euler step at a time, with the
weights frozen. The number of steps is a choice: more follow the curved
paths more faithfully, fewer are faster and blurrier. Each new starting
noise gives a different example, as each seed gave a different story.

Compared with tiny-llm-in-rust's generation: both are repeated passes
through the network with the weights frozen, but there each pass added
one token and the length set the number of passes; here each pass
revises the whole example at once, and the number of passes is chosen.
There's also nothing to cache between passes: a KV cache kept earlier
tokens' keys and values because they never changed, but every part of
the example changes at every sampling step.

## Order of work

1. **Foundation** (done): the matrices, tape, BLAS support, random numbers
   and safetensors files, copied from tiny-llm-in-rust with their tests;
   `make test` and continuous integration from the first commit.
2. **Points in a plane** (done: see [Stage 2](#stage-2-points-in-a-plane)).
   Flow matching with a small MLP on 2D data (two
   interleaved half-moons, a spiral, a ring of Gaussians), each example two
   numbers. Training takes seconds; every step can be plotted as SVG (a
   text format for drawings that browsers display): the data, the learned
   velocity field as arrows, sample paths from noise to data, and how the
   result degrades with fewer Euler steps. Then DDPM on the same data, for
   comparison: what the 1,000 steps buy and cost.
3. **Small images** (built: see [Stage 3](#stage-3-images)). A DiT on
   28 × 28 grayscale digits (see [Decisions](#decisions) for the data): patches, attention
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
   TinyStories (tiny-llm-in-rust's corpus of short, simple stories), reusing tiny-llm-in-rust's tokenizer, compared with
   tiny-llm-in-rust's model at the same size (see [Text, later](#text-later)).

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
| `rng.rs` | SplitMix64 (a small, fast random number generator), with independent streams and normal draws |
| `safetensors.rs` | reading and writing model files, checked against one Hugging Face wrote |

Copying rather than depending on tiny-llm-in-rust keeps each project
standalone and free to change: diffusion needs a few things the language
model didn't, and they'll be added here:

- attention without the causal mask (stage 3), checked by **permutation
  equivariance**: without position embeddings, shuffling the input tokens
  must shuffle the outputs the same way and change nothing else;
- the mean squared error as a fused loss operation (added in stage 2);
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
- **the documents themselves**: PLAN.md's table of contents and links, and
  every acronym in PLAN.md and README.md defined where it's first used
  (`tools/acronyms.py`; names that need no definition are listed in
  `tools/acronyms-known.txt`).

## How to read the figures

Every stage produces figures, and they're built to be read the same way.
This section is for a reader new to diffusion; it uses stage 2's figures
(`make stage2`, then open `out/*.svg` in a browser) as the examples.

**The picture to keep in mind: a wind map.** The trained network is a
wind map that changes with time: at every point and every moment t, which
way to blow and how hard (the velocity field). Generation releases leaves
at random spots (the noise) and lets the wind carry them from t = 0 to
t = 1; where they land is the result. The number of sampling steps is how
often each leaf checks the wind: with few steps, a leaf flies straight for
long stretches and misses the curves.

**Samples panels** (`generated`, and each panel of `-steps.svg`):

- **each dot is one generated example**, which began as one random point;
- **in a steps figure, every panel uses the same starting points and the
  same trained network**: only the number of steps changes, so every
  difference between panels comes from how coarsely the path was
  followed, not from the network;
- **what to look at**: the *shape* (are both moons there, all eight
  blobs?), the *sharpness* (thin, crisp arcs like the data's, or fuzzy?),
  *stragglers* (points stranded between the pieces: leaves that ran out of
  steps before arriving), and the *spread* (too wide is overshooting, too
  narrow is curves cut short);
- **the number in each title** sums all of that up: the energy distance to
  the real data (defined in [Stage 2](#stage-2-points-in-a-plane)), 0 when
  indistinguishable. Two samples of the data itself score about 0, the
  best any model can do.

**The paths panel**: each orange line is one leaf's whole trip, from its
grey starting point to its red landing point. Straight lines mean few
steps suffice, since a straight path needs no corrections; curved ones
need more.

**The velocity-field panels** show the network itself, the wind map at
three moments. At t = 0 every arrow points towards the middle: a point
that is pure noise says nothing about where its path leads, so the best
guess is the data's average. By t = 0.5 the arrows point towards the
nearest part of the data, and by t = 0.9 they pull hard onto it. That
progression is the "decision" being made over time: early steps settle
roughly where a point is going, late steps settle exactly where.

**Image grids** (stage 3: `sample-img`'s PNG files):

- **the samples grid** (`--png`, e.g. `out/digits.png`): 64 generated
  images, 8 across, each from its own random noise. The grid is only a
  way to see many at once; the images are unrelated to one another.
- **the progress grid** (`--progress`, e.g. `out/digits-progress.png`):
  the making of the first 8 of those images. Each *row* is one
  generation, and its 9 *columns* are snapshots at t = 0, 1/8, 2/8 … 1:
  the first column is the starting noise, the last the finished image.
  So the last column, read top to bottom, is the samples grid's first
  row, read left to right.
- **the real grid** (`--real`): 64 images from the training data, to
  compare with.
- **the same seed gives the same starting noise** (seed 2 by default), so
  samples from two checkpoints of one run, or from two models of the same
  image size, start from identical noise: each position in the grid shows
  how the same noise is turned into an image, and the difference between
  the grids is the training alone.

**The habit that makes them useful: ask what changed between two panels,
and what didn't.** The figures change one thing at a time (the steps, the
sampler, the training), so each difference can be traced to its cause.

**An example: the largest change is from 1 step to 2.** In
`moons-ddpm-steps.svg` the score falls from 6.7 to 0.19 between the first
two panels, a factor of 35, while every later doubling of the steps gains
a factor of 2 to 5. The cause is in the arithmetic of
DDIM's jump: each one estimates the clean point by dividing by √ᾱ at the
current step, so it multiplies any error in the network's noise
prediction by √(1 − ᾱ)/√ᾱ. At the start of generation, ᾱ₁₀₀₀ ≈ 0.00004,
so that factor is about 157:

- **with 1 step**, the whole journey is one blind jump: a 157-fold error,
  and no chance to correct it. The points land further from the data than
  the noise they started as;
- **with 2 steps**, the first jump goes only halfway, to step 500, where
  the error is scaled back down to about 43-fold, and then a second step
  looks again: the network sees where the first jump landed, predicts the
  noise from there, and finishes from much nearer the truth.

The first extra look matters most because it's the first chance to
correct anything; after that, more steps refine an answer that's already
roughly right. Flow's figure shows the same shape less extremely (0.81 to
0.20): its single step lands everything on the mean, a bad but not wild
answer, since flow's velocity has no division by a small number.

## Stage 2: points in a plane

### What was built

- `data2d.rs`: the four data sets (moons, spiral, rings, gaussian),
  unlimited and reproducible, and Gaussian noise.
- `flow.rs`: the interpolation, the target, the Euler sampler, and the
  exact velocity field for Gaussian data.
- `ddpm.rs`: DDPM's schedule, its forward process, its ancestral sampler,
  DDIM's sampler, and the exact noise prediction for Gaussian data.
- `mlp.rs`: the network, with the time embedding: 35,842 parameters at
  the defaults (width 128, 3 hidden layers, 8 sine-cosine pairs).
- `optim.rs` (Adam), `train2d.rs` (training), `svg.rs` (plots),
  `metrics.rs` (the energy distance, below), and the tape's new fused
  mean-squared-error operation.
- Commands: `diffusion train2d`, `sample2d`, `steps2d` and `exact`, and
  `make` targets for each data set.

**Measuring a result: the energy distance.** For generated points X and
data points Y it's 2·mean|x − y| − mean|x − x′| − mean|y − y′|, over all
pairs (Székely and Rizzo 2013): zero only when the two distributions are
the same, and growing as they differ in location, spread or shape. Two
independent samples of the same data set give its floor, a little above
zero; a result near the floor is as good as this measure can tell.

**Why the loss doesn't fall to zero.** Many straight lines pass through
any point, each with its own direction x₁ − x₀; the network can only
predict their average, and the spread of the directions around it is
loss nothing can remove. So a flow model's loss levels off well above
zero even when its velocity field is right, unlike a language model's
loss, whose floor is the text's own unpredictability. The level says
little on its own; the samples are the measure.

### Predictions

Written before the samples were measured (the exact-field table below,
which needs no training, came first and shaped the third):

1. **One Euler step puts every sample at the data's mean.** At t = 0 a
   point says nothing about where its line leads, so the ideal velocity is
   the mean minus the point (a test checks this for the exact Gaussian
   field). For moons, centered at the origin, one step should give a tight
   cluster there, not two moons.
2. **The distance to the data falls with the number of steps and levels
   off** near the floor by 32 to 64 Euler steps.
3. **With few steps, samples come out too concentrated**, pulled in from
   the true shape, as the exact field's Euler steps are.
4. **DDIM needs about twice as many steps as flow's Euler steps** for the
   same quality, as with the exact fields; DDPM's ancestral sampler, with
   all 1,000 steps, is as good as either at its best.
5. **Every blob of rings is covered**: nothing in the training makes a
   model drop a blob.

### Exact fields, no network

`diffusion exact`: Gaussian data (standard deviations 0.5 and 0.8), the
exact velocity and noise predictions, and only the samplers' own error.
The samples' standard deviations:

| Steps | Flow, Euler | DDPM, DDIM |
|---|---|---|
| 1 | 0.000, 0.000 | 0.002, 0.004 |
| 2 | 0.201, 0.389 | 0.072, 0.177 |
| 5 | 0.371, 0.612 | 0.264, 0.512 |
| 10 | 0.433, 0.699 | 0.372, 0.645 |
| 25 | 0.474, 0.757 | 0.447, 0.733 |
| 50 | 0.488, 0.777 | 0.474, 0.764 |
| 100 | 0.495, 0.787 | 0.488, 0.781 |
| 250 | 0.499, 0.793 | 0.497, 0.790 |
| 1000 | 0.502, 0.796 | 0.501, 0.795 |

Both samplers start too narrow and converge as the steps increase: the
paths curve, and a straight step cuts the corner inwards. DDIM's error at
any step count is about that of flow's Euler at half as many: flow at 25
steps (0.474) matches DDIM at 50. One step collapses everything onto the
mean, for both, exactly as the formula says. All of this is the sampler
alone, before any network adds errors of its own.

### Results

Trained with the defaults (5,000 steps of 256 points, learning rate
0.002, 35,842 parameters): about 46 seconds each in this project's cloud
workspace (2 cores, about 110 steps per second). `make stage2` reproduces
everything; the figures are `out/*.svg`, and the two for moons are in
`docs/`.

![Moons: data, samples, paths and the learned velocity field](docs/moons-flow.svg)

**Moons, by sampling steps** (`diffusion steps2d`; energy distance to the
data, 2,000 points; two samples of the data itself give −0.0013, an
estimate that can dip just below zero):

| Steps | Flow, Euler | DDPM, DDIM |
|---|---|---|
| 1 | 0.812 | 6.707 |
| 2 | 0.196 | 0.188 |
| 4 | 0.050 | 0.086 |
| 8 | 0.012 | 0.015 |
| 16 | 0.0047 | 0.0049 |
| 32 | 0.0024 | 0.0031 |
| 64 | 0.0013 | 0.0028 |
| 128 | 0.0009 | 0.0026 |
| 1,000 (DDPM's ancestral) | | 0.0014 |

![Moons by number of Euler steps](docs/moons-flow-steps.svg)

Rings and spiral, by flow at 50 steps: −0.0003 and −0.0001, both at their
data's floor.

**The predictions, checked:**

1. **One step puts every sample at the mean: right.** A tight cluster at
   the origin (the figure's first panel), and the velocity field at t = 0
   points from everywhere towards the middle, as the formula says.
2. **The distance falls and levels off by 32 to 64 steps: right.** 0.0024
   at 32, 0.0013 at 64, near the floor.
3. **Few steps give samples that are too concentrated: half right.** At 4
   and 8 steps each moon is a thin, sharp curve, narrower than the data's,
   as predicted; but the larger error is something else: stragglers
   scattered across the gap between the moons, points whose paths were
   cut short before they reached either one. The exact Gaussian, with
   only one blob, had no gap to strand them in.
4. **DDIM needs twice flow's steps: wrong.** From 2 to 16 steps the two
   are about equal. DDIM differs at the ends instead: one step explodes
   (6.7, points scattered far outside the window), and from 32 steps on
   it levels off higher (about 0.003 against flow's 0.0009). The
   explosion has a clear cause: DDIM's single jump from step 1,000
   estimates the clean point by dividing by √ᾱ₁₀₀₀ ≈ 0.006, which
   multiplies any error in the network's noise prediction by about 150.
   With the exact prediction there's no error to multiply, and it
   collapsed onto the mean like flow. The higher plateau is likely the
   same effect in milder form, the noise prediction's errors at high
   noise amplified, but that's a guess, not a measurement. DDPM's
   ancestral sampler, with all 1,000 steps, beats DDIM's best (0.0014),
   but not flow's at 64 steps or more, with a twentieth of the passes.
5. **Every blob of rings is covered: right.** All eight, with a few
   stragglers along the circle between them.

**What the pictures add.** Flow's paths are nearly straight lines from
noise to data (the figure's third panel), which is why so few steps
suffice: a straight path is followed exactly by a single Euler step, and
it's the bends that need more. The velocity field shows when the decision
is made: at t = 0 every arrow points towards the middle, at t = 0.5 the
arrows already point towards the nearest part of the data, and by
t = 0.9 they pull hard onto it, with the field between the moons (or
rings' blobs) splitting points to one side or the other.

**The losses** level off at 1.51 (moons), 1.82 (rings) and 1.45 (spiral)
for flow, and 0.23 for DDPM on moons, which predicts a different target
(the noise, not the velocity). As above, these floors are the spread of
the targets, not the model's error, and they don't compare across data
sets or objectives.

## Stage 3: images

### What was built

- **The tape's new operations**: attention without the causal mask
  (checked against finite differences, and for **permutation
  equivariance**: with no position information, shuffling the tokens
  shuffles the outputs and changes nothing else, which is why the DiT
  needs position embeddings), SiLU, column slices, and adaLN's per-image
  scale, shift and gate.
- `image.rs`: patches and back, bit for bit, at any size; the IDX reader
  for MNIST and Fashion-MNIST (IDX: the files' simple format, a header of
  dimensions, then one byte per pixel); a synthetic data set of shapes
  (squares, discs and crosses) at any size and channel count; PNG
  (Portable Network Graphics) files and grids of images, with the
  format's checksums, CRC-32 (a cyclic redundancy check) and Adler-32,
  checked against their standard test values.
- `dit.rs`: the DiT of [Terms](#terms), with **adaLN-Zero**: the layers
  computing each block's scale, shift and gate start at zero, so every
  gate starts closed, every block starts as the identity, and the
  untrained model predicts exactly 0 (a test checks it). Its gradients
  are checked against finite differences, whole.
- `trainimg.rs` and the commands `diffusion train-img` and `sample-img`:
  flow matching on images, checkpoints every 1,000 steps (so a long run
  can be sampled while it continues), grids of samples, real images for
  comparison, and the progress of single generations from noise to image.
- `make digits` and `make fashion` download the data and train; `make
  shapes` needs no download.
- **Watching a run as it trains**: `make peek` (in a second terminal)
  samples from the latest checkpoint into `out/peek.png` and
  `out/peek-progress.png`, the same as

  ```
  diffusion/target/release/diffusion sample-img --model out/digits.safetensors \
      --png out/peek.png --progress out/peek-progress.png
  ```

  It uses the same seed every time, so successive peeks start from the
  same noise and show the training alone; `PEEK_MODEL=` picks another
  run. (How to read the two images: [How to read the
  figures](#how-to-read-the-figures), Image grids.)

**The default DiT**: width 128, 6 blocks, 4 heads, 4 × 4 patches (49
tokens of 16 values for MNIST): 1,843,344 parameters, against Decision
3's guess of 1.2 million. The guess counted the attention and
feed-forward weights (12d² per block) but not adaLN's, a d × 6d matrix per
block, half as many again.

### Choosing the patch size

The patch size is a **hyperparameter**: a setting chosen before training,
not learned (as the width and the number of blocks are). Patches here are
square, and the side must divide the image's: for 28 × 28 the choices are
sides of 1, 2, 4, 7, 14 or 28 pixels, so the realistic alternatives to the
default 4 × 4 are 2 × 2 and 7 × 7.

| Patch | Values per patch | Tokens | Cost per step, relative |
|---|---|---|---|
| 2 × 2 | 4 | 196 | about 4–5× (attention alone 16×) |
| 4 × 4 (default) | 16 | 49 | 1× |
| 7 × 7 | 49 | 16 | about ⅓ |

What the choice trades:

- **Compute.** The matrix products cost in proportion to the tokens,
  attention in proportion to their square. On the M4, 2 × 2 patches would
  turn a 1½-hour MNIST run into 6 to 8 hours; 7 × 7, into about half an
  hour.
- **Detail per token.** Each patch enters through one matrix (its pixels
  to d numbers) and leaves through another (d numbers back to its
  pixels). A 7 × 7 patch must pack 49 pixels into one vector and produce
  all 49 back out of it, so fine detail inside a patch is harder to get
  right; the classic symptom is **block seams**, strokes that step or
  break at the patch boundaries.
- **What attention can relate.** Attention connects patches. With small
  patches, how neighbouring bits of a stroke fit together is attention's
  job, the transformer's strength; with large ones, it's left to the
  single matrix in and out.
- **The evidence**: the DiT paper found exactly this trade, smaller
  patches giving better images at more compute (Peebles and Xie 2023).
  Its best model used 2 × 2 patches, on a 32 × 32 grid of compressed
  image features rather than raw pixels: 256 tokens.

4 × 4 is the middle choice for 28 × 28: few enough tokens for a run of
under two hours on the M4, small enough patches for seams not to
dominate. `make digits-p7` trains the same model with 7 × 7 patches, for
the comparison (prediction 6, below).

### Predictions

Written before any images were generated:

1. **Speed.** In this project's cloud workspace (2 cores, no Accelerate)
   the default DiT trains at about 25 images per second, 5 seconds a step
   of 128: too slow for MNIST here. On the M4, with about 20 times the
   arithmetic (10 cores at the speed tiny-llm-in-rust measured, plus
   Accelerate), about 0.3 seconds a step, so 20,000 steps in about 1½ to 2
   hours: faster than Decision 3's "a few hours", because the M4's
   measured throughput is higher than the guess assumed.
2. **Digits appear early.** Recognizable digits by a few thousand steps,
   with blurry, broken strokes; clean ones by the end. Some samples will
   be ambiguous blends (a 4 that's half 9): nothing tells the model which
   digit to draw until stage 4.
3. **The progress rows settle coarse to fine**, as the velocity field of
   stage 2 did: by t = 0.5 the rough shape and position of the digit are
   fixed; the last steps sharpen edges.
4. **Fewer steps blur.** At 10 Euler steps the digits survive but soften;
   at 2 to 4 steps they smear towards an average digit, a grey blob,
   stage 2's one-step collapse onto the mean in image form.
5. **Fashion-MNIST is harder**: with the same model and steps, its
   samples look worse than the digits do, because clothing has texture
   and more varied shapes than strokes on a blank page.
6. **7 × 7 patches: recognizable digits with visible blocking.** About a
   third of the time per step, and digits still recognizable, but with
   strokes stepped or misaligned at the 7 × 7 boundaries, which 4 × 4
   patches don't show as clearly.

### Results so far

**Speed (prediction 1): right.** On the M4, the default DiT trains on
MNIST at about 435 images per second, 0.29 seconds a step of 128, so
20,000 steps take 1 h 36 min: inside the predicted 1½ to 2 hours, and
about 18 times this cloud workspace's 25 images per second.

**Synthetic shapes, a check of the whole pipeline** (no prediction was
written for these; `make shapes`: 16 × 16, a DiT of width 64, 4 blocks,
16 tokens, 315,536 parameters, at 350 images per second here):

- *3,000 steps (9 minutes)*: every sample is one white shape on black,
  of a plausible size and place, but blobby: no crisp squares, discs or
  crosses. 10 Euler steps look almost the same as 50 (the paths are
  nearly straight, as in stage 2); 2 steps give grey smudges.
- *12,000 steps (38 minutes)*: much crisper; recognizable crosses,
  squares and discs among them, so the blobs were mostly a matter of
  training time. Two failures remain: shapes cut off along straight
  lines that fall on the 4-pixel patch boundaries (the block seams of
  [Choosing the patch size](#choosing-the-patch-size)), and the odd
  fragment or empty image. The loss went from 0.249 to 0.202.

**MNIST at step 2,000** (a checkpoint, a tenth of the run): strokes
already, a few recognizable digits (1s, a 3, a 5, a 2) and many
digit-like blends. Prediction 3 (coarse to fine) holds: each digit's
shape and position appear by t = 0.5 and barely move after. Prediction 2
is on track with a different flavour than predicted: the strokes are
broken and speckled but not blurry. The model has learned MNIST's
texture (pure black, fully white strokes) before its structure (strokes
that join into a whole digit), which needs distant patches to agree:
attention's job, and the slower thing to learn.

## Text, later

Diffusion isn't only for images. This section describes how it works on
text, then compares it with the language model of tiny-llm-in-rust, which
a reader of this project has already built and studied.

### How text diffusion works

Text is tokens, not numbers on a continuum, so Gaussian noise doesn't fit
it: a token plus a little noise isn't a slightly different token. The
noise is **masking** instead: a special MASK token replacing a real one.

- **The forward process**: at time t, each token is hidden independently
  with probability 1 − t, in this project's convention (t = 0 is noise,
  t = 1 data): every token at t = 0, none at t = 1. A fully masked
  sequence is the text equivalent of pure static.
- **Training**: take a text, pick t, mask each token with probability
  1 − t, and train the model to predict the original token at every
  masked position: a cross-entropy loss over the masked positions only.
  The network is a transformer whose attention is **bidirectional**: every
  position sees every other, masked or not, before and after it. This is
  a **masked diffusion language model** (Sahoo et al. 2024); LLaDA (Large
  Language Diffusion with mAsking) showed the recipe works at the scale of billions of parameters (Nie et
  al. 2025).
- **Generation**: start from a sequence that's all MASK and run a fixed
  number of steps. At each, the model predicts every masked position at
  once, and some of those predictions (the most confident, say) are kept,
  the rest masked again. After the last step nothing is masked. A text
  of 256 tokens might take 64 steps, settling four tokens per step on
  average, and not in order: a story's last sentence can appear before
  its middle.
- **A prompt** is simply the part of the sequence that's never masked:
  every position attends to it, and generation fills in what follows.
- **Infilling** (filling a gap with context on both sides: the missing
  middle of a sentence, a function between two others) comes free, since
  nothing requires the masked positions to be at the end.

Hybrids exist too: Transfusion trains one transformer that writes text
token by token and generates images by diffusion, switching between the
two within a sequence (Zhou et al. 2024).

### Compared with tiny-llm-in-rust's model

| | tiny-llm-in-rust (autoregressive) | Masked diffusion |
|---|---|---|
| Attention | causal: each token sees only those before it | bidirectional: every position sees all |
| Training target | the next token, at every position | the original token, at the masked positions |
| Generation | one token per pass, left to right | several tokens per pass, in any order |
| Passes for 256 tokens | 256 | a chosen number of steps, say 64 |
| KV cache | yes: earlier tokens' keys and values never change | no simple one: every position can change until settled, and each pass recomputes all of them |
| A token, once written | final | settled when kept; before that, revised freely |
| Prompt, infilling | prompt yes; infilling awkward | both natural |

What that means for efficiency, in the terms tiny-llm-in-rust measured:

- **Autoregressive generation is limited by memory bandwidth.** Each
  token needs a full pass that reads every weight to produce one token:
  SmolLM-1.7B (the 1.7-billion-parameter open-weight model
  tiny-llm-in-rust runs) generated 9 tokens per second on the M4 because
  every token read its 6.8 GB of weights, while the arithmetic units
  mostly waited.
- **Diffusion settles several tokens per pass.** Each pass still reads the
  weights once, but does useful arithmetic at every position, so the
  hardware is far better used and the whole text arrives sooner.
- **But the total arithmetic is usually higher.** Every pass processes the
  whole sequence, with no cache to skip the settled part, and some work
  goes into predictions that are discarded and redone. Diffusion trades
  more arithmetic for fewer sequential passes: a good trade where
  arithmetic is plentiful and passes are the bottleneck, as on a GPU
  (graphics processing unit, the chips most models are trained on), not
  a free saving.
- **Costs in use**: text that may still change is hard to stream as it's
  written, which matters for code, structured data and speech.

Where the field stood in August 2026, as a dated note rather than a
result of this project: hosted diffusion models generated roughly 800 to
2,100 tokens per second against 40 to 200 for autoregressive ones, but
scored well below the best autoregressive models on quality, and their
first-answer times weren't better (BenchLM 2026; figures from different
models and hardware, so indicative only).

**The experiment for stage 6**, then, at a size this project can
measure: a masked diffusion model and tiny-llm-in-rust's model, the same
tokenizer, corpus and parameter count, both on the M4. Predictions to
record before running:

- diffusion generates a story faster, since the M4 is bandwidth-limited
  as above, by a factor that depends on the number of steps;
- its stories are less coherent at the same size, by a margin to be
  measured with the same losses and the same requested-word evaluation
  tiny-llm-in-rust used;
- it keeps characters straight differently: it sees the whole story at
  every step, which may help with consistency, but settles tokens in
  parallel, which may let two tokens settled in the same step contradict
  each other, since neither sees the other's choice.

## Decisions

Settled before stage 2, each with its reason; new questions get added
here as they come up, and stay open until settled.

1. **The image data: digits first, clothing as a comparison.** Decided.
   MNIST (Modified National Institute of Standards and Technology
   digits; LeCun et al. 1998: 70,000 handwritten digits, 28 × 28, about
   11 MB compressed) is the classic, and digits make failures easy to see:
   a malformed 4 is obvious. Fashion-MNIST (Xiao et al. 2017: the same
   format and size, ten kinds of clothing instead of digits, MIT-licensed)
   is supported too, by a command-line option, so the same model and
   settings can be compared on a harder kind of image. Both are downloaded
   by the Makefile, never committed, which also leaves MNIST's vaguely
   stated terms out of the repository.
2. **Images are written as PNG** (Portable Network Graphics). Decided.
   PNG is what every viewer opens,
   GitHub included. A minimal writer needs no library: uncompressed
   ("stored") deflate blocks, with the checksums the format requires
   (CRC-32, a cyclic redundancy check, and Adler-32), a few dozen lines, tested by checking those checksums
   and reading the pixels back. (PGM, a plain grayscale format, was the
   simpler alternative, but GitHub doesn't display it.)
3. **The DiT's size is set on the command line, the first guess as the
   defaults.** Decided: width 128, 6 blocks, 4 heads, 4 × 4 patches (49
   tokens), about 1.2 million parameters, each changeable with an option
   (`--width`, `--blocks`, `--heads`, `--patch`), and recorded in the
   checkpoint so a model always says what shape it is. The estimate, to be
   corrected by measurement: roughly 0.4 GFLOP per example per training
   step, so a batch of 128 costs about 50 GFLOP; at the tens of GFLOP per
   second tiny-llm-in-rust's code reached on the M4 with Accelerate,
   that's around a second a step, and 20,000 steps a few hours.
4. **DDPM gets both samplers.** Decided: its own 1,000-step ancestral
   sampler, and DDIM's deterministic shortcut (Song et al. 2021), which
   takes about 50 steps with the same trained model. The comparison (same
   weights, a twentieth of the passes, how much quality is lost) is the
   lesson.
5. **Image size, channels and patch size come from the data and the
   command line; nothing is fixed at 28 × 28.** Decided. A DiT doesn't
   depend on the image's size: an H × W image cut into p × p patches is
   (H/p)·(W/p) tokens, a color patch is p·p·3 numbers instead of p·p, and
   the position embeddings are sized from H/p and W/p when the model is
   built. The shape is recorded in the checkpoint, so a model refuses
   images of another size with a clear error. What limits the size is
   compute: tokens grow with the area, attention's cost with the tokens'
   square (64 × 64 with 4 × 4 patches is 256 tokens, about 27 times
   MNIST's attention cost), and larger patches trade detail for speed.
   Tested at several sizes, channel counts and patch sizes, and with a
   synthetic data set of shapes at any size (`--data shapes --size N
   --channels C`), so larger images can be tried without a download.
   Training starts on MNIST, as decided; CIFAR-10 (32 × 32, color) would
   be the natural next data set.

## Where things live

```
tiny-diffusion-in-rust/
├── README.md      # a short introduction, for GitHub
├── PLAN.md        # this file
├── LICENSE        # MIT
├── Makefile       # make test runs every check; one target per step as stages arrive
├── diffusion/     # the Rust crate: the library and the `diffusion` command
│   ├── src/       # tensor, tape, blas, rng, safetensors (the foundation);
│   │              # data2d, flow, ddpm, mlp, optim, train2d, svg, metrics (stage 2);
│   │              # image, dit, trainimg (stage 3); main.rs, the commands
│   └── tests/data # test fixtures
├── docs/          # figures for this file and the README
├── tools/         # toc.py: this file's table of contents and link check;
│                  # acronyms.py: every acronym defined at first use;
│                  # ci/smoke.sh: every command end to end, in seconds
├── .github/       # workflows/main.yaml: formatting, lint and tests, on Linux and macOS
├── data/          # downloaded datasets (git-ignored)
└── out/           # generated files: models, plots, images (git-ignored)
```

## Bibliography

Venues: ICML (International Conference on Machine Learning), NeurIPS
(Neural Information Processing Systems), ICLR (International Conference on
Learning Representations), ICCV (International Conference on Computer
Vision); arXiv is the open server where most papers in the field appear
first.

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
- Székely, G. J., & Rizzo, M. L. (2013). "Energy statistics: A class of
  statistics based on distances." Journal of Statistical Planning and
  Inference, 143(8), 1249–1272. The energy distance.
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
- Zhou, C., et al. (2024). "Transfusion: Predict the Next Token and
  Diffuse Images with One Multi-Modal Model."
  [arXiv:2408.11039](https://arxiv.org/abs/2408.11039).
- BenchLM (2026). "Diffusion vs Autoregressive LLMs," August 3, 2026.
  [benchlm.ai](https://www.benchlm.ai/blog/posts/diffusion-vs-autoregressive-llm).
  Measured speeds and quality scores of hosted models.
