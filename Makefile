SHELL := /bin/bash
OUT   ?= out

# CARGO_FEATURES=blas builds with Apple's Accelerate library for the large
# matrix products: the default on macOS. CARGO_FEATURES= builds the plain
# Rust loops.
CARGO_FEATURES ?= $(if $(filter Darwin,$(shell uname -s)),blas,)
CARGO_FEATURE_FLAGS = $(if $(CARGO_FEATURES),--features $(CARGO_FEATURES))
CRATE     = --manifest-path diffusion/Cargo.toml
DIFFUSION = diffusion/target/release/diffusion
DIFFUSION_SRC = $(wildcard diffusion/src/*.rs) diffusion/Cargo.toml

# Print a section header: $(call hdr,text)
hdr = @printf "\n\033[35;1m== %s ==\033[0m\n" "$(1)"

.PHONY: build
build: $(DIFFUSION) ## Build the diffusion command.

$(DIFFUSION): $(DIFFUSION_SRC)
	$(call hdr,build diffusion)
	cargo build --release $(CRATE) $(CARGO_FEATURE_FLAGS)

.PHONY: info
info: $(DIFFUSION) ## Show what the build contains.
	$(DIFFUSION) info

$(OUT):
	mkdir -p $@

# Stage 2, points in a plane (PLAN.md). out/<data>-<objective>.safetensors
# is a trained model; .svg its samples, paths and velocity field;
# -steps.svg the samples by number of sampling steps. TRAIN2D_ARGS passes
# options to train2d, e.g. TRAIN2D_ARGS='--steps 2000 --width 64'.
TRAIN2D_ARGS ?=

$(OUT)/%-flow.safetensors: $(DIFFUSION) | $(OUT)
	$(call hdr,train2d $* flow)
	$(DIFFUSION) train2d --data $* --objective flow $(TRAIN2D_ARGS) -o $@

$(OUT)/%-ddpm.safetensors: $(DIFFUSION) | $(OUT)
	$(call hdr,train2d $* ddpm)
	$(DIFFUSION) train2d --data $* --objective ddpm $(TRAIN2D_ARGS) -o $@

$(OUT)/%-steps.svg: $(OUT)/%.safetensors
	$(call hdr,steps2d $*)
	$(DIFFUSION) steps2d --model $< --svg $@

$(OUT)/%.svg: $(OUT)/%.safetensors
	$(call hdr,sample2d $*)
	$(DIFFUSION) sample2d --model $< --svg $@

.PHONY: moons spiral rings gaussian moons-ddpm stage2 exact
moons spiral rings gaussian: %: $(OUT)/%-flow.svg $(OUT)/%-flow-steps.svg ## Train flow on that data set and plot it (e.g. make moons).
moons-ddpm: $(OUT)/moons-ddpm.svg $(OUT)/moons-ddpm-steps.svg ## DDPM on moons, for comparison (DDIM and ancestral sampling).
stage2: exact moons moons-ddpm rings spiral ## Everything of stage 2: a few minutes.
exact: $(DIFFUSION) ## The samplers with the exact Gaussian fields, no network.
	$(call hdr,exact)
	$(DIFFUSION) exact

# Stage 3, images (PLAN.md). The data sets are downloaded into data/ and
# decompressed; the models depend on the program only order-only (|), so a
# code change doesn't silently retrain a run of an hour or more: retrain on
# purpose with make -B. IMG_ARGS passes options to train-img, e.g.
# IMG_ARGS='--steps 5000 --width 96'.
URL_mnist   ?= https://ossci-datasets.s3.amazonaws.com/mnist
URL_fashion ?= https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion
IMG_ARGS    ?=
SHAPES_ARGS ?= --size 16 --width 64 --blocks 4 --batch 64 --steps 3000 --warmup 200 --lr 2e-3

data/%/train-images-idx3-ubyte data/%/train-labels-idx1-ubyte:
	$(call hdr,download $*)
	mkdir -p data/$*
	for f in train-images-idx3-ubyte train-labels-idx1-ubyte; do \
		curl -fL -o data/$*/$$f.gz $(URL_$*)/$$f.gz && gunzip -f data/$*/$$f.gz || exit 1; \
	done

# The test images: 10,000 per set, which no model trains on (the
# classifier is checked on them, stage 4).
data/%/t10k-images-idx3-ubyte data/%/t10k-labels-idx1-ubyte:
	$(call hdr,download $* test images)
	mkdir -p data/$*
	for f in t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do \
		curl -fL -o data/$*/$$f.gz $(URL_$*)/$$f.gz && gunzip -f data/$*/$$f.gz || exit 1; \
	done

.PHONY: data/mnist data/fashion
data/mnist data/fashion: data/%: data/%/train-images-idx3-ubyte ## Download MNIST or Fashion-MNIST (make data/mnist).

$(OUT)/digits.safetensors: data/mnist/train-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,train-img digits)
	$(DIFFUSION) train-img --data mnist $(IMG_ARGS) -o $@

$(OUT)/digits-p7.safetensors: data/mnist/train-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,train-img digits with 7x7 patches)
	$(DIFFUSION) train-img --data mnist --patch 7 $(IMG_ARGS) -o $@

$(OUT)/fashion.safetensors: data/fashion/train-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,train-img fashion)
	$(DIFFUSION) train-img --data fashion $(IMG_ARGS) -o $@

$(OUT)/shapes.safetensors: | $(DIFFUSION) $(OUT)
	$(call hdr,train-img shapes)
	$(DIFFUSION) train-img --data shapes $(SHAPES_ARGS) -o $@

$(OUT)/%.png: $(OUT)/%.safetensors
	$(call hdr,sample-img $*)
	$(DIFFUSION) sample-img --model $< --png $@ --progress $(OUT)/$*-progress.png --real $(OUT)/$*-real.png

# Look at a run while it trains: sample from its latest checkpoint (written
# every 1,000 steps) into out/peek.png and out/peek-progress.png. Same seed
# each time, so successive peeks start from the same noise and differ only
# by the training. PEEK_MODEL picks the run: make peek PEEK_MODEL=out/fashion.safetensors
PEEK_MODEL ?= $(OUT)/digits.safetensors

.PHONY: peek
peek: | $(DIFFUSION) ## Sample from a running image model's latest checkpoint (PEEK_MODEL).
	@test -f $(PEEK_MODEL) || { echo "no $(PEEK_MODEL) yet: the first checkpoint comes at step 1,000"; exit 1; }
	$(DIFFUSION) sample-img --model $(PEEK_MODEL) --png $(OUT)/peek.png --progress $(OUT)/peek-progress.png

.PHONY: digits digits-p7 fashion shapes
digits: $(OUT)/digits.png ## Train the DiT on MNIST digits and draw samples: 1-2 hours on an M4.
digits-p7: $(OUT)/digits-p7.png ## The same with 7x7 patches (16 tokens): about half an hour.
fashion: $(OUT)/fashion.png ## The same on Fashion-MNIST, for comparison.
shapes: $(OUT)/shapes.png ## A small DiT on synthetic 16x16 shapes, no download: minutes.

# Stage 4, conditioning (PLAN.md). First the measuring instrument: a digit
# classifier, trained on MNIST's training images and checked on its test
# images. CLASSIFIER_ARGS passes options to it.
CLASSIFIER_ARGS ?=

$(OUT)/classifier.safetensors: data/mnist/train-images-idx3-ubyte data/mnist/t10k-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,classifier)
	$(DIFFUSION) classifier --data mnist $(CLASSIFIER_ARGS) -o $@

# Then the models with labels: 7 x 7 patches to develop and measure, one
# 4 x 4 run to confirm (Decision 7). The usual %.png rule samples them
# (labels taken in turn, 0 to 9, along the grid).
$(OUT)/digits-cond.safetensors: data/mnist/train-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,train-img digits with labels and 7x7 patches)
	$(DIFFUSION) train-img --data mnist --patch 7 --conditional $(IMG_ARGS) -o $@

$(OUT)/digits-cond-p4.safetensors: data/mnist/train-images-idx3-ubyte | $(DIFFUSION) $(OUT)
	$(call hdr,train-img digits with labels and 4x4 patches)
	$(DIFFUSION) train-img --data mnist --patch 4 --conditional $(IMG_ARGS) -o $@

# The measurement table and a grid per guidance scale (one row per digit,
# the same noise down each column): out/digits-cond-w2.png and so on.
# GUIDE_MODEL picks the model: make guidance GUIDE_MODEL=out/digits-cond-p4.safetensors
GUIDE_MODEL ?= $(OUT)/digits-cond.safetensors
GUIDANCE    ?= 0 1 1.5 2 3 5 8
comma := ,
empty :=
space := $(empty) $(empty)

.PHONY: digits-cond digits-cond-p4 guidance
digits-cond: $(OUT)/digits-cond.png ## Stage 4: the DiT with labels, 7x7 patches: about half an hour.
digits-cond-p4: $(OUT)/digits-cond-p4.png ## Stage 4's confirmation with 4x4 patches: about 1 1/2 hours.

guidance: $(OUT)/classifier.safetensors | $(DIFFUSION) ## Measure GUIDE_MODEL at each guidance scale, and draw a grid of digits for each.
	@test -f $(GUIDE_MODEL) || { echo "no $(GUIDE_MODEL) yet: make digits-cond first"; exit 1; }
	$(call hdr,measure $(GUIDE_MODEL) by guidance scale)
	$(DIFFUSION) measure --model $(GUIDE_MODEL) --classifier $< --guidance $(subst $(space),$(comma),$(strip $(GUIDANCE)))
	$(call hdr,does the noise decide the style? (prediction 6))
	$(DIFFUSION) style --model $(GUIDE_MODEL)
	$(call hdr,grids by guidance scale)
	for w in $(GUIDANCE); do \
		$(DIFFUSION) sample-img --model $(GUIDE_MODEL) --classes --n 8 --guidance $$w \
			--png $(OUT)/$(basename $(notdir $(GUIDE_MODEL)))-w$$w.png || exit 1; \
	done

# Stage 5, looking inside (PLAN.md): the probes on stage 3's model without
# labels and stage 4's with them, 4 x 4 patches both; tables on the
# terminal, and out/inside-<model>-preview.png and -attention.png.
INSIDE_MODELS ?= digits digits-cond-p4

.PHONY: inside
inside: $(OUT)/classifier.safetensors | $(DIFFUSION) ## Stage 5: look inside the trained models (INSIDE_MODELS): a few minutes.
	@for m in $(INSIDE_MODELS); do \
		test -f $(OUT)/$$m.safetensors || { echo "no $(OUT)/$$m.safetensors: train it first (make $$m)"; exit 1; }; \
	done
	for m in $(INSIDE_MODELS); do \
		printf "\n\033[35;1m== inside %s ==\033[0m\n" $$m; \
		$(DIFFUSION) inside --model $(OUT)/$$m.safetensors --classifier $< \
			--preview $(OUT)/inside-$$m-preview.png --attention $(OUT)/inside-$$m-attention.png || exit 1; \
	done

# Stage 6, text (PLAN.md): a masked diffusion language model on
# tiny-llm-in-rust's tokenizer and corpus, read from its out/ folder
# (LLM_OUT), and compared with its trained model there (model.safetensors).
# The model uses rotary positions (--rope, Decision 8 in PLAN.md).
# TEXT_ARGS passes options to train-text. TEXT_AR_SAMPLES is a file of
# tiny-llm-in-rust's own stories (make llm-stories), measured beside this
# model's when it exists.
LLM_OUT         ?= ../tiny-llm-in-rust/out
TEXT_ARGS       ?=
TEXT_AR_SAMPLES ?= $(OUT)/llm-stories.txt

$(OUT)/text.safetensors: | $(DIFFUSION) $(OUT)
	@for f in corpus.ids tok.bpe; do \
		test -f $(LLM_OUT)/$$f || { echo "no $(LLM_OUT)/$$f: run make encode in tiny-llm-in-rust, or set LLM_OUT"; exit 1; }; \
	done
	$(call hdr,train-text)
	$(DIFFUSION) train-text --ids $(LLM_OUT)/corpus.ids --rope \
		$(if $(wildcard $(LLM_OUT)/model.safetensors),--compare $(LLM_OUT)/model.safetensors) $(TEXT_ARGS) -o $@

.PHONY: text text-sample
text: $(OUT)/text.safetensors ## Stage 6: train the masked diffusion language model on tiny-llm-in-rust's corpus: minutes.

text-sample: $(OUT)/text.safetensors | $(DIFFUSION) ## Stage 6: stories at 16, 64 and 256 steps, timed; an infilled story; the text measures.
	$(call hdr,sample-text)
	$(DIFFUSION) sample-text --model $< --tokenizer $(LLM_OUT)/tok.bpe --ids $(LLM_OUT)/corpus.ids --out-prefix $(OUT)/text
	$(call hdr,text measures)
	python3 tools/text_stats.py $(OUT)/text-real.txt \
		$(if $(wildcard $(TEXT_AR_SAMPLES)),--tiny-llm $(TEXT_AR_SAMPLES)) \
		$(OUT)/text-s16.txt $(OUT)/text-s64.txt $(OUT)/text-s256.txt

# tiny-llm-in-rust's own stories, 8 of up to 256 tokens from 'Once upon a
# time', by its own program, for make text-sample to measure alongside.
LLM_DIR := $(LLM_OUT)/..
.PHONY: llm-stories
llm-stories: | $(OUT) ## Stage 6: 8 stories from tiny-llm-in-rust's trained model, into TEXT_AR_SAMPLES.
	@test -x $(LLM_DIR)/transformer/target/release/transformer || { echo "build tiny-llm-in-rust first (make build there)"; exit 1; }
	$(call hdr,stories from tiny-llm-in-rust)
	for s in 1 2 3 4 5 6 7 8; do \
		$(LLM_DIR)/transformer/target/release/transformer generate --model $(LLM_OUT)/model.safetensors \
			--tokenizer $(LLM_OUT)/tok.bpe --config $(LLM_DIR)/config.toml --prompt 'Once upon a time' \
			--seed $$s --max-tokens 256 || exit 1; \
	done > $(TEXT_AR_SAMPLES)
	@echo "wrote $(TEXT_AR_SAMPLES)"

.PHONY: classifier measure-stage3
classifier: $(OUT)/classifier.safetensors ## Train the digit classifier, stage 4's measuring instrument: a minute or two.

measure-stage3: $(OUT)/classifier.safetensors | $(DIFFUSION) ## The classifier on stage 3's digit models (they must exist: make digits digits-p7).
	@for m in digits digits-p7; do \
		test -f $(OUT)/$$m.safetensors || { echo "no $(OUT)/$$m.safetensors: train it first (make $$m)"; exit 1; }; \
	done
	$(call hdr,measure digits)
	$(DIFFUSION) measure --model $(OUT)/digits.safetensors --classifier $<
	$(call hdr,measure digits-p7)
	$(DIFFUSION) measure --model $(OUT)/digits-p7.safetensors --classifier $<

.PHONY: smoke
smoke: $(DIFFUSION) ## Every command end to end on a tiny model, in seconds (tools/ci/smoke.sh).
	$(call hdr,$@)
	tools/ci/smoke.sh > /dev/null
	@echo "every command ran"

# Checks: the same ones CI runs (.github/workflows/main.yaml); keep the two
# in step. Each stage adds its tests here as it arrives (PLAN.md, Design
# for test).
.PHONY: test
test: test-rust test-python smoke lint ## Run every check: the tests, the smoke test and lint.
	@printf "\nAll checks passed.\n"

.PHONY: test-rust
test-rust: ## Rust unit tests.
	$(call hdr,$@)
	cargo test --release -q $(CRATE) $(CARGO_FEATURE_FLAGS)

.PHONY: test-python
test-python: ## Python unit tests for the tools.
	$(call hdr,$@)
	python3 -m unittest discover -s tools

.PHONY: lint
lint: ## Formatting, clippy, pylint (if installed), PLAN.md's links, acronyms defined.
	$(call hdr,$@)
	cargo fmt $(CRATE) -- --check
	cargo clippy --release -q $(CRATE) --all-targets $(CARGO_FEATURE_FLAGS) -- -D warnings
	@if command -v pylint > /dev/null; then \
		echo "pylint tools/*.py"; pylint --score=n tools/*.py; \
	else echo "pylint not installed: skipped (pip install pylint)"; fi
	python3 tools/toc.py --check PLAN.md
	python3 tools/acronyms.py PLAN.md README.md --known tools/acronyms-known.txt
	@missing=$$(grep -oE '^[A-Za-z0-9_]+ *\?=' Makefile | tr -d ' ?=' | \
		while read v; do echo " $(HELP_VARS) " | grep -q " $$v " || echo $$v; done); \
	test -z "$$missing" || { echo "make help doesn't list: $$missing (add to HELP_VARS)"; exit 1; }; \
	echo "make help lists every variable"
	@$(MAKE) -s help > /dev/null && echo "make help runs"
	@for f in $$(grep -ohE 'docs/[A-Za-z0-9._-]+' PLAN.md README.md | sort -u); do \
		test -f $$f || { echo "PLAN.md or README.md shows $$f, which doesn't exist"; exit 1; }; \
	done; echo "every figure the documents show exists"

.PHONY: toc
toc: ## Update PLAN.md's table of contents and check its internal links.
	python3 tools/toc.py PLAN.md

.PHONY: clean
clean: ## Remove generated files (keeps data/).
	cd diffusion && cargo clean
	rm -rf $(OUT)

# The variables make help lists, one per line, with what each does and its
# value now. A variable meant to be set from the command line gets a line
# here and a HELP_ description.
HELP_VARS := OUT CARGO_FEATURES TRAIN2D_ARGS IMG_ARGS SHAPES_ARGS PEEK_MODEL \
	CLASSIFIER_ARGS GUIDE_MODEL GUIDANCE INSIDE_MODELS LLM_OUT TEXT_ARGS \
	TEXT_AR_SAMPLES URL_mnist URL_fashion
HELP_OUT             := where models, plots and images are written
HELP_CARGO_FEATURES  := blas: Apple Accelerate for large matrix products (default on macOS); empty: plain Rust
HELP_TRAIN2D_ARGS    := options for train2d, the stage 2 models (diffusion train2d --help)
HELP_IMG_ARGS        := options for train-img, the digit, clothing and labelled models (diffusion train-img --help)
HELP_SHAPES_ARGS     := the whole option set for make shapes, replacing the default
HELP_PEEK_MODEL      := the running model make peek samples from
HELP_CLASSIFIER_ARGS := options for the classifier (diffusion classifier --help)
HELP_GUIDE_MODEL     := the labelled model make guidance measures
HELP_GUIDANCE        := the guidance scales make guidance measures and draws
HELP_INSIDE_MODELS   := the models make inside looks into (names in out/, without .safetensors)
HELP_LLM_OUT         := tiny-llm-in-rust's out/ folder: its tok.bpe, corpus.ids and model.safetensors
HELP_TEXT_ARGS       := options for train-text, the masked diffusion language model (diffusion train-text --help)
HELP_TEXT_AR_SAMPLES := tiny-llm-in-rust's stories (make llm-stories writes them), measured beside this model's by make text-sample
HELP_URL_mnist       := where MNIST is downloaded from
HELP_URL_fashion     := where Fashion-MNIST is downloaded from

# A value inside single quotes, with any single quote in it escaped.
q = $(subst ','\'',$(1))

.PHONY: help
help: ## This help message.
	$(call hdr,$@)
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf("   %-15s %s\n", $$1, $$2)}' | sort
	@printf "\n   Variables, set on the command line (make digits IMG_ARGS='--steps 5000'):\n"
	@$(foreach v,$(HELP_VARS),printf "   %-15s %s\n   %-15s now: %s\n" '$(v)' '$(call q,$(HELP_$(v)))' '' '$(call q,$(or $($(v)),(empty)))';)
