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

.PHONY: smoke
smoke: $(DIFFUSION) ## Every command end to end on a tiny model, in seconds (tools/ci/smoke.sh).
	$(call hdr,$@)
	tools/ci/smoke.sh > /dev/null
	@echo "every command ran"

# Checks: the same ones CI runs (.github/workflows/main.yaml); keep the two
# in step. Each stage adds its tests here as it arrives (PLAN.md, Design
# for test).
.PHONY: test
test: test-rust smoke lint ## Run every check: the tests, the smoke test and lint.
	@printf "\nAll checks passed.\n"

.PHONY: test-rust
test-rust: ## Rust unit tests.
	$(call hdr,$@)
	cargo test --release -q $(CRATE) $(CARGO_FEATURE_FLAGS)

.PHONY: lint
lint: ## Formatting, clippy, pylint (if installed), PLAN.md's links.
	$(call hdr,$@)
	cargo fmt $(CRATE) -- --check
	cargo clippy --release -q $(CRATE) --all-targets $(CARGO_FEATURE_FLAGS) -- -D warnings
	@if command -v pylint > /dev/null; then \
		echo "pylint tools/*.py"; pylint --score=n tools/*.py; \
	else echo "pylint not installed: skipped (pip install pylint)"; fi
	python3 tools/toc.py --check PLAN.md

.PHONY: toc
toc: ## Update PLAN.md's table of contents and check its internal links.
	python3 tools/toc.py PLAN.md

.PHONY: clean
clean: ## Remove generated files (keeps data/).
	cd diffusion && cargo clean
	rm -rf $(OUT)

.PHONY: help
help: ## This help message.
	$(call hdr,$@)
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
		awk -F':.*## ' '{printf("   %-12s %s\n", $$1, $$2)}' | sort
	@printf "\n   Variables: CARGO_FEATURES=%s OUT=%s\n" "$(CARGO_FEATURES)" "$(OUT)"
