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

# Checks: the same ones CI runs (.github/workflows/main.yaml); keep the two
# in step. Each stage adds its tests here as it arrives (PLAN.md, Design
# for test).
.PHONY: test
test: test-rust lint ## Run every check: the tests and lint.
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
