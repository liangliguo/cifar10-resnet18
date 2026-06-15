# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ResNet-18 fine-tuned on CIFAR-10, used as a testbed for adversarial attacks
(FGSM / DeepFool / APGD) and a PGD adversarial-training defense. The bulk of the
work lives in a single Jupyter notebook; `ppt/` holds a Beamer report of the results.

## Commands

```bash
uv sync                       # create .venv and install all deps (Python 3.12)
uv run jupyter lab            # launch the notebook (or: uv run jupyter notebook)
.venv/bin/python make_confusion_clean.py   # standalone: clean-test confusion matrix → ppt/confusion_clean.pdf
```

GPU: the project defaults to **CPU-only** PyTorch wheels. For an NVIDIA GPU, edit
`[tool.uv.sources]` in `pyproject.toml` to point `torch`/`torchvision` at the
`torch-cu128` index, then re-run `uv sync`. There is no test suite or linter.

Compile the report (`ppt/at.tex`). It needs Noto CJK fonts (`Noto Serif/Sans CJK SC`).

- **macOS (this dev box): use Tectonic** — there is no TeXLive/`xelatex` installed here.
  Tectonic bundles XeTeX, auto-runs multiple passes (no need to run twice), and picks up the
  Noto CJK fonts from `~/Library/Fonts`:

  ```bash
  cd ppt && tectonic at.tex          # → ppt/at.pdf
  ```

  Harmless warnings on this box: `nullfont` missing-digit (beamer footer), `m/it undefined`
  (CJK has no real italic, falls back to upright), and `absolute path .../NotoSerifCJKsc`
  (uses system fonts, not reproducible elsewhere). Check page count with `pdfinfo at.pdf`.

- **Elsewhere (TeXLive installed): XeLaTeX**, run twice for TOC/nav:

  ```bash
  cd ppt && xelatex at.tex && xelatex at.tex
  ```

## Notebook architecture (`resnet-18-apgd.ipynb`)

18 cells in two halves. **Cells run in order and share global state** — later cells
reuse models, loaders, and helper functions defined earlier, so re-running one
attack cell in isolation will fail unless its prerequisites have been run.

- **Cells 0–9 — base training:** data loading, CIFAR ResNet-18, early-stopped
  training, evaluation, save `resnet18_cifar10_best.pth`. Cell 4 has
  `SKIP_TRAINING=True` to load an existing checkpoint instead of retraining.
- **Cell 11 — "优化版本" (foundation for all attacks):** defines `atk_model`,
  the helper functions, and `test_loader` that cells 12–17 depend on. **Run this
  before any attack/defense cell.**
- **Cells 12–15 — attacks + viz:** t-SNE, DeepFool, APGD, confusion matrices.
- **Cell 16 — PGD adversarial training** → `resnet18_cifar10_pgd_at.pth` (robust model).
- **Cell 17 — honest evaluation:** base vs robust model under FGSM < PGD-20 < APGD.

The notebook is written to run on Kaggle (multi-GPU via `DataParallel`, paths under
`/kaggle/input`) but falls back to local `./data`.

## Critical conventions (do not break these)

- **[0,1] pixel space + normalization inside the model.** During *training* (cell 0),
  normalization is part of the `transforms` pipeline. During *attacks* (cell 11+),
  images stay in `[0,1]` and normalization is wrapped as the model's first layer
  (`atk_model`). This is required because `torchattacks` does `clamp(0,1)` internally;
  normalizing in the transform would let that clamp destroy the adversarial image.
  `make_confusion_clean.py` mirrors the **training** convention (normalize in transform,
  feed the bare model), which is why it reproduces the checkpoint's 88.53% exactly.
- **CIFAR ResNet-18 surgery:** `conv1` → 3×3 stride 1, `maxpool` → `Identity`.
  Without this, 32×32 inputs collapse to 8×8 and accuracy tanks. Any code that
  rebuilds the model must apply the same change before `load_state_dict`.
- **Evaluation口径:** report *conditional* attack success rate (only samples the
  model got right but the attack flipped), and always pair **Clean Acc + Robust Acc**.
  Use the strength ladder FGSM < PGD-20 < APGD; robust acc must decrease along it,
  otherwise suspect gradient masking.
- **PGD for training, DeepFool for evaluation only.** DeepFool finds the *minimum*
  perturbation (samples sit on the boundary, too weak for training); PGD maximizes
  loss within a fixed ε budget, which is what adversarial training needs.

## Files & artifacts

- `*.pth` checkpoints and `to/` (Kaggle run outputs: figures, logs) are gitignored.
  `to/dest/resnet18_cifar10_best.pth` is the trained base model used by the scripts.
- `ppt/at.tex` is the report; figure PDFs in `ppt/` are committed deliverables, but
  LaTeX build artifacts (`.aux`, `.toc`, `.synctex.gz`, …) are gitignored.
