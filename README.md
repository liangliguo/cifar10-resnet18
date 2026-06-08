# cifar10-resnet18

ResNet-18 fine-tuned on CIFAR-10, with adversarial attacks (FGSM / DeepFool / APGD)
and a Gaussian-smoothing defense. See `resnet-18-apgd.ipynb`.

## Environment setup (uv)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Create the virtual environment and install all dependencies
uv sync

# Launch Jupyter / run the notebook
uv run jupyter lab          # or: uv run jupyter notebook
```

Register the environment as a Jupyter kernel (optional):

```bash
uv run python -m ipykernel install --user --name cifar10-resnet18
```

### GPU vs CPU

The config defaults to the CPU-only PyTorch wheels. If you have an NVIDIA GPU,
edit `[tool.uv.sources]` in `pyproject.toml` to point `torch`/`torchvision` at
the `torch-cu128` index instead, then re-run `uv sync`.
