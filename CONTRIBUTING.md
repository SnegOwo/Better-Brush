# Contributing to Better Brush

Thanks for taking a look. Better Brush is a small, solo-developed project, so
the most useful contributions are bug reports with a reproduction and focused
pull requests.

## Development setup

```bash
git clone https://github.com/SnegOwo/Better-Brush.git
cd Better-Brush
python -m venv .venv
.venv\Scripts\activate

# CUDA build of PyTorch first (plain `pip install torch` is CPU-only on Windows):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python -m src.main
```

## Before opening a pull request

```bash
python -m pytest tests/ -q      # full suite, no GPU required
python -m ruff check src/       # lint (src/vendor is excluded)
```

Keep changes focused, match the style of the surrounding code, and add a test
when you fix a bug. Do not commit model weights, footage, or build output —
`.gitignore` already keeps the usual suspects out.

## License of contributions

By opening a pull request you agree that your contribution is licensed under
the same [FSL-1.1-MIT license](LICENSE) as the rest of the project.

## Reporting security issues

Please do not open a public issue for a security problem. Email
[dev@sneg.dev](mailto:dev@sneg.dev) instead.
