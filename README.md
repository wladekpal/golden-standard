# Installation

Using uv, there is installation happening while running the code.

```bash
uv run src/train.py 
```

Running tests:
```bash
uv run pytest src/envs/tests.py
```

# For development

Install pre-commit globally (you can [follow this article](https://adamj.eu/tech/2025/05/07/pre-commit-install-uv/)):

```bash
uv tool install pre-commit --with pre-commit-uv
```