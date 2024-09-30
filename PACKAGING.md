PM: package manager.

## Package Managers

1. [Poetry Install Guide](https://python-poetry.org/docs/)
2. [uv install](https://docs.astral.sh/uv/getting-started/installation)

## Virtual Environments

One could use `venv` or `virtualenv` or `conda`. Conda tends to be more useful when installing GPU stuff and system libraries, and otherwise not necessary.

In both cases below, it's useful to `alias sve=source .venv/bin/activate/`.

```bash
python3.11 -m venv .venv
sve
```

then use the PM within the env.

`uv`can also [manage `.venv`.](https://docs.astral.sh/uv/pip/environments/); for example `uv venv` creates one at `.venv`.

Normally, in any case, you'd use the version:

```bash
uv venv --python 3.11
```

### Python versions

It is handy to have and test with supported python versions, `uv` makes it easy:

```bash
uv python install 3.10 3.11 3.12
```
