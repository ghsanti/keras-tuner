## Setup Environment

First, fork the repository.

There are 3 different options:

1. **GitHub Codespaces**,
2. **VS Code & Remote-Containers**.
3. **Locally in a venv**

### Option 1: GitHub Codespace

When you open your fork on a Codespace, it automatically installs all needed (including extensions like Ruff and Pyright.)

_You can start developing._

### Option 2: VS Code & Local-Containers

Open VS Code.
Install the `Dev-Containers` extension.
Press `F1` key. Enter `Dev-Containers: Open Folder in Container` to open the repository root folder. The environment is already setup there.

You can use this remotely as well.

### Option 3: Local copy

Once you clone your fork, it's recommended here that you use `venv` and develop within a virtual environment, as described below.

`cd` into your `keras-tuner` folder and run commands below.

```bash
python3.11 -m venv .venv && source .venv/bin/activate && sh .devcontainer/setup.sh
```

## Run Tests

You can use the `python -m pytest path/to/file` to run the tests.

## Code Style

We use `ruff` and `pyright` both have VSCode extensions.
Just search in VSCode Extensions and install them.

Also `shell/lint_format.sh` can format the code.

It's useful to have this settings in your JSON-workspace:

```json
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
```

## Pull Request Guide

Before you submit a pull request, check that it meets these guidelines:

1. Is this the first pull request that you're making with GitHub? If so, read the guide [Making a pull request to an open-source project](https://github.com/gabrieldemarmiesse/getting_started_open_source).

2. Include "resolves #issue_number" in the description of the pull request if applicable and briefly describe your contribution.

3. For the case of bug fixes, add new test cases which would fail before your bug fix.

## Rebuilding Protos

If you make changes to any `.proto` file, you'll have to rebuild the generated
`*_pb2.py` files. To do this, run these commands from the root directory of this
project:

```bash
pip install grpcio-tools &&\
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/keras_tuner.proto &&\
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. keras_tuner/protos/service.proto
```
