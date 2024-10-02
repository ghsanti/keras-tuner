# KerasTuner

[![license](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/ghsanti/keras-tuner/blob/main/LICENSE)
![py-version](https://img.shields.io/badge/Python-3.10+-blue)
[![tests](https://github.com/keras-team/keras-tuner/workflows/Tests/badge.svg?branch=main)](https://github.com/keras-team/ghsanti/actions?query=workflow%3ATests+branch%3Amain)
[![codecov](https://codecov.io/gh/ghsanti/keras-tuner/branch/main/graph/badge.svg)](https://codecov.io/gh/ghsanti/keras-tuner)
![pyright](https://img.shields.io/badge/pyright-blue)
![ruff](https://img.shields.io/badge/ruff-orange)
![pre-commit](https://img.shields.io/badge/pre_commit-green)

[![jax](https://img.shields.io/badge/jax-blue)](https://github.com/jax-ml/jax)
[![tf](https://img.shields.io/badge/tensorflow-yellow)](https://github.com/tensorflow/tensorflow)
[![pytorch](https://img.shields.io/badge/pytorch-orange)](https://github.com/pytorch/pytorch)

Personal fork of the great [KerasTuner: Original Repo](https://github.com/keras-team/keras-tuner).

## Install

Check out to a codespace, or install locally:

```bash
pip install git+https://github.com/ghsanti/keras-tuner
pip install jax[cpu] # or tf, or torch.
```

Try [example.py](https://github.com/ghsanti/keras-tuner/blob/master/example.py) and see the results.

<details>
<summary>
Main Changes
</summary>
- Detailed results and type annotations
</details>

<details>
<summary>
Built-in algorithms:
</summary>
Find the best parameters using the the built-in algorithms:

- Bayesian Optimization,
- Hyperband,
- Random Search

or extend in order to experiment with new search algorithms.

</details>

<details>
<summary>
Code Example
</summary>

```python
import keras_tuner
import keras
def build_model(hp):
  model = keras.Sequential()
  model.add(keras.layers.Dense(
      hp.Choice('units', [8, 16, 32]),
      activation='relu'))
  model.add(keras.layers.Dense(1))
  model.compile(loss='mse')
  return model

tuner = keras_tuner.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5 # tries with the same parameters.
  )

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
best_model = tuner.get_best_models()[0]
```

</details>

- [Starter guide](https://keras.io/guides/keras_tuner/getting_started/).

## Contributing Guide

Please refer to the [CONTRIBUTING.md](https://github.com/ghsanti/keras-tuner/blob/master/CONTRIBUTING.md) for the contributing guide.
