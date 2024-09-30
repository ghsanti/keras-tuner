# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for Tuner class."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import keras
import numpy as np
from keras.api.backend import backend
from keras.api.models import Model, Sequential

from keras_tuner import errors
from keras_tuner.backend import io

from .hyperparameters import (
    Boolean,
    Choice,
    Fixed,
    Float,
    Int,
)
from .objective import (
    DefaultObjective,
)
from .trial import Trial

if TYPE_CHECKING:
    from keras_tuner.types import (
        _EpochLogs,
        _FloatListOrFloat,
        _Model,
        _SomeObjective,
        _SupportedTrialResults,
    )

    from .hyperparameters.HyperParameters import HyperParameters
    from .tuner import Tuner
else:
    _EpochLogs = Any
    _FloatListOrFloat = Any
    _SomeObjective = Any
    _SupportedTrialResults = Any
    HyperParameters = Any
    Tuner = Any
    _Model = Any


class TunerCallback(keras.callbacks.Callback):  # noqa: D101
    def __init__(self, tuner: Tuner, trial: Trial):
        super().__init__()
        self.tuner = tuner
        self.trial = trial

    def on_epoch_begin(  # noqa: D102
        self, epoch: int, logs: _EpochLogs | None = None
    ) -> None:
        self.tuner.on_epoch_begin(self.trial, self.model, epoch, logs=logs)

    def on_batch_begin(  # noqa: D102
        self, batch: int, logs: _EpochLogs | None = None
    ) -> None:
        self.tuner.on_batch_begin(self.trial, self.model, batch, logs)

    def on_batch_end(self, batch: int, logs: _EpochLogs | None = None) -> None:  # noqa: D102
        self.tuner.on_batch_end(self.trial, self.model, batch, logs)

    def on_epoch_end(self, epoch: int, logs: _EpochLogs | None = None) -> None:  # noqa: D102
        self.tuner.on_epoch_end(self.trial, self.model, epoch, logs=logs)


class SaveBestEpoch(keras.callbacks.Callback):
    """A Keras callback to save the model weights at the best epoch.

    Args:
        objective: An `Objective` instance.
        keras_path: String. The file path to save the model weights.

    """

    def __init__(self, objective: _SomeObjective, keras_path: Path) -> None:
        super().__init__()
        self.objective = objective
        self.keras_path = keras_path
        if self.objective.direction == "max":
            self.best_value = float("-inf")
        else:
            self.best_value = float("inf")

    def on_epoch_end(self, epoch: int, logs: _EpochLogs | None = None) -> None:  # noqa: D102
        if logs is None:
            msg = "Logs must be a dictionary but found None."
            raise TypeError(msg)
        if not self.objective.has_value(logs):
            # Save on every epoch if metric value is not in the logs. Either no
            # objective is specified, or objective is computed and returned
            # after `fit()`.
            self._save_model()
            return
        current_value = self.objective.get_value(logs)
        if self.objective.better_than(current_value, self.best_value):
            self.best_value = current_value
            self._save_model()

    def _save_model(self) -> None:
        if self.keras_path is None:
            return

        if not isinstance(self.model, Model | Sequential):
            msg = "Model must be a Model|Functional|Sequential."
            raise TypeError(msg)
        if backend() != "tensorflow":
            self.model.save(self.keras_path, overwrite=True)
            return

        # Create temporary keras model files on non-chief workers.
        write_keras_path = io.write_filepath(
            self.keras_path, self.model.distribute_strategy
        )
        self.model.save(write_keras_path, overwrite=True)
        # Remove temporary keras model files on non-chief workers.
        if write_keras_path is not None:
            io.remove_temp_dir_with_filepath(
                write_keras_path, self.model.distribute_strategy
            )


_Outputs = list[dict[str, _FloatListOrFloat]]


def convert_to_metrics_dict(
    results: _SupportedTrialResults, objective: _SomeObjective
) -> _Outputs:
    """Normalize results to list of dicts."""
    # List of multiple execution results to be averaged.
    # Check this case first to deal each case individually to check for errors.
    results_list = results if isinstance(results, list) else [results]
    formatted_results = []
    for result in results_list:
        # Single value.
        if isinstance(result, (int | float | np.floating)):
            formatted_results.append({objective.name: float(result)})
        elif isinstance(result, dict):
            # A dictionary.
            formatted_results.append(result)
        elif isinstance(result, keras.callbacks.History):
            # A History.
            formatted_results.append(result.history)
        else:
            msg = f"""Results must be number, dict, KerasHistory or list.
            but found type: {type(results)}"""
            raise TypeError(msg)
    return formatted_results


def validate_trial_results(
    results: _SupportedTrialResults, objective: _SomeObjective, func_name: str
) -> None:
    """Quick check before running more expensive computation."""
    if isinstance(results, list):
        for elem in results:
            validate_trial_results(elem, objective, func_name)
        return

    # Single value.
    if isinstance(results, int | float | np.floating):
        return

    # None
    if results is None:  # we should still check this value.
        msg = (
            f"The return value of {func_name} is None. "
            "Did you forget to return the metrics? "
        )
        raise errors.FatalTypeError(msg)

    # objective left unspecified,
    # and objective value is not a single float.
    if isinstance(objective, DefaultObjective) and not (
        isinstance(results, dict) and objective.name in results
    ):
        msg = f"""Expected the return value of {func_name} to be
            a single float when `objective` is left unspecified.
            Received return value: {results} of type {type(results)}."""
        raise errors.FatalTypeError(msg)

    # A dictionary.
    if isinstance(results, dict):
        if objective.name not in results:
            msg = f"""Expected the returned dictionary from {func_name} to have
                the specified objective, {objective.name},
                as one of the keys.
                Received: {results}."""
            raise errors.FatalValueError(msg)
        return

    # A History.
    if isinstance(results, keras.callbacks.History):
        return

    # Other unsupported types that the user may accidentally pass.
    msg = f"""Expected the return value of {func_name} to be
        one of float, dict, keras.callbacks.History,
        or a list of one of these types.
        Recevied return value: {results} of type {type(results)}."""
    raise errors.FatalTypeError(msg)


def convert_hyperparams_to_hparams(hyperparams: HyperParameters, hparams_api):
    """Convert KerasTuner HyperParameters to TensorBoard HParams."""
    hparams = {}
    for hp in hyperparams.space:
        hparams_value = {}
        try:
            hparams_value = hyperparams.get(hp.name)
        except ValueError:
            continue

        hparams_domain = {}
        if isinstance(hp, Choice):
            hparams_domain = hparams_api.Discrete(hp.values)
        elif isinstance(hp, Int):
            if hp.step is not None and hp.step != 1:
                # Note: `hp.max_value` is inclusive, unlike the end index
                # of Python `range()`, which is exclusive
                values = list(range(hp.min_value, hp.max_value + 1, hp.step))
                hparams_domain = hparams_api.Discrete(values)
            else:
                hparams_domain = hparams_api.IntInterval(
                    hp.min_value, hp.max_value
                )
        elif isinstance(hp, Float):
            if hp.step is not None:
                # Note: `hp.max_value` is inclusive, unlike the end index
                # of Numpy's arange(), which is exclusive
                values = np.arange(
                    hp.min_value, hp.max_value + 1e-7, step=hp.step
                ).tolist()
                hparams_domain = hparams_api.Discrete(values)
            else:
                hparams_domain = hparams_api.RealInterval(
                    hp.min_value, hp.max_value
                )
        elif isinstance(hp, Boolean):
            hparams_domain = hparams_api.Discrete([True, False])
        elif isinstance(hp, Fixed):
            hparams_domain = hparams_api.Discrete([hp.value])
        else:
            msg = f"`HyperParameter` type not recognized: {hp}"
            raise TypeError(msg)

        hparams_key = hparams_api.HParam(hp.name, hparams_domain)
        hparams[hparams_key] = hparams_value

    return hparams
