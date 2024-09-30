# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interfaces to manage metrics.

* To and from config
* To and from proto
* Utilities.

The way protos work is that each python class maps (roughly) to a proto class.

"""

# Comment about Typing
# Note that type["ClassName"] is the class object itself.
# and "ClassName" is the instance.

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import keras
import numpy as np
from keras.api.losses import Loss
from keras.api.metrics import Metric

from keras_tuner.protos import keras_tuner_pb2 as proto
from keras_tuner.utils import to_list

if TYPE_CHECKING:
    from keras_tuner.types import (
        _FloatList,
        _FloatListOrFloat,
        _KerasMetric,
        _MetricDirection,
        _MetricHistoryConfig,
        _MetricStats,
        _MetricTrackerConfig,
    )
else:
    _FloatList = Any
    _FloatListOrFloat = Any
    _MetricDirection = Any
    _MetricHistoryConfig = Any
    _MetricStats = Any
    _MetricTrackerConfig = Any
    _KerasMetric = Any


class ExecutionMetric:
    """Metric value at a given execution.

    It aggregates multiple steps using `append`.

    Args:
        value: The evaluated metric values.

    """

    def __init__(
        self,
        value: _FloatListOrFloat,
    ):
        self.value: _FloatList = to_list(value)

    def append(self, value: _FloatListOrFloat) -> None:
        """Extend list of results with a value(s)."""
        self.value.extend(to_list(value))

    def __eq__(self, other: object) -> bool:
        """Check if objects have the same value or not."""
        return (
            other.value == self.value
            if isinstance(other, ExecutionMetric)
            else False
        )

    def __repr__(self) -> str:
        """Stringify class instance."""
        return f"ExecutionMetric(value={self.value})"

    def to_proto(self) -> object:
        """Pass value to proto."""
        return proto.ExecutionMetric(value=self.value)

    @classmethod
    def from_proto(
        cls: type["ExecutionMetric"], proto: object
    ) -> "ExecutionMetric":
        """Class instance from proto value."""
        return cls(value=proto.value)


class MetricHistory:
    """Handle executions of a single metric.

    It contains a collection of `ExecutionMetric` instances.

    Args:
        direction: String. The direction of the metric to optimize. The value
            should be "min" or "max".

    """

    def __init__(
        self,
        direction: _MetricDirection = "min",
        executions: "list[ExecutionMetric] | None" = None,
    ):
        if direction not in {"min", "max"}:
            msg = f"`direction` should be one of min|max, but got: {direction}"
            raise ValueError(msg)
        self.direction = direction
        # used for quick comparison.
        self._executions: list[ExecutionMetric] = executions or []

    def get_executions_values(self) -> list[_FloatList]:
        """Return the values of all executions for this metric."""
        if len(self._executions) > 0:
            return [execution.value for execution in self._executions]
        return []

    def get_last_execution_values(self) -> _FloatList | None:
        """Get the last execution values (not a single value)."""
        values = self.get_executions_values()
        if isinstance(values, list) and len(values) != 0:
            return values[-1]
        return None

    def append_execution_from_values(self, value: _FloatList) -> None:
        """Append ExecutionMetric instance with `value` to list of results."""
        values = [float(v) for v in value]
        self._executions.append(ExecutionMetric(values))

    def get_best_overall_value(self) -> float | None:
        """Best overall value on this metric's history.

        Returns:
            Best of the last values or None.

        """
        all_values = self.get_executions_values()

        if not all_values:  # won't be best neither.
            return None

        reduce_fn = np.nanmin if self.direction == "min" else np.nanmax

        return float(reduce_fn(all_values, axis=(0, 1)))

    def get_best_location(
        self, best_value: float | None = None
    ) -> tuple[int, int] | None:
        """Get the location or (exec, epoch) tuple of the best value.

        Args:
        best_value: Optional float. Otherwise it calculates the best. Before
        finding the index.

        """
        best_value = (
            best_value
            if best_value is not None
            else self.get_best_overall_value()
        )

        if best_value is None:
            return None

        for exec_idx, values in enumerate(self.get_executions_values()):
            for val_idx, value in enumerate(values):
                if value == best_value:
                    return (exec_idx, val_idx)
        msg = "Best value was not found so the indices can't be determined."
        raise ValueError(msg)

    def get_history(self) -> list[ExecutionMetric]:
        """List of ExecutionMetric for this Metric."""
        return self._executions

    def set_history(self, vals: list[ExecutionMetric]) -> None:
        """Set list of ExecutionMetrics."""
        self._executions = vals

    def set_history_from_values(self, vals: list[_FloatList]) -> None:
        """Set list of ExecutionMetrics from execution values."""
        self._executions = []
        for item in vals:
            self._executions.append(ExecutionMetric(item))

    def get_statistics(self) -> _MetricStats | None:
        """Get the summary statistics of executions for this Metric."""
        values = self.get_executions_values()
        if len(values) != 0:
            return {
                "min": float(np.nanmin(values, (0, 1))),
                "max": float(np.nanmax(values, (0, 1))),
                "mean": float(np.nanmean(values, (1))),
                "median": float(np.nanmedian(values, (1))),
                "var": float(np.nanvar(values, (1))),
                "std": float(np.nanstd(values, (1))),
            }
        return None

    def get_config(self) -> _MetricHistoryConfig:
        """Make dict with directions and executions values."""
        return {
            "direction": cast(_MetricDirection, self.direction),
            "executions": [obs.value for obs in self.get_history()],
        }

    @classmethod
    def from_config(
        cls: type["MetricHistory"], config: _MetricHistoryConfig
    ) -> "MetricHistory":
        """Create a MetricHistory from a configuration dictionary."""
        instance = cls(config["direction"])
        instance.set_history_from_values(config["executions"])
        return instance

    def to_proto(self) -> object:
        """Create a 'Mh' protobuffer from a MetricHistory instance (self)."""
        Mh = proto.MetricHistory  # type: ignore  # noqa: PGH003
        # must match order and name in proto.
        return Mh(
            executions=[obs.to_proto() for obs in self.get_history()],
            direction=self.direction,
        )

    @classmethod
    def from_proto(
        cls: type["MetricHistory"], proto: object
    ) -> "MetricHistory":
        """Create MetricHistory instance from proto."""
        if hasattr(proto, "direction"):
            direction = "max" if proto.direction == "max" else "min"
            class_execs = [ExecutionMetric(v.value) for v in proto.executions]
            return cls(direction=direction, executions=class_execs)  # type: ignore  # noqa: PGH003

        msg = "Both 'direction' and 'executions' must be defined in proto."
        raise TypeError(msg)


_MetricNameToHistory = dict[str, "MetricHistory"]
_MetricsTrackerInput = _KerasMetric | Callable | _MetricNameToHistory
_MetricsTrackerInputs = list[_MetricsTrackerInput]


class MetricsTracker:
    """Record of the values of multiple executions of all metrics.

    Stores `MetricHistory` instances for all the metrics.

    Args:
        to_register: List of metrics dicts.

    """

    def __init__(self, to_register: _MetricsTrackerInputs | None = None):
        self.metrics: _MetricNameToHistory = {}
        self.register_metrics(to_register)

    def exists(self, name: str) -> bool:
        """Check whether the metric with `name` exists."""
        return name in self.metrics

    def register_metrics(
        self, to_register: _MetricsTrackerInputs | None
    ) -> None:
        """Register each as MetricHistory if not already there.

        Args:
            to_register: list of metric dicts.

        """
        to_register = to_register or []
        for metric in to_register:
            is_registered = False
            if isinstance(metric, dict):
                for name, history in metric.items():
                    if isinstance(history, MetricHistory):
                        self.register(metric, history.direction)
                        self.set_history(name, history.get_history())
                        is_registered = True
            if not is_registered:
                self.register(metric)

    def register(
        self,
        metric: _MetricsTrackerInput | str,
        direction: str | None = None,
    ) -> None:
        """Register a metric in the metrics dictionary.

        Args:
            metric_name: name of the metric.
            direction: Optionally use a direction.

        """
        metric_name = get_string_metric_name(metric)
        if not isinstance(metric_name, str):
            msg = f"expected `metric_name:str` but found {type(metric_name)}"
            raise TypeError(msg)
        if self.exists(metric_name):
            msg = f"Metric already exists: {metric_name}"
            raise ValueError(msg)
        if direction is None:
            direction = infer_metric_direction(metric_name)
        if direction is None:
            # Objective direction is handled separately, but
            # non-objective direction defaults to min.
            direction = "min"
        direction = cast(_MetricDirection, direction)
        if isinstance(metric, str):
            self.metrics[metric] = MetricHistory(direction)
        else:
            self.metrics[metric_name] = MetricHistory(direction)

    def get_direction(self, metric_name: str) -> _MetricDirection:
        """Get the statistics for a chosen metric."""
        self._assert_exists(metric_name)

        return cast(_MetricDirection, self.metrics[metric_name].direction)

    def get_best_overall_value(self, metric_name: str) -> float | None:
        """Get the single best value of a chosen metric."""
        self._assert_exists(metric_name)
        return self.metrics[metric_name].get_best_overall_value()

    def get_best_overall_value_location(
        self, metric_name: str
    ) -> tuple[int, int] | None:
        """Get the location of the single best value for a chosen metric."""
        self._assert_exists(metric_name)
        return self.metrics[metric_name].get_best_location()

    def get_last_execution_values(self, metric_name: str) -> _FloatList | None:
        """Get the last execution **values** for a chosen metric."""
        self._assert_exists(metric_name)
        return self.metrics[metric_name].get_last_execution_values()

    def get_statistics(self, metric_name: str) -> _MetricStats | None:
        """Get the statistics for a chosen metric."""
        self._assert_exists(metric_name)
        return self.metrics[metric_name].get_statistics()

    def append_execution_value(
        self, metric_name: str, value: _FloatListOrFloat
    ) -> None:
        """Append ExecutionMetric to a specific Metric, from its values."""
        value = (
            [float(v) for v in value]
            if isinstance(value, list)
            else [float(value)]
        )
        if not self.exists(metric_name):
            self.register(metric_name)

        self.metrics[metric_name].append_execution_from_values(value)

    def set_history_from_values(
        self, metric_name: str, value: list[_FloatList]
    ) -> None:
        """Create and append ExecutionMetric to a specific Metric."""
        if not self.exists(metric_name):
            self.register(metric_name)
        for v in value:
            if not isinstance(v, list):
                msg = "Expected nested list but got {type(v)}"
                raise TypeError(msg)

        self.metrics[metric_name].set_history_from_values(value)

    def set_history(
        self, metric_name: str, value: list[ExecutionMetric]
    ) -> None:
        """Append ExecutionMetric to a specific Metric, from its values."""
        if not self.exists(metric_name):
            self.register(metric_name)
        for v in value:
            if not isinstance(v, ExecutionMetric):
                msg = "Expected nested list but got {type(v)}"
                raise TypeError(msg)

        self.metrics[metric_name].set_history(value)

    def get_history(self, metric_name: str) -> list[ExecutionMetric]:
        """Get the history of a chosen metric."""
        self._assert_exists(metric_name)
        return self.metrics[metric_name].get_history()

    def get_config(self) -> _MetricTrackerConfig:
        """Get dictionary of metric names to MetricHistoryConfig data."""
        return {
            name: metric_history.get_config()
            for name, metric_history in self.metrics.items()
        }

    @classmethod
    def from_config(
        cls: type["MetricsTracker"], config: _MetricTrackerConfig
    ) -> "MetricsTracker":
        """Make MetricsTracker Instance from a configuration dictionary."""
        metrics: _MetricsTrackerInputs = [
            {name: MetricHistory.from_config(metric_history)}
            for name, metric_history in config.items()
        ]
        return cls(to_register=metrics)

    #  Note that protobuffers use "metrics" instead of "to_register".
    def to_proto(self) -> object:
        """Create proto from MetricsTracker instance."""
        Mt = proto.MetricsTracker
        return Mt(
            metrics={
                name: self.metrics[name].to_proto()
                for name in list(self.metrics.keys())
            }
        )

    @classmethod
    def from_proto(cls, proto: object) -> "MetricsTracker":
        """Create a MetricsTracker instance from a proto."""
        metrics: _MetricsTrackerInputs = [
            {name: MetricHistory.from_proto(proto.metrics[name])}
            for name in list(proto.metrics.keys())
        ]
        return cls(to_register=metrics)

    def _assert_exists(self, name: str) -> None:
        """Ensure that name is a metric."""
        if name not in self.metrics:
            msg = f"Unknown metric: {name}"
            raise ValueError(msg)


_MAX_METRICS = (
    "Accuracy",
    "BinaryAccuracy",
    "CategoricalAccuracy",
    "SparseCategoricalAccuracy",
    "TopKCategoricalAccuracy",
    "SparseTopKCategoricalAccuracy",
    "TruePositives",
    "TrueNegatives",
    "Precision",
    "Recall",
    "AUC",
    "SensitivityAtSpecificity",
    "SpecificityAtSensitivity",
)

_MAX_METRIC_FNS = (
    "accuracy",
    "categorical_accuracy",
    "binary_accuracy",
    "sparse_categorical_accuracy",
)


def infer_metric_direction(metric: str) -> _MetricDirection | None:
    """Infer max or min based on the name of the metric."""
    if isinstance(metric, str):  # case 1: string name
        metric_name = metric

        if metric_name.startswith("val_"):
            metric_name = metric_name.replace("val_", "", 1)

        if metric_name.startswith("weighted_"):
            metric_name = metric_name.replace("weighted_", "", 1)

        # Special-cases (from `keras/engine/training_utils.py`)
        if metric_name in {"loss", "crossentropy", "ce"}:
            return "min"
        if metric_name == "acc":
            return "max"

        try:
            metric = keras.metrics.deserialize(metric_name)
        except ValueError:
            try:
                metric = keras.losses.deserialize(metric_name)
            except:
                return None

    # Metric class, Loss class, or function.
    name = ""
    if isinstance(metric, Loss | Metric):
        name = metric.__class__.__name__
        if name == "MeanMetricWrapper":
            name = metric._fn.__name__  # type: ignore  # noqa: PGH003, SLF001
    elif isinstance(metric, Callable):
        name = metric.__name__

    if name in _MAX_METRICS or name in _MAX_METRIC_FNS:
        return "max"
    if hasattr(keras.metrics, name) or hasattr(keras.losses, name):
        return "min"

    # Direction can't be inferred.
    return None


def get_string_metric_name(metric: _MetricsTrackerInput | str) -> str:
    """Return the name of the metric."""
    name = metric if isinstance(metric, str) else None

    if isinstance(metric, dict):  # case 1: string name
        name = next(iter(metric.keys()))

    if name is not None:
        if name.startswith("val_"):
            name = name.replace("val_", "", 1)

        if name.startswith("weighted_"):
            name = name.replace("weighted_", "", 1)

    elif isinstance(metric, Loss | Metric):  # case 2: Loss, Metric instance.
        name = metric.__class__.__name__
        if name == "MeanMetricWrapper":
            name = metric._fn.__name__  # type: ignore  # noqa: PGH003, SLF001
    elif isinstance(metric, Callable):  # case 3: Callable function.
        # binary_crossentropy, sparse_categorical_crossentropy
        name = metric.__name__
    else:
        msg = "Invalid metric input."
        raise TypeError(msg)
    return name
