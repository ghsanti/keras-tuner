"""Application's common types."""

from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    import keras
    from keras.api.losses import Loss
    from keras.api.metrics import Metric

    from keras_tuner.engine.metrics_tracking import MetricHistory

    from .engine.objective import (
        DefaultObjective,
        MultiObjective,
        Objective,
    )
else:
    Loss = Any
    Metric = Any
    MultiObjective = Any
    Objective = Any
    DefaultObjective = Any
    keras = Any
    _FloatList = Any
    _FloatListOrFloat = Any
    _NumberValues = Any
    _WhichExecutionValues = Any
    _MetricStats = Any
    _MetricDirection = Any
    _MetricValues = Any
    _MetricNameToHistory = Any
    _MetricsTrackerInput = Any
    _MetricsTrackerInputs = Any
    _MetricHistoryConfig = Any
    _MetricTrackerConfig = Any
    _Verbose = Any
    _EpochLogs = Any
    _Model = Any
    _KerasMetric = Any
    _SomeObjectiveOrName = Any
    HyperModel = Any
    HyperParameters = Any
    _AllObjectives = Any
    _FloatList = Any
    _FloatListOrFloat = Any
    _NumberValues = Any
    _WhichExecutionValues = Any


# KERAS specific
_Model = keras.models.Model | keras.models.Sequential
_KerasMetric = Loss | Metric
_kerasLoss = keras.Loss | str
_kerasMetric = keras.Metric | str
_kerasOptimizer = str | keras.Optimizer

# basic types
_FloatList = list[float]
_FloatListOrFloat = float | _FloatList
_NumberValues = int | float


# Metrics types
_WhichExecutionValues = Literal["all", "best", "last"]


class _MetricStats(TypedDict):
    min: float
    max: float
    std: float
    mean: float
    var: float
    median: float


_MetricDirection = Literal["min", "max"]
_MetricValues = list[_FloatList]


_MetricNameToHistory = dict[str, "MetricHistory"]
_MetricsTrackerInput = _KerasMetric | Callable | _MetricNameToHistory
_MetricsTrackerInputs = list[_MetricsTrackerInput]


class _MetricHistoryConfig(TypedDict):
    direction: _MetricDirection
    executions: list[_FloatList]


_MetricTrackerConfig = dict[str, _MetricHistoryConfig]
_Verbose = Literal["auto"] | int

_EpochLogs = dict[str, float]


# Main Output type.
_TrialResult = _NumberValues | dict | keras.callbacks.History
_SupportedTrialResults = list[_TrialResult] | _TrialResult


_SomeObjective = DefaultObjective | MultiObjective | Objective
_SomeObjectiveOrName = _SomeObjective | str

_AllObjectives = _SomeObjectiveOrName | list[_SomeObjectiveOrName]
