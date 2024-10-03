"""Application's common types."""

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypedDict,
)

import keras
from keras.api.losses import Loss
from keras.api.metrics import Metric

if TYPE_CHECKING:
    from .engine.metrics_tracking import MetricHistory
    from .engine.objective import (
        DefaultObjective,
        MultiObjective,
        Objective,
    )
else:
    MetricHistory = Any
    DefaultObjective = Any
    MultiObjective = Any
    Objective = Any


# ===============================Custom Types=====

# All the types can be imported without blocking.
# Since they are type aliases.

# KERAS
_Model: TypeAlias = keras.models.Model | keras.models.Sequential
_KerasMetric: TypeAlias = Loss | Metric
_kerasLoss: TypeAlias = keras.Loss | str
_kerasMetric: TypeAlias = keras.Metric | str
_kerasOptimizer: TypeAlias = str | keras.Optimizer

# Basic
_FloatList: TypeAlias = list[float]
_FloatListOrFloat: TypeAlias = float | _FloatList
_NumberValues: TypeAlias = int | float


# Metrics
class _MetricStats(TypedDict):
    min: float
    max: float
    std: float
    mean: float
    var: float
    median: float


_MetricDirection: TypeAlias = Literal["min", "max"]
_MetricValues: TypeAlias = list[_FloatList]


_MetricNameToHistory: TypeAlias = dict[str, "MetricHistory"]
_MetricsTrackerInput: TypeAlias = _KerasMetric | Callable | _MetricNameToHistory
_MetricsTrackerInputs: TypeAlias = list[_MetricsTrackerInput]


class _MetricHistoryConfig(TypedDict):
    direction: _MetricDirection
    executions: list[_FloatList]


_MetricTrackerConfig: TypeAlias = dict[str, _MetricHistoryConfig]
_Verbose: TypeAlias = Literal["auto"] | int

_EpochLogs: TypeAlias = dict[str, float]


# Main Output type.
_TrialResult: TypeAlias = _NumberValues | dict | keras.callbacks.History
_SupportedTrialResults: TypeAlias = list[_TrialResult] | _TrialResult


_SomeObjective: TypeAlias = DefaultObjective | MultiObjective | Objective
_SomeObjectiveOrName: TypeAlias = _SomeObjective | str

_AllObjectives: TypeAlias = _SomeObjectiveOrName | list[_SomeObjectiveOrName]

# ============HP Types============================


_ConditionValues: TypeAlias = list[int] | list[float] | list[bool] | list[str]
