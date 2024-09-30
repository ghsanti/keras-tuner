"""Application's common types."""

from typing import Literal, TypedDict

import keras
import numpy as np
from keras.api.losses import Loss
from keras.api.metrics import Metric

from .engine.objective import (
    DefaultObjective,
    MultiObjective,
    Objective,
)

# basic types
_FloatList = list[float]
_FloatListOrFloat = float | _FloatList
_NumberValues = int | float | np.floating


# Metrics types
class _MetricStats(TypedDict):
    min: float
    max: float
    std: float
    mean: float
    var: float
    median: float


_MetricDirection = Literal["min", "max"]


class _MetricHistoryConfig(TypedDict):
    direction: _MetricDirection
    executions: list[_FloatList]


_MetricTrackerConfig = dict[str, _MetricHistoryConfig]
_Verbose = Literal["auto"] | int

_EpochLogs = dict[str, float]


# Main Output type.
_TrialResult = _NumberValues | dict | keras.callbacks.History
_SupportedTrialResults = list[_TrialResult] | _TrialResult


# KERAS specific
_Model = keras.models.Model | keras.models.Sequential
_KerasMetric = Loss | Metric
_kerasLoss = keras.Loss | str
_kerasMetric = keras.Metric | str
_kerasOptimizer = str | keras.Optimizer

_SomeObjective = DefaultObjective | MultiObjective | Objective
_SomeObjectiveOrName = _SomeObjective | str

_AllObjectives = _SomeObjectiveOrName | list[_SomeObjectiveOrName]
