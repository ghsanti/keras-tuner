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

from typing import TYPE_CHECKING, Any, cast

from keras_tuner.engine import metrics_tracking

if TYPE_CHECKING:
    from keras_tuner.types import (
        _EpochLogs,
        _MetricDirection,
        _SomeObjective,
        _SomeObjectiveOrName,
    )
else:
    _EpochLogs = Any
    _SomeObjective = Any
    _SomeObjectiveOrName = Any
    _MetricDirection = Any


class Objective:
    """The objective for optimization during tuning.

    Args:
        name: String. The name of the objective.
        direction: String. The value should be "min" or "max" indicating
            whether the objective value should be minimized or maximized.

    """

    def __init__(self, name: str, direction: _MetricDirection) -> None:
        self.name = name
        self.direction = cast(_MetricDirection, direction)

    def has_value(self, logs: _EpochLogs) -> bool:
        """Check if objective value exists in logs.

        Args:
            logs: metric_name to metric_value dict.

        Returns:
            Is objective in log.

        """
        return self.name in logs

    def get_value(self, logs: _EpochLogs) -> float:
        """Get the objective value from the metrics logs.

        Args:
            logs: metric_name to metric_value dict.

        Returns:
            The objective value.

        """
        return logs[self.name]

    def better_than(self, new_val: float, reference: float) -> bool:
        """Check whether `new_val` is better than `reference`.

        Returns:
            Whether the new_val is an improvement over reference.

        """
        return (new_val > reference and self.direction == "max") or (
            new_val < reference and self.direction == "min"
        )

    def __eq__(self, obj: object) -> bool:
        """Check if `obj` has the same name and direction, and class."""
        if isinstance(obj, Objective | DefaultObjective):
            return self.name == obj.name and self.direction == obj.direction
        return False

    def __str__(self) -> str:
        """Provide a human-readable string for when a user prints the class."""
        return f'Objective(name="{self.name}", direction="{self.direction}")'


class DefaultObjective(Objective):
    """Default objective to minimize if not provided by the user."""

    def __init__(self) -> None:
        super().__init__(name="default_objective", direction="min")


class MultiObjective(Objective):
    """A container for a list of objectives.

    Args:
        objectives: A list of `Objective`s.

    """

    def __init__(self, objectives: list[Objective]):
        super().__init__(name="multi_objective", direction="min")
        self.objectives = objectives
        self.name_to_direction = {
            objective.name: objective.direction for objective in self.objectives
        }

    def has_value(self, logs: _EpochLogs) -> bool:
        """Check whether all objectives have a log."""
        return all(key in logs for key in self.name_to_direction)

    def get_value(self, logs: _EpochLogs) -> float:
        """Reduce metrics values to single value."""
        obj_value = 0
        for metric_name, metric_value in logs.items():
            if isinstance(metric_value, list):
                msg = "Metric value must be a number."
                raise TypeError(msg)
            if metric_name not in self.name_to_direction:
                continue
            if self.name_to_direction[metric_name] == "min":
                obj_value += metric_value
            else:
                obj_value -= metric_value
        return obj_value

    def __eq__(self, obj):
        if self.name_to_direction.keys() != obj.name_to_direction.keys():
            return False
        return sorted(self.objectives, key=lambda x: x.name) == sorted(
            obj.objectives, key=lambda x: x.name
        )

    def __str__(self):
        return (
            "Multi"
            + super().__str__()
            + f": [{', '.join(map(lambda x: str(x), self.objectives))}]"
        )


def create_objective(
    objective: list[_SomeObjectiveOrName] | _SomeObjectiveOrName | None,
) -> _SomeObjective:
    """Create an objective class given any of the possibilities."""
    if objective is None:
        return DefaultObjective()
    if isinstance(objective, list):
        return MultiObjective([create_objective(obj) for obj in objective])
    if isinstance(objective, Objective):
        return objective
    if not isinstance(objective, str):  # check for users and debugging.
        msg = (
            "`objective` not understood, expected str or "
            f"`Objective` object, found: {objective}"
        )
        raise TypeError(msg)

    # try to infer direction using string name.
    direction = metrics_tracking.infer_metric_direction(objective)
    if direction is None:
        error_msg = (
            'Could not infer optimization direction ("min" or "max") '
            'for unknown metric "{obj}". Please specify the objective  as'
            "a `keras_tuner.Objective`, for example `keras_tuner.Objective("
            '"{obj}", direction="min")`.'
        )
        error_msg = error_msg.format(obj=objective)
        raise ValueError(error_msg)
    return Objective(name=objective, direction=direction)
