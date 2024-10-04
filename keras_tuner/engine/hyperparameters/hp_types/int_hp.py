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


from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.engine.hyperparameters import hp_utils
from keras_tuner.protos import keras_tuner_pb2 as protos

from . import numerical


class Int(numerical.Numerical):
    """Integer hyperparameter.

    Note that unlike Python's `range` function, `max_value` is *included* in
    the possible values this parameter can take on.


    Example #1:

    ```py
    hp.Int("n_layers", min_value=6, max_value=12)
    ```

    The possible values are [6, 7, 8, 9, 10, 11, 12].

    Example #2:

    ```py
    hp.Int("n_layers", min_value=6, max_value=13, step=3)
    ```

    `step` is the minimum distance between samples.
    The possible values are [6, 9, 12].

    Example #3:

    ```py
    hp.Int("batch_size", min_value=2, max_value=32, step=2, sampling="log")
    ```

    When `sampling="log"` the `step` is multiplied between samples.
    The possible values are [2, 4, 8, 16, 32].

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        min_value: Integer, the lower limit of range, inclusive.
        max_value: Integer, the upper limit of range, inclusive.
        step: Optional integer, the distance between two consecutive samples in
            the range. If left unspecified, it is possible to sample any
            integers in the interval. If `sampling="linear"`, it will be the
            minimum additve between two samples. If `sampling="log"`, it will be
            the minimum multiplier between two samples.
        sampling: String. One of "linear", "log", "reverse_log". Defaults to
            "linear". When sampling value, it always start from a value in range
            [0.0, 1.0). The `sampling` argument decides how the value is
            projected into the range of [min_value, max_value].
            "linear": min_value + value * (max_value - min_value)
            "log": min_value * (max_value / min_value) ^ value
            "reverse_log":
                max_value - min_value * ((max_value/min_value)^(1-value) - 1)
        default: Integer, default value to return for the parameter. If
            unspecified, the default value will be `min_value`.

    """

    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        *,
        step: int | None = None,
        sampling: str = "linear",
        default: int | None = None,
        **kwargs,
    ) -> None:
        if step is None and sampling == "linear":
            step = 1
        super().__init__(
            name=name,
            min_value=int(min_value),
            max_value=int(max_value),
            step=step,
            sampling=sampling,
            default=default,
            **kwargs,
        )

        if not isinstance(min_value, int) or not isinstance(max_value, int):
            msg = "must be an int."
            raise TypeError(msg)

    def __repr__(self) -> str:
        """Representation of the class instance with values."""
        return (
            f"Int(name: '{self.name}', min_value: {self.min_value}, "
            f"max_value: {self.max_value}, step: {self.step}, "
            f"sampling: {self.sampling}, default: {self.default})"
        )

    def prob_to_value(self, prob: float) -> int | None:
        if self.step is None:
            # prob is in range [0.0, 1.0), use max_value + 1 so that
            # max_value may be sampled.
            result = self._sample_numerical_value(prob, self.max_value + 1)
            return int(result) if result is not None else None
        result = self._sample_with_step(prob)
        return int(result) if result is not None else None

    def value_to_prob(self, value: float) -> float | None:
        if self.step is None:
            return self._numerical_to_prob(
                # + 0.5 to center the prob
                value + 0.5,
                # + 1 to include the max_value
                self.max_value + 1,
            )
        return self._to_prob_with_step(value)

    @property
    def default(self) -> int:
        """Default value for the parameter."""
        if self._default is not None and not isinstance(
            self._default, int | float
        ):
            msg = f"""Default value for Int must be an integer or None.
            Found type: {type(self._default)}"""
            raise TypeError(msg)
        return int(
            self._default if self._default is not None else self.min_value
        )

    def get_config(self):
        config = super().get_config()
        config["min_value"] = self.min_value
        config["max_value"] = self.max_value
        config["step"] = self.step
        config["sampling"] = self.sampling
        config["default"] = self._default
        return config

    @classmethod
    def from_proto(cls, proto):
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(
            name=proto.name,
            min_value=proto.min_value,
            max_value=proto.max_value,
            step=proto.step or None,
            sampling=hp_utils.sampling_from_proto(proto.sampling),
            default=proto.default,
            conditions=conditions,
        )

    def to_proto(self):
        return protos.Int(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step if self.step is not None else 0,
            sampling=hp_utils.sampling_to_proto(self.sampling),
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )
