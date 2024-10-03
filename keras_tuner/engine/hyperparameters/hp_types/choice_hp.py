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
from keras_tuner.engine.hyperparameters.HyperParameter import HyperParameter
from keras_tuner.protos import keras_tuner_pb2 as protos
from keras_tuner.types import _ConditionValues


class Choice(HyperParameter):
    """Choice of one value among a predefined set of possible values.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        values: A list of possible values. Values must be int, float,
            str, or bool. All values must be of the same type.
        ordered: Optional boolean, whether the values passed should be
            considered to have an ordering. Defaults to `True` for float/int
            values.  Must be `False` for any other values.
        default: Optional default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.

    """

    def __init__(
        self,
        name,
        values: _ConditionValues,
        ordered=None,
        default=None,
        **kwargs,
    ):
        super().__init__(name=name, default=default, **kwargs)
        if not values:
            msg = "`values` must be provided for `Choice`."
            raise ValueError(msg)

        # Type checking.
        types = {type(v) for v in values}
        if len(types) > 1:
            msg = (
                "A `Choice` can contain only one type of value, "
                f"found values: {values!s} with types {types}."
            )
            raise TypeError(msg)

        # Standardize on str, int, float, bool.
        if isinstance(values[0], str):
            values = [str(v) for v in values]
            if default is not None:
                default = str(default)
        elif isinstance(values[0], int):
            values = [int(v) for v in values]
            if default is not None:
                default = int(default)
        elif not isinstance(values[0], (bool | float)):
            raise TypeError(
                "A `Choice` can contain only `int`, `float`, `str`, or "
                "`bool`, found values: " + str(values) + "with "
                "types: " + str(type(values[0]))
            )
        self._values = values

        if default is not None and default not in values:
            msg = (
                "The default value should be one of the choices. "
                f"You passed: values={values}, default={default}"
            )
            raise ValueError(msg)
        self._default = default

        # Get or infer ordered.
        self.ordered = ordered
        is_numeric = isinstance(values[0], (int | float))
        if self.ordered and not is_numeric:
            msg = "`ordered` must be `False` for non-numeric types."
            raise ValueError(msg)
        if self.ordered is None:
            self.ordered = is_numeric

    def __repr__(self):
        return f"""Choice(name: '{self.name}', values: {self._values},
        ordered: {self.ordered}, default: {self.default})"""

    @property
    def values(self):
        return self._values

    @property
    def default(self):
        return self._values[0] if self._default is None else self._default

    def prob_to_value(self, prob):
        return self._values[hp_utils.prob_to_index(prob, len(self._values))]

    def value_to_prob(self, value):
        return hp_utils.index_to_prob(
            self._values.index(value), len(self._values)
        )

    def get_config(self):
        config = super().get_config()
        config["values"] = self._values
        config["ordered"] = self.ordered
        return config

    @classmethod
    def from_proto(cls, proto):
        values = [getattr(val, val.WhichOneof("kind")) for val in proto.values]
        default = getattr(proto.default, proto.default.WhichOneof("kind"), None)
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(
            name=proto.name,
            values=values,
            ordered=proto.ordered,
            default=default,
            conditions=conditions,
        )

    def to_proto(self):
        if isinstance(self.values[0], str):
            values = [protos.Value(string_value=v) for v in self.values]
            default = protos.Value(string_value=self.default)
        elif isinstance(self.values[0], int):
            values = [protos.Value(int_value=v) for v in self.values]
            default = protos.Value(int_value=self.default)
        else:
            values = [protos.Value(float_value=v) for v in self.values]
            default = protos.Value(float_value=self.default)
        return protos.Choice(
            name=self.name,
            ordered=self.ordered,
            values=values,
            default=default,
            conditions=[c.to_proto() for c in self.conditions],
        )
