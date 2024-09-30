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

from keras import utils

# we re-export to make them available further up the chain.
from keras_tuner.engine.hyperparameters.hp_types import (
    Boolean,
    Choice,
    Fixed,
    Float,
    Int,
)

from . import hp_types
from .HyperParameter import HyperParameter
from .HyperParameters import HyperParameters

OBJECTS = [*hp_types.OBJECTS, HyperParameter, HyperParameters]

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


def deserialize(config: dict[str, object]) -> object:
    """Turn the json-dictionary into class instance."""
    return utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


def serialize(obj: object) -> object:
    """Turn the class instance into a json-dictionary."""
    return utils.serialize_keras_object(obj)
