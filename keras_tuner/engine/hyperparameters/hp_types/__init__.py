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

# we export the modules' main functions so that they can be imported from
# hp_types instead.
from .boolean_hp import Boolean
from .choice_hp import Choice
from .fixed_hp import Fixed
from .float_hp import Float
from .int_hp import Int

OBJECTS = (
    Fixed,
    Float,
    Int,
    Choice,
    Boolean,
)

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


def deserialize(config):
    return utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


def serialize(obj):
    return utils.serialize_keras_object(obj)
