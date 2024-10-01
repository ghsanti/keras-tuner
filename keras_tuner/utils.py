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
"""KerasTuner utilities."""

import json
from typing import TYPE_CHECKING

import keras

from keras_tuner.backend import io

if TYPE_CHECKING:
    from keras_tuner.types import _NumberValues
else:
    _NumberValues = None

# Check if we are in a ipython/colab environement
try:
    import IPython

    class_name = IPython.get_ipython().__class__.__name__
    IS_NOTEBOOK = "Terminal" not in class_name
except (NameError, ImportError):
    IS_NOTEBOOK = False


if IS_NOTEBOOK:
    from IPython import display


def try_clear() -> None:
    if IS_NOTEBOOK:
        display.clear_output()
    else:
        print()


def create_directory(path: str, remove_existing: bool = False) -> None:
    """Create the directory if it doesn't exist."""
    if not io.exists(path):
        io.makedirs(path)

    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        io.rmtree(path)
        io.makedirs(path)


def serialize_keras_object(obj):
    return keras.utils.serialize_keras_object(obj)


def deserialize_keras_object(config, module_objects=None, custom_objects=None):
    return keras.utils.deserialize_keras_object(
        config, custom_objects, module_objects
    )


def save_json(path: str, obj: object) -> str:
    """Save Python object to a json file.

    Args:
        path: The path to the json file.
        obj: The Python object to be saved.

    Returns:
        The json string format of the object.

    """
    obj_str = json.dumps(obj)
    with io.File(path, "w") as f:
        f.write(obj_str)
    return obj_str


def load_json(path: str) -> object:
    """Load json from file.

    Args:
        path: String. The path to the json file.

    Returns:
        A Python object.

    """
    with io.File(path, "r") as f:
        obj_str = f.read()
    return json.loads(obj_str)


def to_list(values: _NumberValues | tuple | list) -> list:
    """Get tuple or numeric value into list. Lists are left unchanged."""
    if isinstance(values, int | float):
        return [values]
    if isinstance(values, list | tuple):
        return list(values)  # type: ignore  # noqa: PGH003

    msg = f"Can not convert values of type {type(values)} to list."
    raise TypeError(msg) from None
