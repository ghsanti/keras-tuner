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

import pytest

from . import Boolean


def test_boolean():
    # Test default default
    boolean = Boolean("bool")
    assert boolean.default is False
    # Test default setting
    boolean = Boolean("bool", default=True)
    assert boolean.default is True
    # Wrong default type
    with pytest.raises(ValueError, match="must be a Python boolean"):
        Boolean("bool", default=None)
    # Test serialization
    boolean = Boolean("bool", default=True)
    boolean = Boolean.from_config(boolean.get_config())
    assert boolean.default is True
    assert boolean.name == "bool"

    # Test random_sample
    assert boolean.random_sample() in {True, False}
    assert boolean.random_sample(123) == boolean.random_sample(123)
    assert {boolean.value_to_prob(True), boolean.value_to_prob(False)} == {
        0.25,
        0.75,
    }


def test_boolean_repr():
    assert repr(Boolean("bool")) == repr(Boolean("bool"))


def test_boolean_values_property():
    assert list(Boolean("bool").values) == [True, False]
