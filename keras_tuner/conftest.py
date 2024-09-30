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

# this a global scope for tests.

import keras
import pytest
from keras.api.backend import backend


@pytest.fixture(autouse=True)
def set_seeds_before_tests() -> None:
    """Test wrapper to set the seed before each test.

    This wrapper runs for all the tests in the test suite.
    """
    seed = 0
    keras.utils.set_random_seed(seed)
    # Use channels_first for torch backend.
    if backend() == "torch":
        keras.backend.set_image_data_format("channels_first")
    else:
        keras.backend.set_image_data_format("channels_last")
