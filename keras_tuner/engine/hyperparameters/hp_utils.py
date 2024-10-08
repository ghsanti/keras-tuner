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

import math

from keras_tuner.protos import keras_tuner_pb2 as protos


def sampling_from_proto(sampling):
    if sampling == protos.Sampling.LINEAR:
        return "linear"
    if sampling == protos.Sampling.LOG:
        return "log"
    if sampling == protos.Sampling.REVERSE_LOG:
        return "reverse_log"
    msg = (
        "Expected sampling to be one of predefined proto values. "
        f"Received: '{sampling}'."
    )
    raise ValueError(msg)


def sampling_to_proto(sampling):
    if sampling == "linear":
        return protos.Sampling.LINEAR
    if sampling == "log":
        return protos.Sampling.LOG
    if sampling == "reverse_log":
        return protos.Sampling.REVERSE_LOG
    msg = (
        "Expected sampling to be 'linear', 'log', or 'reverse_log'. "
        f"Received: '{sampling}'."
    )
    raise ValueError(msg)


def prob_to_index(prob: float, n_index: float):
    """Convert cumulative probability to 0-based index in the given range."""
    ele_prob = 1 / n_index
    index = int(math.floor(prob / ele_prob))
    # Can happen when `prob` is very close to 1.
    if index == n_index:
        index -= 1
    return index


def index_to_prob(index: float, n_index: float) -> float:
    """Convert 0-based index in the given range to cumulative probability."""
    ele_prob = 1 / n_index
    # Center the value in its probability bucket.
    return (index + 0.5) * ele_prob
