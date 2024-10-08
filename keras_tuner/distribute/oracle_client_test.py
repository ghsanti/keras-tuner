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
"""Tests for distributed tuning."""

import copy
import os
from unittest import mock

import numpy as np
import portpicker
import pytest

import keras_tuner
import keras
from keras_tuner.distribute import oracle_client
from keras_tuner.distribute import utils as dist_utils
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.test_utils import mock_distribute
from keras_tuner.tuners import randomsearch


class SimpleTuner(keras_tuner.engine.base_tuner.BaseTuner):
    def run_trial(self, trial):
        score = self.hypermodel.build(trial.hyperparameters)
        self.save_model(trial.trial_id, score)
        return {"score": score}

    def save_model(self, trial_id, score, step=0):
        save_path = os.path.join(self.project_dir, trial_id)
        with open(save_path, "w") as f:
            f.write(str(score))

    def load_model(self, trial):
        save_path = os.path.join(self.project_dir, trial.trial_id)
        with open(save_path) as f:
            score = int(f.read())
        return score


def test_base_tuner_distribution(tmp_path):
    num_workers = 3

    def _test_base_tuner():
        def build_model(hp):
            return hp.Int("a", 1, 100)

        tuner = SimpleTuner(
            oracle=keras_tuner.oracles.RandomSearchOracle(
                objective=keras_tuner.Objective("score", "max"), max_trials=10
            ),
            hypermodel=build_model,
            directory=tmp_path,
        )
        tuner.search()

        if dist_utils.is_chief_oracle():
            # Model is just a score.
            scores = tuner.get_best_models(10)
            assert len(scores)
            assert scores == sorted(copy.copy(scores), reverse=True)

    mock_distribute.mock_distribute(
        _test_base_tuner, num_workers=num_workers, wait_for_chief=True
    )


def test_random_search(tmp_path):
    # TensorFlow model building and execution is not thread-safe.
    num_workers = 1

    x = np.random.uniform(-1, 1, size=(2, 5))
    y = np.ones((2,))

    def _test_random_search():
        def build_model(hp):
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(hp.Int("num_units", 1, 3), input_shape=(5,))
            )
            model.add(keras.layers.Dense(1))
            model.compile(loss="mse")
            return model

        tuner = keras_tuner.tuners.RandomSearch(
            hypermodel=build_model,
            objective="val_loss",
            max_trials=10,
            directory=tmp_path,
        )
        tuner.search(
            x, y, validation_data=(x, y), epochs=1, batch_size=2, verbose=0
        )

        # Suppress warnings about optimizer state not being restored by
        # tf.keras.

        if dist_utils.is_chief_oracle():
            trials = tuner.oracle.get_best_trials(2)
            assert trials[0].score <= trials[1].score

            models = tuner.get_best_models(2)
            assert models[0].evaluate(x, y, verbose=0) <= models[1].evaluate(
                x, y, verbose=0
            )

    mock_distribute.mock_distribute(
        _test_random_search, num_workers, wait_for_chief=True
    )


def test_client_no_attribute_error():
    with mock.patch.object(os, "environ", mock_distribute.MockEnvVars()):
        port = str(portpicker.pick_unused_port())
        os.environ["KERASTUNER_ORACLE_IP"] = "127.0.0.1"
        os.environ["KERASTUNER_ORACLE_PORT"] = port
        os.environ["KERASTUNER_TUNER_ID"] = "worker0"
        hps = keras_tuner.HyperParameters()
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        client = oracle_client.OracleClient(oracle)
        with pytest.raises(AttributeError, match="has no attribute"):
            client.unknown_attribute


@mock.patch("keras_tuner.distribute.oracle_client.OracleClient.get_space")
def test_should_not_report_update_trial_return_running(get_space):
    get_space.return_value = hp_module.HyperParameters()
    with mock.patch.object(os, "environ", mock_distribute.MockEnvVars()):
        port = str(portpicker.pick_unused_port())
        os.environ["KERASTUNER_ORACLE_IP"] = "127.0.0.1"
        os.environ["KERASTUNER_ORACLE_PORT"] = port
        os.environ["KERASTUNER_TUNER_ID"] = "worker0"
        hps = keras_tuner.HyperParameters()
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        client = oracle_client.OracleClient(oracle)
        client.should_report = False
        assert client.update_trial("a", {"score": 100}).status == "RUNNING"
