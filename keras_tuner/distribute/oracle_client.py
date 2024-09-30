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
"""OracleClient class."""

import os
from typing import TYPE_CHECKING, Any

import grpc

from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine.oracle import Oracle
from keras_tuner.protos import keras_tuner_pb2_grpc as grpc_service
from keras_tuner.protos import service_pb2 as service

if TYPE_CHECKING:
    from keras_tuner.engine.trial import Trial
else:
    Trial = Any

# The timeout is so high to prevent a rare race condition from happening.
# We need clients to wait till chief oracle server starts. This normally takes
# a few minutes, but sometimes might take longer.
# See https://github.com/keras-team/keras-tuner/issues/990 for more details.
# Initially we didn't have any timeout. It was introduced to avoid tuner jobs
# hanging forever if chief oracle stops responding.
# See https://github.com/keras-team/keras-tuner/pull/957.
TIMEOUT = 60 * 60  # 60 mins


class OracleClient:
    """Wraps an `Oracle` on a worker to send requests to the chief."""

    def __init__(self, oracle: Oracle):
        self._oracle = oracle

        ip_addr = os.environ["KERASTUNER_ORACLE_IP"]
        port = os.environ["KERASTUNER_ORACLE_PORT"]
        channel = grpc.insecure_channel(f"{ip_addr}:{port}")
        self.stub = grpc_service.OracleStub(channel)
        self.tuner_id = os.environ["KERASTUNER_TUNER_ID"]

        # In multi-worker mode, only the chief of each cluster should report
        # results to the chief Oracle.
        self.multi_worker = False
        self.should_report = True

    def __getattr__(self, name: str) -> object:
        """Getter for underlying Oracle."""
        whitelisted_attrs = {
            "objective",
            "max_trials",
            "allow_new_entries",
            "tune_new_entries",
        }
        if name in whitelisted_attrs:
            return getattr(self._oracle, name)
        msg = f'`OracleClient` object has no attribute "{name}"'
        raise AttributeError(msg)

    def get_space(self) -> HyperParameters:
        """Load the Hyperparameters from Proto in distributed training."""
        response = self.stub.GetSpace(
            service.GetSpaceRequest(),
            wait_for_ready=True,
            timeout=TIMEOUT,
        )
        return HyperParameters.from_proto(response.hyperparameters)

    def update_space(self, hyperparameters: HyperParameters) -> None:
        """Update the Hyperparameters from Proto in distributed training."""
        if self.should_report:
            self.stub.UpdateSpace(
                service.UpdateSpaceRequest(
                    hyperparameters=hyperparameters.to_proto()
                ),
                wait_for_ready=True,
                timeout=TIMEOUT,
            )

    def create_trial(self, tuner_id: str) -> Trial:
        """Create trial in distributed mode."""
        response = self.stub.CreateTrial(
            service.CreateTrialRequest(tuner_id=tuner_id),
            wait_for_ready=True,
            timeout=TIMEOUT,
        )
        return Trial.from_proto(response.trial)

    def update_trial(self, trial_id: str, metrics) -> Trial:
        """Update trial in distributed mode."""
        # TODO: support early stopping in multi-worker.
        if self.should_report:
            response = self.stub.UpdateTrial(
                service.UpdateTrialRequest(trial_id=trial_id, metrics=metrics),
                wait_for_ready=True,
                timeout=TIMEOUT,
            )
            if not self.multi_worker:
                return Trial.from_proto(response.trial)
        return Trial(self.get_space(), status="RUNNING")

    def end_trial(self, trial: Trial) -> None:
        """End `trial` from distributed computation."""
        if self.should_report:
            self.stub.EndTrial(
                service.EndTrialRequest(trial=trial.to_proto()),
                wait_for_ready=True,
                timeout=TIMEOUT,
            )

    def get_trial(self, trial_id: str) -> Trial:
        """Get Trial with `trial_id` from distributed computation."""
        response = self.stub.GetTrial(
            service.GetTrialRequest(trial_id=trial_id),
            wait_for_ready=True,
            timeout=TIMEOUT,
        )
        return Trial.from_proto(response.trial)

    def get_best_trials(self, num_trials: int = 1) -> list[Trial]:
        """Get best Trials from distributed computation."""
        response = self.stub.GetBestTrials(
            service.GetBestTrialsRequest(num_trials=num_trials),
            wait_for_ready=True,
            timeout=TIMEOUT,
        )
        return [Trial.from_proto(trial) for trial in response.trials]
