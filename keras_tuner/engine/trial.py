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
"""Trial class."""

import hashlib
import random
import time
from pathlib import Path
from typing import cast

from keras_tuner import utils
from keras_tuner.protos import keras_tuner_pb2 as protos

from . import metrics_tracking, stateful
from .hyperparameters.HyperParameters import HyperParameters


class TrialStatus:
    # The Trial may start to run.
    RUNNING = "RUNNING"
    # The Trial is empty. The Oracle is waiting on something else before
    # creating the trial. Should call Oracle.create_trial() again.
    IDLE = "IDLE"
    # The Trial has crashed or been deemed infeasible for the current run, but
    # subject to retries.
    INVALID = "INVALID"
    # The Trial is empty. Oracle finished searching. No new trial needed. The
    # tuner should also end the search.
    STOPPED = "STOPPED"
    # The Trial finished normally.
    COMPLETED = "COMPLETED"
    # The Trial is failed. No more retries needed.
    FAILED = "FAILED"

    @staticmethod
    def to_proto(status):
        ts = protos.TrialStatus
        if status is None:
            return ts.UNKNOWN
        if status == TrialStatus.RUNNING:
            return ts.RUNNING
        if status == TrialStatus.IDLE:
            return ts.IDLE
        if status == TrialStatus.INVALID:
            return ts.INVALID
        if status == TrialStatus.STOPPED:
            return ts.STOPPED
        if status == TrialStatus.COMPLETED:
            return ts.COMPLETED
        if status == TrialStatus.FAILED:
            return ts.FAILED
        msg = f"Unknown status {status}"
        raise ValueError(msg)

    @staticmethod
    def from_proto(proto):
        ts = protos.TrialStatus
        if proto == ts.UNKNOWN:
            return None
        if proto == ts.RUNNING:
            return TrialStatus.RUNNING
        if proto == ts.IDLE:
            return TrialStatus.IDLE
        if proto == ts.INVALID:
            return TrialStatus.INVALID
        if proto == ts.STOPPED:
            return TrialStatus.STOPPED
        if proto == ts.COMPLETED:
            return TrialStatus.COMPLETED
        if proto == ts.FAILED:
            return TrialStatus.FAILED
        msg = f"Unknown status {proto}"
        raise ValueError(msg)


class Trial(stateful.Stateful):
    """The runs with the same set of hyperparameter values.

    `Trial` objects are managed by the `Oracle`. A `Trial` object contains all
    the information related to the executions with the same set of
    hyperparameter values. A `Trial` may be executed multiple times for more
    accurate results or for retrying when failed. The related information
    includes hyperparameter values, the Trial ID, and the trial results.

    Args:
        hyperparameters: HyperParameters. It contains the hyperparameter values
            for the trial.
        trial_id: String. The unique identifier for a trial.
        status: one of the TrialStatus attributes. It marks the current status
            of the Trial.
        message: String. The error message if the trial status is "INVALID".

    """

    def __init__(
        self,
        hyperparameters: HyperParameters | None,
        trial_id: str | None = None,
        status: str | None = TrialStatus.RUNNING,
        message: str | None = None,
    ):
        self.hyperparameters = hyperparameters
        self.trial_id: str = (
            generate_trial_id() if trial_id is None else trial_id
        )

        self.metrics = metrics_tracking.MetricsTracker()
        self.score: float | None = None
        self.best_step: tuple[int, int] | None = None
        self.status = status
        self.message = message

    def summary(self) -> None:
        """Display hyperparameters, score and any messages."""
        print(f"Trial {self.trial_id} summary")

        print("Hyperparameters:")
        self.display_hyperparameters()

        if self.score is not None:
            print(f"Score: {self.score}")

        if self.message is not None:
            print(self.message)

    def display_hyperparameters(self) -> None:
        """Print HP-values to user."""
        if self.hyperparameters is None:
            msg = "HyperParameters should be defined but found None."
            raise TypeError(msg)
        if self.hyperparameters.values:
            for hp, value in self.hyperparameters.values.items():
                print(f"{hp}:", value)
        else:
            print("default configuration")

    def get_state(self):
        if self.hyperparameters is None:
            msg = "HyperParameters should be defined but found None."
            raise TypeError(msg)
        return {
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters.get_config(),
            "metrics": self.metrics.get_config(),
            "score": self.score,
            "best_step": self.best_step,
            "status": self.status,
            "message": self.message,
        }

    def set_state(self, state):
        self.trial_id = state["trial_id"]
        hp = HyperParameters.from_config(state["hyperparameters"])
        self.hyperparameters = hp
        self.metrics = metrics_tracking.MetricsTracker.from_config(
            state["metrics"]
        )
        self.score = state["score"]
        self.best_step = state["best_step"]
        self.status = state["status"]
        self.message = state["message"]

    @classmethod
    def from_state(cls, state):
        trial = cls(hyperparameters=None)
        trial.set_state(state)
        return trial

    @classmethod
    def load(cls: type["Trial"], fname: Path) -> "Trial":
        """Load the Trial from the json-configuration file."""
        return cls.from_state(utils.load_json(fname))

    def to_proto(self):
        is_score = self.score is not None
        tuple_length = 2
        is_location = (
            isinstance(self.best_step, tuple)
            and len(self.best_step) == tuple_length
        )
        if is_score and is_location:
            best_step = cast(tuple, self.best_step)
            score = protos.Trial.Score(
                value=self.score,
                step=protos.Trial.Score.Step(
                    exec_idx=best_step[0], epoch_idx=best_step[1]
                ),
            )
        else:
            score = None
        return protos.Trial(
            trial_id=self.trial_id,
            hyperparameters=self.hyperparameters.to_proto(),
            score=score,
            status=TrialStatus.to_proto(self.status),
            metrics=self.metrics.to_proto(),
        )

    @classmethod
    def from_proto(cls, proto):
        instance = cls(
            HyperParameters.from_proto(proto.hyperparameters),
            trial_id=proto.trial_id,
            status=TrialStatus.from_proto(proto.status),
        )
        if proto.HasField("score"):
            instance.score = proto.score.value
            instance.best_step = proto.score.step
        instance.metrics = metrics_tracking.MetricsTracker.from_proto(
            proto.metrics
        )
        return instance


def generate_trial_id() -> str:
    """Hash-like ID to identify the trial."""
    s = str(time.time()) + str(random.randint(1, int(1e7)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
