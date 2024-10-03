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
"""Oracle base class."""

import collections
import hashlib
import random
import threading
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from keras_tuner import backend, utils

from . import stateful
from .hyperparameters import HyperParameters
from .objective import (
    MultiObjective,
    Objective,
    create_objective,
)
from .trial import Trial, TrialStatus

if TYPE_CHECKING:
    from keras_tuner.types import (
        _FloatListOrFloat,
        _MetricDirection,
        _SomeObjectiveOrName,
    )
else:
    _FloatListOrFloat = Any
    _SomeObjectiveOrName = Any
    _MetricDirection = Any


# Map each `Oracle` instance to its `Lock`.
LOCKS = collections.defaultdict(lambda: threading.Lock())
# Map each `Oracle` instance to the thread name aquired the `Lock`.
THREADS: dict[str, str | None] = collections.defaultdict(lambda: None)


def synchronized(func, *args, **kwargs):
    """Synchronize the multi-threaded calls to `Oracle` functions.

    In parallel tuning, there may be concurrent gRPC calls from multiple threads
    to the `Oracle` methods like `create_trial()`, `update_trial()`, and
    `end_trial()`. To avoid concurrent writing to the data, use `@synchronized`
    to ensure the calls are synchronized, which only allows one call to run at a
    time.

    Concurrent calls to different `Oracle` objects would not block one another.
    Concurrent calls to the same or different functions of the same `Oracle`
    object would block one another.

    You can decorate a subclass function, which overrides an already decorated
    function in the base class, without worrying about creating a deadlock.
    However, the decorator only support methods within classes, and cannot be
    applied to standalone functions.

    You do not need to decorate `Oracle.populate_space()`, which is only
    called by `Oracle.create_trial()`, which is decorated.

    Example:
    ```py
    class MyOracle(keras_tuner.Oracle):
        @keras_tuner.synchronized
        def create_trial(self, tuner_id):
            super().create_trial(tuner_id)
            ...

        @keras_tuner.synchronized
        def update_trial(self, trial_id, metrics):
            super().update_trial(trial_id, metrics)
            ...

        @keras_tuner.synchronized
        def end_trial(self, trial):
            super().end_trial(trial)
            ...
    ```

    """

    def backward_compatible_end_trial(self, trial_id: str, status):
        trial = Trial(self.get_space(), trial_id, status)
        return [self, trial], {}

    def wrapped_func(*args, **kwargs):
        # For backward compatible with the old end_trial signature:
        # def end_trial(self, trial_id, status="COMPLETED"):
        if func.__name__ == "end_trial" and (
            "trial_id" in kwargs
            or "status" in kwargs
            or isinstance(args[1], str)
        ):
            args, kwargs = backward_compatible_end_trial(*args, **kwargs)

        oracle = args[0]
        thread_name = threading.currentThread().getName()
        need_acquire = THREADS[oracle] != thread_name

        if need_acquire:
            LOCKS[oracle].acquire()
            THREADS[oracle] = thread_name
        ret_val = func(*args, **kwargs)
        if need_acquire:
            THREADS[oracle] = None
            LOCKS[oracle].release()
        return ret_val

    return wrapped_func


class Oracle(stateful.Stateful):
    """Implements a hyperparameter optimization algorithm.

    In a parallel tuning setting, there is only one `Oracle` instance. The
    workers would communicate with the centralized `Oracle` instance with gPRC
    calls to the `Oracle` methods.

    `Trial` objects are often used as the communication packet through the gPRC
    calls to pass information between the worker `Tuner` instances and the
    `Oracle`. For example, `Oracle.create_trial()` returns a `Trial` object, and
    `Oracle.end_trial()` accepts a `Trial` in its arguments.

    New copies of the same `Trial` instance are reconstructed as it going
    through the gRPC calls. The changes to the `Trial` objects in the worker
    `Tuner`s are synced to the original copy in the `Oracle` when they are
    passed back to the `Oracle` by calling `Oracle.end_trial()`.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        seed: Int. Random seed.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are
            invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum
            number of consecutive failed `Trial`s. When this number is reached,
            the search will be stopped. A `Trial` is marked as failed when none
            of the retries succeeded.

    """

    def __init__(
        self,
        objective: _SomeObjectiveOrName
        | list[_SomeObjectiveOrName]
        | None = None,
        max_trials: int | None = None,
        hyperparameters: HyperParameters | None = None,
        allow_new_entries: bool = True,
        tune_new_entries: bool = True,
        seed: int | None = None,
        max_retries_per_trial: int = 0,
        max_consecutive_failed_trials: int = 3,
    ):
        self.objective = create_objective(objective)
        self.max_trials = max_trials
        if not hyperparameters:
            if not tune_new_entries:
                msg = """If you set `tune_new_entries=False`, you must
                    specify the search space via the
                    `hyperparameters` argument."""
                raise ValueError(msg)
            if not allow_new_entries:
                msg = """If you set `allow_new_entries=False`, you must
                    specify the search space via the
                    `hyperparameters` argument."""
                raise ValueError(msg)
            self.hyperparameters = HyperParameters()
        else:
            self.hyperparameters = hyperparameters
        self.allow_new_entries = allow_new_entries
        self.tune_new_entries = tune_new_entries

        # trial_id -> Trial
        self.trials: dict[str, Trial] = {}
        # tuner_id -> Trial
        self.ongoing_trials = {}
        # List of trial_ids in the order of the trials start
        self.start_order = []
        # List of trial_ids in the order of the trials end
        self.end_order = []
        # Map trial_id to failed times
        self._run_times = collections.defaultdict(lambda: 0)
        # Used as a queue of trial_id to retry
        self._retry_queue = []
        # Client Tuner IDs
        self.tuner_ids = set()

        self.seed = seed or random.randint(1, 10000)
        self._seed_state = self.seed
        # Hashes of values in the trials, which only hashes the active values.
        self._tried_so_far = set()
        # Dictionary mapping trial_id to the the hash of the values.
        self._id_to_hash = collections.defaultdict(lambda: None)
        # Maximum number of identical values that can be generated
        # before we consider the space to be exhausted.
        self._max_collisions = 20

        # Set in `BaseTuner` via `set_project_dir`.
        self.directory = None
        self.project_name = None

        # In multi-worker mode, only the chief of each cluster should report
        # results. These 2 attributes exist in `Oracle` just make it consistent
        # with `OracleClient`, in which the attributes are utilized.
        self.multi_worker = False
        self.should_report = True

        # Handling the retries and failed trials.
        self.max_retries_per_trial = max_retries_per_trial
        self.max_consecutive_failed_trials = max_consecutive_failed_trials

        # Print the logs to screen
        self._display = Display(oracle=self)

    @property
    def verbose(self) -> int:
        """Check verbosity of display."""
        return self._display.verbose

    @verbose.setter
    def verbose(self, value: int | Literal["auto"]) -> None:
        if value == "auto":
            value = 1
        self._display.verbose = value

    def _populate_space(self, trial_id: str):
        warnings.warn(
            "The `_populate_space` method is deprecated, "
            "please use `populate_space`.",
            DeprecationWarning,
            stacklevel=1,
        )
        return self.populate_space(trial_id)

    def populate_space(self, trial_id: str):
        """Fill the hyperparameter space with values for a trial.

        This method should be overridden in subclasses and called in
        `create_trial` in order to populate the hyperparameter space with
        values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            should be one of "RUNNING" (the trial can start normally), "IDLE"
            (the oracle is waiting on something and cannot create a trial), or
            "STOPPED" (the oracle has finished searching and no new trial should
            be created).

        """
        raise NotImplementedError

    def _score_trial(self, trial: Trial):
        warnings.warn(
            "The `_score_trial` method is deprecated, "
            "please use `score_trial`.",
            DeprecationWarning,
            stacklevel=1,
        )
        self.score_trial(trial)

    def score_trial(self, trial: Trial) -> None:
        """Score a completed `Trial`.

        This method can be overridden in subclasses to provide a score for
        a set of hyperparameter values. This method is called from `end_trial`
        on completed `Trial`s.

        Args:
            trial: A completed `Trial` object.

        """
        trial.score = trial.metrics.get_best_overall_value(self.objective.name)
        trial.best_step = trial.metrics.get_best_overall_value_location(
            self.objective.name
        )

    @synchronized
    def create_trial(self, tuner_id: str) -> Trial:
        """Create a new `Trial` to be run by the `Tuner`.

        A `Trial` corresponds to a unique set of hyperparameters to be run
        by `Tuner.run_trial`.

        Args:
            tuner_id: A string, the ID that identifies the `Tuner` requesting a
                `Trial`. `Tuners` that should run the same trial (for instance,
                when running a multi-worker model) should have the same ID.

        Returns:
            A `Trial` object containing a set of hyperparameter values to run
            in a `Tuner`.

        """
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        # Record all running client Tuner IDs.
        self.tuner_ids.add(tuner_id)

        # Pick the Trials waiting for retry first.
        if len(self._retry_queue) > 0:
            trial = self.trials[self._retry_queue.pop()]
            trial.status = TrialStatus.RUNNING
            self.ongoing_trials[tuner_id] = trial
            self.save()
            self._display.on_trial_begin(trial)
            return trial

        # Make the trial_id the current number of trial, pre-padded with 0s
        trial_id = f"{{:0{len(str(self.max_trials))}d}}"
        trial_id = trial_id.format(len(self.trials))

        if self.max_trials and len(self.trials) >= self.max_trials:
            status = TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response["status"]
            values = response.get("values", None)

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}

        trial = Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == TrialStatus.RUNNING:
            # Record the populated values (active only). Only record when the
            # status is RUNNING. If other status, the trial will not run, the
            # values are discarded and should not be recorded, in which case,
            # the trial_id may appear again in the future.
            self._record_values(trial)

            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self.start_order.append(trial_id)
            self._save_trial(trial)
            self.save()
            self._display.on_trial_begin(trial)

        # Remove the client Tuner ID when triggered the client to exit
        if status == TrialStatus.STOPPED:
            self.tuner_ids.remove(tuner_id)

        return trial

    @synchronized
    def update_trial(
        self,
        trial_id: str,
        metrics: (
            list[dict[str, _FloatListOrFloat]] | dict[str, _FloatListOrFloat]
        ),
    ) -> Trial:
        """Report the status of a trial (by worker).

        Args:
            trial_id: a previously seen trial id.
            metrics: Each dict's keys are metric names, and the values
            are the executions' metric values.

        Returns:
            Trial object.

        """
        trial: Trial = self.trials[trial_id]
        if not isinstance(metrics, list):
            metrics = [metrics]
        for metric_exec in metrics:
            self._check_objective_found(metric_exec)
            for metric_name, metric_value in metric_exec.items():
                if not trial.metrics.exists(metric_name):
                    direction = _maybe_infer_direction_from_objective(
                        self.objective, metric_name
                    )
                    trial.metrics.register(metric_name, direction=direction)
                trial.metrics.append_execution(metric_name, metric_value)
        # TODO: averaging requests by user should occur here, before saving.
        self._save_trial(trial)
        # TODO: To signal early stopping, set Trial.status to "STOPPED".
        return trial

    def _check_consecutive_failures(self):
        # For thread safety, check all trials for consecutive failures.
        consecutive_failures = 0
        for trial_id in self.end_order:
            trial = self.trials[trial_id]
            if trial.status == TrialStatus.FAILED:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if consecutive_failures == self.max_consecutive_failed_trials:
                raise RuntimeError(
                    "Number of consecutive failures exceeded the limit "
                    f"of {self.max_consecutive_failed_trials}.\n"
                    + (trial.message or "")
                )

    @synchronized
    def end_trial(self, trial: Trial):
        """Logistics when a `Trial` finished running.

        Record the `Trial` information and end the trial or send it for retry.

        Args:
            trial: The Trial to be ended. `trial.status` should be one of
                `"COMPLETED"` (the trial finished normally), `"INVALID"` (the
                trial has crashed or been deemed infeasible, but subject to
                retries), or `"FAILED"` (The Trial is failed. No more retries
                needed.). `trial.message` is an optional string, which is the
                error message if the trial status is `"INVALID"` or `"FAILED"`.

        """
        # To support parallel tuning, the information in the `trial` argument is
        # synced back to the `Oracle`. Update the self.trials with the given
        # trial.
        old_trial = self.trials[trial.trial_id]
        old_trial.hyperparameters = trial.hyperparameters
        old_trial.status = trial.status
        old_trial.message = trial.message
        trial = old_trial

        self.update_space(trial.hyperparameters)
        if trial.status == TrialStatus.COMPLETED:
            self.score_trial(trial)

            if trial.score is None or np.isnan(trial.score):
                trial.status = TrialStatus.INVALID

        # Record the values again in case of new hps appeared.
        self._record_values(trial)

        self._run_times[trial.trial_id] += 1

        # Check if need to retry the trial.
        if not self._retry(trial):
            self.end_order.append(trial.trial_id)
            self._check_consecutive_failures()

        self._save_trial(trial)
        self.save()

        self._display.on_trial_end(trial)

        # Pop the ongoing trial at last, which would notify the chief server to
        # stop when ongoing_trials is empty.
        for tuner_id, ongoing_trial in self.ongoing_trials.items():
            if ongoing_trial.trial_id == trial.trial_id:
                self.ongoing_trials.pop(tuner_id)
                break

    def _retry(self, trial: Trial) -> bool:
        """Send the trial for retry if needed.

        Args:
            trial: Trial. The trial to check.

        Returns:
            Boolean. Whether the trial should be retried.

        """
        if trial.status != TrialStatus.INVALID:
            return False

        trial_id = trial.trial_id
        max_run_times = self.max_retries_per_trial + 1

        if self._run_times[trial_id] >= max_run_times:
            trial.status = TrialStatus.FAILED
            return False

        print(
            f"Trial {trial_id} failed {self._run_times[trial_id]} "
            "times. "
            f"{max_run_times - self._run_times[trial_id]} "
            "retries left."
        )
        self._retry_queue.append(trial_id)
        return True

    def get_space(self):
        """Return the `HyperParameters` search space."""
        return self.hyperparameters.copy()

    def update_space(self, hyperparameters):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            hyperparameters: An updated `HyperParameters` object.

        """
        hps = hyperparameters.space
        new_hps = [
            hp
            for hp in hps
            if not self.hyperparameters._exists(hp.name, hp.conditions)
        ]

        if new_hps and not self.allow_new_entries:
            msg = (
                "`allow_new_entries` is `False`, "
                f"but found new entries {new_hps}"
            )
            raise RuntimeError(msg)
        if not self.tune_new_entries:
            # New entries should always use the default value.
            return
        self.hyperparameters.merge(new_hps)

    def get_trial(self, trial_id: str) -> Trial:
        """Return the `Trial` specified by `trial_id`."""
        return self.trials[trial_id]

    def get_best_trials(self, num_trials: int = 1):
        """Return the best `Trial`s."""
        # completed trials.
        trials = [
            t
            for t in self.trials.values()
            if t.status == TrialStatus.COMPLETED and t.score is not None
        ]
        sorted_trials = sorted(
            trials,
            key=lambda trial: trial.score
            or 0,  # note that it's defined from above!
            reverse=(self.objective.direction == "max"),
        )

        if len(sorted_trials) < num_trials:
            sorted_trials = sorted_trials + [
                t
                for t in self.trials.values()
                if t.status != TrialStatus.COMPLETED
            ]
        return sorted_trials[:num_trials]

    def remaining_trials(self) -> int | None:
        """Get the number of remaining trails."""
        return (
            self.max_trials - len(self.trials.items())
            if self.max_trials
            else None
        )

    def get_state(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        # Just save the IDs for ongoing trials, since these are in `trials`.
        return {
            "ongoing_trials": {
                tuner_id: trial.trial_id
                for tuner_id, trial in self.ongoing_trials.items()
            },
            # Hyperparameters are part of the state because they can be added to
            # during the course of the search.
            "hyperparameters": self.hyperparameters.get_config(),
            "start_order": self.start_order,
            "end_order": self.end_order,
            "retries": {
                trial_id: int(value) - 1
                for trial_id, value in self._run_times.items()
            },
            "retry_queue": self._retry_queue,
            "seed": self.seed,
            "seed_state": self._seed_state,
            "tried_so_far": list(self._tried_so_far),
            "id_to_hash": self._id_to_hash,
            "display": self._display.get_state(),
        }

    def set_state(self, state):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        self.ongoing_trials = {
            tuner_id: self.trials[trial_id]
            for tuner_id, trial_id in state["ongoing_trials"].items()
        }
        self.hyperparameters = HyperParameters.from_config(
            state["hyperparameters"]
        )
        self.start_order = state["start_order"]
        self.end_order = state["end_order"]
        self._run_times = collections.defaultdict(lambda: 0)
        self._run_times.update(
            {
                trial_id: int(value) + 1
                for trial_id, value in state["retries"].items()
            },
        )
        self._retry_queue = state["retry_queue"]
        self.seed = state["seed"]
        self._seed_state = state["seed_state"]
        self._tried_so_far = set(state["tried_so_far"])
        self._id_to_hash = collections.defaultdict(lambda: None)
        self._id_to_hash.update(state["id_to_hash"])
        self._display.set_state(state["display"])

    def _set_project_dir(self, directory: Path, project_name: str) -> None:
        """Set the project directory and reloads the Oracle."""
        self._directory = Path(directory)
        self._project_name = project_name

    @property
    def _project_dir(self) -> Path:
        """Get the top level directory where we store results."""
        dirname = Path(self._directory, self._project_name)
        utils.create_directory(dirname)
        return dirname

    def save(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        super().save(self._get_oracle_fname())

    def reload(self):
        # Reload trials from their own files.
        trial_fnames = backend.io.glob(
            str(Path(self._project_dir, "trial_*", "trial.json"))
        )
        for fname in trial_fnames:
            trial = Trial.load(fname)
            self.trials[trial.trial_id] = trial
        try:
            super().reload(self._get_oracle_fname())
        except KeyError as e:
            msg = f"""Error reloading `Oracle` from existing project.
            If you did not mean to reload from an existing project,
            change the `project_name` or pass `overwrite=True`
            when creating the `Tuner`. Found existing
            project at: {self._project_dir}"""
            raise RuntimeError(msg) from e

        # Empty the ongoing_trials and send them for retry.
        for trial in self.ongoing_trials.values():
            self._retry_queue.append(trial.trial_id)
        self.ongoing_trials = {}

    def _get_oracle_fname(self):
        return Path(self._project_dir, "oracle.json")

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = "".join(f"{k!s}={values[k]!s}" for k in keys)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def _check_objective_found(self, metrics: dict):
        # objective must be a subset of metrics
        if isinstance(self.objective, MultiObjective):
            # list of names from dict
            objective_names = list(self.objective.name_to_direction.keys())
        else:  # single name to list
            if not self.objective or not self.objective.name:
                msg = "No objective name found. Did you define one?"
                raise ValueError(msg)
            objective_names = [self.objective.name]
        # only leave objectives that are not in metrics.
        objective_names = [obj for obj in objective_names if obj not in metrics]
        if objective_names:
            msg = f"""Objective value missing in metrics reported to
                the Oracle, expected: {objective_names},
                found: {metrics.keys()}"""
            raise ValueError(msg)

    def _get_trial_dir(self, trial_id: str):
        dirname = str(Path(self._project_dir, f"trial_{trial_id!s}"))
        utils.create_directory(dirname)
        return dirname

    def _save_trial(self, trial: Trial):
        # Write trial status to trial directory
        trial_id = trial.trial_id
        trial.save(str(Path(self._get_trial_dir(trial_id), "trial.json")))

    def _random_values(self):
        """Fill the hyperparameter space with random values.

        Returns:
            A dictionary mapping hyperparameter names to suggested values.

        """
        collisions = 0
        while collisions < self._max_collisions:
            hps = HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                if hps.is_active(hp):  # Only active params in `values`.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            if self._duplicate(hps.values):
                collisions += 1
                continue
            return hps.values
        return None

    def _duplicate(self, values):
        """Check if the values has been tried in previous trials.

        Args:
            A dictionary mapping hyperparameter names to suggested values.

        Returns:
            Boolean. Whether the values has been tried in previous trials.

        """
        return self._compute_values_hash(values) in self._tried_so_far

    def _record_values(self, trial: Trial):
        hyperparameters = trial.hyperparameters
        hyperparameters.ensure_active_values()
        new_hash_value = self._compute_values_hash(hyperparameters.values)
        self._tried_so_far.add(new_hash_value)

        # In case of new hp appeared, remove the old hash value.
        old_hash_value = self._id_to_hash[trial.trial_id]
        if old_hash_value != new_hash_value:
            self._id_to_hash[trial.trial_id] = new_hash_value
            # Check before removing. If this is a retry run, the old value may
            # have been removed already.
            if old_hash_value in self._tried_so_far:
                self._tried_so_far.remove(old_hash_value)


# TODO: Add more extensive display.
class Display(stateful.Stateful):
    def __init__(self, oracle: Oracle, verbose: int = 1):
        self.verbose = verbose
        self.oracle = oracle
        self.col_width = 18

        # Start time for the overall search
        self.search_start = None

        # Start time of the trials
        # {trial_id: start_time}
        self.trial_start = {}
        # Trial number of the trials, starting from #1.
        # {trial_id: trial_number}
        self.trial_number = {}

    def get_state(self):
        return {
            "search_start": (
                self.search_start.isoformat()
                if self.search_start is not None
                else self.search_start
            ),
            "trial_start": {
                key: value.isoformat()
                for key, value in self.trial_start.items()
            },
            "trial_number": self.trial_number,
        }

    def set_state(self, state):
        self.search_start = (
            datetime.fromisoformat(state["search_start"])
            if state["search_start"] is not None
            else state["search_start"]
        )
        self.trial_start = {
            key: datetime.fromisoformat(value)
            for key, value in state["trial_start"].items()
        }

        self.trial_number = state["trial_number"]

    def on_trial_begin(self, trial: Trial) -> None:
        """Set initial time and print hyperparameters' table.

        If unset, it sets the "overall search time" as well.
        """
        if self.verbose < 1:
            return

        start_time = datetime.now(tz=timezone.utc)
        self.trial_start[trial.trial_id] = start_time
        if self.search_start is None:  # overall search time, not Trial's.
            self.search_start = start_time
        current_number = len(self.oracle.trials)
        self.trial_number[trial.trial_id] = current_number

        print()
        print(f"Search: Running Trial #{current_number}")
        print()
        self.show_hyperparameter_table(trial)
        print()

    def on_trial_end(self, trial: Trial) -> None:
        """Print trial_id, time_taken, best_score."""
        if self.verbose < 1:
            return

        utils.try_clear()

        time_taken_str = self.format_duration(
            datetime.now(tz=timezone.utc) - self.trial_start[trial.trial_id]
        )
        print(
            f"Trial {self.trial_number[trial.trial_id]} "
            f"Complete [{time_taken_str}]"
        )

        if trial.score is not None:
            print(f"{self.oracle.objective.name}: {trial.score}")

        print()
        best_trials = self.oracle.get_best_trials()
        best_score = best_trials[0].score if len(best_trials) > 0 else None
        print(f"Best {self.oracle.objective.name} So Far: {best_score}")

        # avoid raising an error since it would bust the process.
        if self.search_start is None:
            msg = """Found Display.search_time=None, then subtraction isn't
            defined. You will see an 'Unknown' elapsed time."""
            warnings.warn(msg, stacklevel=1)
            print("Total elapsed time: Unknown")
        else:
            time_elapsed_str = self.format_duration(
                datetime.now(tz=timezone.utc) - self.search_start
            )
            print(f"Total elapsed time: {time_elapsed_str}")

    def show_hyperparameter_table(self, trial: Trial) -> None:
        """Summary table with parameters and best values."""
        template = f"{{0:{self.col_width}}}|{{1:{self.col_width}}}|{{2}}"
        best_trials = self.oracle.get_best_trials()
        best_trial = best_trials[0] if len(best_trials) > 0 else None
        if trial.hyperparameters.values:
            print(
                template.format("Value", "Best Value So Far", "Hyperparameter")
            )
            for hp, value in trial.hyperparameters.values.items():
                best_value = (
                    best_trial.hyperparameters.values.get(hp)
                    if best_trial
                    else "?"
                )
                print(
                    template.format(
                        self.format_value(value),
                        self.format_value(best_value),
                        hp,
                    )
                )
        else:
            print("default configuration")

    def format_value(self, val: str | float) -> str:
        """Format `val` to specific width."""
        if isinstance(val, (int | float)) and not isinstance(val, bool):
            return f"{val:.5g}"
        val_str = str(val)
        if len(val_str) > self.col_width:
            val_str = f"{val_str[:self.col_width - 3]}..."
        return val_str

    def format_duration(self, dt: timedelta) -> str:
        """Format the delta time in d-h-m-s."""
        s = round(dt.total_seconds())
        d, r = divmod(s, 86400)
        h, r = divmod(r, 3600)
        m, r = divmod(r, 60)
        s, r = divmod(r, 60)

        if d > 0:
            return f"{d:d}d {h:02d}h {m:02d}m {s:02d}s"
        return f"{h:02d}h {m:02d}m {s:02d}s"


def _maybe_infer_direction_from_objective(
    objective: Objective | list[Objective], metric_name: str
) -> _MetricDirection | None:
    obj_list = [objective] if isinstance(objective, Objective) else objective
    return next(
        (
            cast(_MetricDirection, obj.direction)
            for obj in obj_list
            if obj.name == metric_name
        ),
        None,
    )
