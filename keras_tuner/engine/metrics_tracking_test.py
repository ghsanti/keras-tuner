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

import random

import numpy as np
import pytest

from keras import losses, metrics
from keras_tuner.engine import metrics_tracking


def test_register_from_metrics():
    # As well as direction inference.
    tracker = metrics_tracking.MetricsTracker(
        to_register=[metrics.CategoricalAccuracy(), metrics.MeanSquaredError()]
    )
    assert set(tracker.metrics.keys()) == set(
        {
            "CategoricalAccuracy",
            "MeanSquaredError",
        }
    )
    assert tracker.metrics["CategoricalAccuracy"].direction == "max"
    assert tracker.metrics["MeanSquaredError"].direction == "min"


def test_register():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("new_metric", direction="max")
    assert set(tracker.metrics.keys()) == {"new_metric"}
    assert tracker.metrics["new_metric"].direction == "max"
    with pytest.raises(ValueError, match="`direction` should be one of"):
        tracker.register("another_metric", direction="wrong")
    with pytest.raises(ValueError, match="already exists"):
        tracker.register("new_metric", direction="max")


def test_exists():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("new_metric", direction="max")
    assert tracker.exists("new_metric")
    assert not tracker.exists("another_metric")


def test_update():
    tracker = metrics_tracking.MetricsTracker()
    tracker.append_execution_value("new_metric", 0.5)  # automatic registration
    assert set(tracker.metrics.keys()) == {"new_metric"}
    assert tracker.metrics["new_metric"].direction == "min"  # default direction
    assert tracker.get_history("new_metric") == [
        metrics_tracking.ExecutionMetric(0.5)
    ]


def test_metric_observation_repr():
    assert (
        repr(metrics_tracking.ExecutionMetric(0.5))
        == "ExecutionMetric(value=[0.5])"
    )


def test_get_history():
    tracker = metrics_tracking.MetricsTracker()
    # note that each append is a new execution.
    tracker.append_execution_value("new_metric", 0.5)
    tracker.append_execution_value("new_metric", 1.5)
    tracker.append_execution_value("new_metric", 2.0)
    assert tracker.get_history("new_metric") == [
        metrics_tracking.ExecutionMetric(0.5),
        metrics_tracking.ExecutionMetric(1.5),
        metrics_tracking.ExecutionMetric(2.0),
    ]
    with pytest.raises(ValueError, match="Unknown metric"):
        tracker.get_history("another_metric")


def test_set_history_from_values():
    tracker = metrics_tracking.MetricsTracker()
    tracker.set_history_from_values(
        "new_metric",
        [
            [
                0.5,
                1.5,
                2.0,
            ]
        ],
    )
    values = tracker.get_history("new_metric")
    assert values == [
        metrics_tracking.ExecutionMetric([0.5, 1.5, 2.0]),
    ]


def test_set_history():
    tracker = metrics_tracking.MetricsTracker()
    tracker.set_history(
        "new_metric",
        [
            metrics_tracking.ExecutionMetric(
                [
                    0.5,
                    1.5,
                    2.0,
                ]
            )
        ],
    )
    values = tracker.get_history("new_metric")
    assert values == [
        metrics_tracking.ExecutionMetric([0.5, 1.5, 2.0]),
    ]


def test_get_best_step_value_none():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("val_loss", "min")
    assert tracker.get_best_overall_value_location("val_loss") is None


def test_get_best_value():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("metric_min", "min")
    tracker.register("metric_max", "max")
    assert tracker.get_best_overall_value_location("metric_min") is None

    tracker.append_execution_value(
        "metric_min",
        [1.0, 2.0, 3.0],
    )
    tracker.append_execution_value(
        "metric_max",
        [1.0, 2.0, 3.0],
    )
    assert tracker.get_best_overall_value_location("metric_min") == (0, 0)
    assert tracker.get_best_overall_value_location("metric_max") == (0, 2)


def test_get_statistics():
    tracker = metrics_tracking.MetricsTracker()
    history = [random.random() for _ in range(10)]
    tracker.append_execution_value("new_metric", history)
    stats = tracker.get_statistics("new_metric")
    assert stats != None
    assert set(stats.keys()) == {"min", "max", "mean", "median", "var", "std"}
    assert stats["min"] == np.min(history)
    assert stats["max"] == np.max(history)
    assert stats["mean"] == np.mean(history)
    assert stats["median"] == np.median(history)
    assert stats["var"] == np.var(history)
    assert stats["std"] == np.std(history)


def test_get_last_value():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("new_metric", "min")
    assert tracker.get_last_execution_values("new_metric") is None
    tracker.append_execution_value(
        "new_metric",
        [1.0, 2.0, 3.0],
    )
    last_execution_values = tracker.get_last_execution_values("new_metric")
    assert isinstance(last_execution_values, list)
    assert last_execution_values[-1] == 3.0


def test_serialization():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("metric_min", "min")
    tracker.register("metric_max", "max")

    tracker.append_execution_value(
        "metric_min",
        [1.0, 2.0, 3.0],
    )
    tracker.append_execution_value(
        "metric_max",
        [1.0, 2.0, 3.0],
    )
    new_tracker = metrics_tracking.MetricsTracker.from_config(
        tracker.get_config()
    )
    assert new_tracker.metrics.keys() == tracker.metrics.keys()


def test_metric_execution_proto_list():
    obs = metrics_tracking.ExecutionMetric([-10, -20])
    proto = obs.to_proto()
    assert proto.value == [-10.0, -20]
    new_obs = metrics_tracking.ExecutionMetric.from_proto(proto)
    assert new_obs == obs


def test_metric_execution():
    obs = metrics_tracking.ExecutionMetric(-10)
    proto = obs.to_proto()
    assert proto.value == [-10]
    new_obs = metrics_tracking.ExecutionMetric.from_proto(proto)
    assert new_obs == obs


def test_metric_history_proto():
    tracker = metrics_tracking.MetricHistory("max")
    tracker.append_execution_from_values([5])
    tracker.append_execution_from_values([10])

    proto = tracker.to_proto()
    assert proto.direction
    assert proto.executions[0].value == [5]
    assert proto.executions[1].value == [10]

    new_tracker = metrics_tracking.MetricHistory.from_proto(proto)
    assert new_tracker.direction == "max"
    assert new_tracker.get_history() == [
        metrics_tracking.ExecutionMetric(5),
        metrics_tracking.ExecutionMetric(10),
    ]


def test_metrics_tracker_proto():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register("score", direction="max")
    tracker.append_execution_value("score", value=[10, 20])
    tracker.append_execution_value("score", value=30)

    proto = tracker.to_proto()
    executions = proto.metrics["score"].executions
    assert executions[0].value == [10, 20]
    assert executions[1].value == [30]
    assert proto.metrics["score"].direction
    new_tracker = metrics_tracking.MetricsTracker.from_proto(proto)
    assert new_tracker.metrics["score"].direction == "max"
    assert new_tracker.metrics["score"].get_history() == [
        metrics_tracking.ExecutionMetric([10, 20]),
        metrics_tracking.ExecutionMetric(30),
    ]


def test_metric_direction_inference():
    # Test min metrics.
    assert metrics_tracking.infer_metric_direction("MAE") == "min"
    assert (
        metrics_tracking.infer_metric_direction(metrics.binary_crossentropy)
        == "min"
    )
    assert (
        metrics_tracking.infer_metric_direction(metrics.FalsePositives())
        == "min"
    )

    # All losses in keras.losses are considered as 'min'.
    assert metrics_tracking.infer_metric_direction("squared_hinge") == "min"
    assert metrics_tracking.infer_metric_direction(losses.hinge) == "min"
    assert (
        metrics_tracking.infer_metric_direction(
            losses.CategoricalCrossentropy()
        )
        == "min"
    )

    # Test max metrics.
    assert metrics_tracking.infer_metric_direction("binary_accuracy") == "max"
    assert (
        metrics_tracking.infer_metric_direction(metrics.categorical_accuracy)
        == "max"
    )
    assert metrics_tracking.infer_metric_direction(metrics.Precision()) == "max"

    # Test unknown metrics.
    assert metrics_tracking.infer_metric_direction("my_metric") is None

    def my_metric_fn(x, y):
        return x

    assert metrics_tracking.infer_metric_direction(my_metric_fn) is None

    class MyMetric(metrics.Metric):
        def update_state(self, x, y):
            return 1

        def result(self):
            return 1

    assert metrics_tracking.infer_metric_direction(MyMetric()) is None

    # Test special cases.
    assert metrics_tracking.infer_metric_direction("loss") == "min"
    assert metrics_tracking.infer_metric_direction("acc") == "max"
    assert metrics_tracking.infer_metric_direction("val_acc") == "max"
    assert metrics_tracking.infer_metric_direction("crossentropy") == "min"
    assert metrics_tracking.infer_metric_direction("ce") == "min"
    assert metrics_tracking.infer_metric_direction("weighted_acc") == "max"
    assert metrics_tracking.infer_metric_direction("val_weighted_ce") == "min"
    assert (
        metrics_tracking.infer_metric_direction("weighted_binary_accuracy")
        == "max"
    )


def test_get_config():
    # quick test for the shape of the dictionary created.
    tracker_instance = metrics_tracking.MetricsTracker(
        to_register=[metrics.CategoricalAccuracy(), metrics.MeanSquaredError()]
    )
    config = tracker_instance.get_config()
    expected_keys = config.keys()
    assert len(expected_keys) == 2
    for key in config:
        sub_keys = config[key]
        assert len(sub_keys) == 2
        assert "executions" in sub_keys
        assert "direction" in sub_keys
