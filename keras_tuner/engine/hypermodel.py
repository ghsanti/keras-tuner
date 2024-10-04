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
"""HyperModel base class.

`get_hypermodel` function is defined.

It returns a `HyperModel|None`, either by:
    * passing the HyperModel,
    * passing a Callable that takes `hps` and returns a _Model.

The second case the HyperModel is created automatically.
"""

from typing import TYPE_CHECKING, Any, Protocol

from keras_tuner import errors
from keras_tuner.types import _FloatListOrFloat, _Model

if TYPE_CHECKING:
    from keras.api.callbacks import History

    from .hyperparameters.HyperParameters import HyperParameters
else:
    HyperParameters = Any
    History = Any


class HyperModel:
    """Defines a search space of models.

    Args:
        name: Optional string, the name of this HyperModel.
        tunable: Boolean, whether the hyperparameters defined in this
            hypermodel should be added to search space. If `False`, either the
            search space for these parameters must be defined in advance, or
            the default values will be used. Defaults to True.

    Examples:
    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp):  # override `build`
            model = keras.Sequential()
            model.add(
                keras.layers.Dense(
                    hp.Choice("units", [8, 16, 32]), activation="relu"
                )
            )
            model.add(keras.layers.Dense(1, activation="relu"))
            model.compile(loss="mse")
            return model  # return model
    ```

    Optionally override `.fit`:

    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp): ...

        def fit(self, hp, model, *args, **kwargs):
            return model.fit(*args, epochs=hp.Int("epochs", 5, 20), **kwargs)
    ```

    If you have a customized training process, you can return the objective
    value as a float.

    If you want to keep track of more metrics, you can return a dictionary of
    the metrics to track.

    ```python
    class MyHyperModel(kt.HyperModel):
        def build(self, hp): ...

        def fit(self, hp, model, *args, **kwargs):
            ...
            return {
                "loss": loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
    ```


    """

    def __init__(self, name: str | None = None, *, tunable: bool = True):
        self.name = name
        self.tunable = tunable

        self._build = self.build
        self.build = self._build_wrapper

    def build(self, hp: HyperParameters) -> _Model:
        """Build a model. This must be implemented when subclassing.

        Args:
            hp: A `HyperParameters` instance.

        Returns:
            A model instance.

        """
        raise NotImplementedError

    def _build_wrapper(self, hp: HyperParameters, *args, **kwargs) -> _Model:
        """User passes a build method, we use the wrapped."""
        if not self.tunable:
            # Copy `HyperParameters` object so that new entries are not added
            # to the search space.
            hp = hp.copy()
        return self._build(hp, *args, **kwargs)

    def declare_hyperparameters(self, hp: HyperParameters):
        pass

    def fit(
        self, hp: HyperParameters, model: _Model, *args, **kwargs
    ) -> History | dict[str, _FloatListOrFloat] | float:
        """Train the model. This can optionally overridden, or use default.

        Args:
            hp: HyperParameters.
            model: `keras.Model` built in the `build()` function.
            **kwargs: All arguments passed to `Tuner.search()` are in the
                `kwargs` here. It always contains a `callbacks` argument, which
                is a list of default Keras callback functions for model
                checkpointing, Tensorboard configuration, and other tuning
                utilities. If `callbacks` is passed by the user from
                `Tuner.search()`, these default callbacks will be appended to
                the user provided list.

        Returns:
            A `History` object, which is the return value of `model.fit()`, a
            dictionary, or a float.

            If return a dictionary, it should be a dictionary of the metrics to
            track. The keys are the metric names, which contains the
            `objective` name. The values should be the metric values.

            If return a float, it should be the `objective` value.

        """
        return model.fit(*args, **kwargs)


class BuildType(Protocol):
    """Call method for HyperModel.build."""

    def __call__(self, hp: HyperParameters) -> _Model:
        """Call signature."""
        raise NotImplementedError


class DefaultHyperModel(HyperModel):
    """Produces HyperModel from a model building function.

    This is a simple way so that a user passes just `build`
    returning a model to the `Tuner.search()`
    """

    def __init__(
        self,
        build: BuildType,
        name: str | None = None,
        *,
        tunable: bool = True,
    ):
        super().__init__(name=name, tunable=tunable)
        self.build = build


def get_hypermodel(
    hypermodel: HyperModel | BuildType | None,
) -> HyperModel:
    """Get a HyperModel from a HyperModel or callable.

    This is just a simple interface function.
    """
    if hypermodel is None:
        return None
    if isinstance(hypermodel, HyperModel):
        return hypermodel

    if not callable(hypermodel):
        msg = (
            "The `hypermodel` argument should be either "
            "a callable with signature `build(hp)` returning a model, "
            "or an instance of `HyperModel`."
        )
        raise errors.FatalValueError(msg)
    return DefaultHyperModel(hypermodel)
