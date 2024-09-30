import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
from keras import layers

import keras_tuner as kt

print(keras.backend.backend())
rng = np.random.default_rng()
X = rng.standard_normal((200, 32, 32, 3))
y = rng.integers(size=(200, 1), low=0, high=10)
"""## Model and Search Space Definition"""


def build_model(hp: kt.HyperParameters):
    model = keras.Sequential()
    model.add(layers.Input((32, 32, 3)))
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=10,
            activation="gelu",
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float(
        "lr", min_value=1e-4, max_value=1e-2, sampling="log"
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        run_eagerly=True,
    )
    return model


"""## Define the Tuner Class"""

# max_trials n of HPs combinations to try.
# executions tries w same HPs. (use =1 for only one configuration.)
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="accuracy",  # must match metrics.
    max_trials=10,
    executions_per_trial=3,
    overwrite=True,
    directory="./example-results",
    project_name="test",
)

tuner.search_space_summary()

tuner.search(
    X,
    y,
    epochs=3,
    # tensorboard not working currently, hopefully soon.
    # callbacks=[keras.callbacks.TensorBoard()],
    verbose=2,
)
