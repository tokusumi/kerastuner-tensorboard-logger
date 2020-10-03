import pytest
from datetime import timedelta
import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import Hyperband
import tensorflow_datasets as tfds
from sklearn import datasets
from sklearn.model_selection import train_test_split

from kerastuner_tensorboard_logger.logger import timedelta_to_hms
from kerastuner_tensorboard_logger import TensorBoardLogger, setup_tb


def test_timedelta_to_hms():
    td = timedelta(minutes=10, hours=2, seconds=30, microseconds=111)
    out = timedelta_to_hms(td)
    assert out == "2h:10m:30s"


def build_small_model(hp):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


def make_dataset():
    tfds.disable_progress_bar()

    # the data, split between train and test sets
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.take(300)
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.take(300)
    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test


def build_model(hp):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            hp.Choice("filter", values=[16, 32, 64]),
            3,
            activation=hp.Choice("activation", values=["tanh", "softplus", "relu"]),
        )
    )
    model.add(
        keras.layers.Conv2D(
            2 * hp.Choice("filter", values=[16, 32, 64]),
            3,
            activation=hp.Choice("activation", values=["tanh", "softplus", "relu"]),
        )
    )
    if hp.Boolean("batchnorm"):
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            hp.Int("units", min_value=32, max_value=512, step=32), activation="relu"
        )
    )
    model.add(
        keras.layers.Dropout(
            hp.Float("dropout", min_value=0.25, max_value=0.75, step=0.25)
        )
    )
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model


def test_search_manual():
    """e2e test
    manual test is required. log files for tensorboard,
    then, run tensorboard server as:

    ```bash
    tensorboard --logdir tests/logs/hparams
    ```

    """
    tuner = Hyperband(
        build_small_model,
        objective="val_acc",
        max_epochs=5,
        directory="tests/logs/search",
        project_name="search_manual",
        overwrite=True,
        logger=TensorBoardLogger(
            metrics="val_acc", logdir="tests/logs/hparams/search_manual", overwrite=True
        ),
    )

    iris = datasets.load_iris()
    x, val_x, y, val_y = train_test_split(iris.data, iris.target)

    tuner.search(x, y, epochs=5, validation_data=(val_x, val_y))


def test_heavy_search_manual():
    """e2e test
    manual test is required. log files for tensorboard,
    then, run tensorboard server as:

    ```bash
    tensorboard --logdir tests/logs/hparams
    ```

    """
    tuner = Hyperband(
        build_model,
        objective="val_acc",
        max_epochs=5,
        directory="tests/logs/search",
        project_name="heavy_search_manual",
        overwrite=True,
        logger=TensorBoardLogger(
            metrics="val_acc",
            logdir="tests/logs/hparams/heavy_search_manual",
            overwrite=True,
        ),
    )

    train_data, test_data = make_dataset()
    tuner.search(train_data, epochs=5, validation_data=test_data)


def test_initialize_manual():
    """test optional initialization
    manual test is required. log files for tensorboard,
    then, run tensorboard server as:

    ```bash
    tensorboard --logdir tests/logs/hparams
    ```

    """
    tuner = Hyperband(
        build_model,
        objective="val_acc",
        max_epochs=5,
        directory="tests/logs/search",
        project_name="initialize_manual",
        overwrite=True,
        logger=TensorBoardLogger(
            metrics="val_acc",
            logdir="tests/logs/hparams/initialize_manual",
            overwrite=True,
        ),
    )
    setup_tb(tuner)
    train_data, test_data = make_dataset()
    tuner.search(train_data, epochs=3, validation_data=test_data)


def test_parse():
    trained_trial_state = {
        "trial_id": "bb0649bfdb92155d308f12dca83152e1",
        "hyperparameters": {
            "space": [
                {
                    "class_name": "Int",
                    "config": {
                        "name": "units",
                        "default": None,
                        "min_value": 32,
                        "max_value": 512,
                        "step": 32,
                        "sampling": None,
                    },
                },
                {
                    "class_name": "Choice",
                    "config": {
                        "name": "learning_rate",
                        "default": 0.01,
                        "values": [0.01, 0.001, 0.0001],
                        "ordered": True,
                    },
                },
            ],
            "values": {
                "units": 512,
                "learning_rate": 0.001,
                "tuner/epochs": 2,
                "tuner/initial_epoch": 0,
                "tuner/bracket": 1,
                "tuner/round": 0,
            },
        },
        "metrics": {
            "metrics": {
                "loss": {
                    "direction": "min",
                    "observations": [{"value": [1.0315839052200317], "step": 0}],
                },
                "acc": {
                    "direction": "max",
                    "observations": [{"value": [0.6875], "step": 0}],
                },
                "val_loss": {
                    "direction": "min",
                    "observations": [{"value": [0.9468530416488647], "step": 0}],
                },
                "val_acc": {
                    "direction": "max",
                    "observations": [{"value": [0.6052631735801697], "step": 0}],
                },
            }
        },
        "score": None,
        "best_step": None,
        "status": "RUNNING",
    }
    gt_metrics = {
        "loss": 1.0315839052200317,
        "acc": 0.6875,
        "val_loss": 0.9468530416488647,
        "val_acc": 0.6052631735801697,
    }

    metrics = ["val_acc", "val_loss"]
    logger = TensorBoardLogger(metrics=metrics)
    hparams = logger.parse_hparams(trained_trial_state)
    assert hparams == {
        "units": 512,
        "learning_rate": 0.001,
        "tuner/epochs": 2,
        "tuner/initial_epoch": 0,
        "tuner/bracket": 1,
        "tuner/round": 0,
    }
    target_metrics = []
    for target_metric, metric in logger.parse_metrics(trained_trial_state):
        assert metric == gt_metrics.get(target_metric)
        target_metrics.append(target_metric)
    assert set(target_metrics) == set(metrics)